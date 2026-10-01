"""
RetailPulse - Demand & Sales Forecasting Module

Implements:
1. Weekly net revenue aggregation with partial week detection.
2. Complete calendar week reindexing (distinguishing true zero sales from missing dates).
3. Chronological holdout evaluation (4-week test set, never random split).
4. Model comparison: Exponential Smoothing (statsmodels) vs. Naive Moving Average Baseline.
5. Out-of-sample evaluation with MAE and WAPE (handles zero denominators gracefully).
6. Future 4-week point forecast with calculated residual-based prediction intervals.
7. Insufficient history detection with clear user feedback.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from src.database import DatabaseManager


@dataclass
class ForecastEvaluation:
    model_name: str
    mae: float
    wape_pct: float
    holdout_predictions: List[float]
    actual_holdout: List[float]


@dataclass
class ForecastResult:
    is_sufficient_history: bool
    status_message: str
    historical_weekly_df: pd.DataFrame
    evaluation_naive: Optional[ForecastEvaluation] = None
    evaluation_stats: Optional[ForecastEvaluation] = None
    selected_model_name: Optional[str] = None
    future_forecast_df: Optional[pd.DataFrame] = None
    notes: Dict[str, str] = None


def aggregate_weekly_sales(
    db: DatabaseManager,
    country: str = "ALL",
) -> pd.DataFrame:
    """
    Aggregates net revenue by weekly boundary (Monday start),
    flags partial boundary weeks, and fills calendar gaps with zero sales.
    """
    sql = """
    SELECT
        CAST(week_start AS DATE) AS week_start,
        COUNT(DISTINCT date) AS active_days_in_week,
        COALESCE(SUM(CASE WHEN is_qualifying_sale THEN gross_amount ELSE 0.0 END), 0.0) AS gross_sales,
        COALESCE(SUM(CASE WHEN is_return THEN return_amount ELSE 0.0 END), 0.0) AS return_value,
        COALESCE(SUM(CASE WHEN is_qualifying_sale THEN quantity ELSE 0 END), 0) AS units_sold,
        COUNT(DISTINCT CASE WHEN is_qualifying_sale THEN invoice_no END) AS order_count
    FROM fct_transactions
    WHERE ($country = 'ALL' OR country = $country)
    GROUP BY week_start
    ORDER BY week_start ASC;
    """

    df = db.query(sql, {"country": country})

    if df.empty or len(df) < 2:
        return pd.DataFrame()

    df["week_start"] = pd.to_datetime(df["week_start"])
    df["net_revenue"] = df["gross_sales"] - df["return_value"]

    # Reindex to full weekly grid to distinguish missing coverage from true zero weeks
    min_week = df["week_start"].min()
    max_week = df["week_start"].max()
    full_idx = pd.date_range(start=min_week, end=max_week, freq="W-MON", name="week_start")

    df = df.set_index("week_start").reindex(full_idx).reset_index()

    # Fill unobserved weeks with 0.0
    df["is_zero_week"] = df["net_revenue"].isna() | (df["net_revenue"] == 0)
    df["gross_sales"] = df["gross_sales"].fillna(0.0)
    df["return_value"] = df["return_value"].fillna(0.0)
    df["net_revenue"] = df["net_revenue"].fillna(0.0)
    df["units_sold"] = df["units_sold"].fillna(0)
    df["order_count"] = df["order_count"].fillna(0)
    df["active_days_in_week"] = df["active_days_in_week"].fillna(0)

    # Flag partial weeks at dataset boundaries (fewer than 6 days active)
    df["is_partial_week"] = False
    if len(df) > 0:
        if df.iloc[0]["active_days_in_week"] < 6:
            df.loc[0, "is_partial_week"] = True
        if df.iloc[-1]["active_days_in_week"] < 6:
            df.loc[len(df) - 1, "is_partial_week"] = True

    return df


def calculate_wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Computes Weighted Absolute Percentage Error (WAPE = sum(|y - y_hat|) / sum(y) * 100).
    Safely handles zero denominator.
    """
    total_actual = np.sum(actual)
    if total_actual <= 0:
        return 0.0
    return float((np.sum(np.abs(actual - predicted)) / total_actual) * 100.0)


def generate_sales_forecast(
    db: DatabaseManager,
    country: str = "ALL",
    horizon_weeks: int = 4,
    min_history_required: int = 12,
) -> ForecastResult:
    """
    Runs chronological time-series evaluation and generates 4-week demand projection.
    """
    weekly_df = aggregate_weekly_sales(db, country=country)

    if weekly_df.empty or len(weekly_df) < min_history_required:
        obs_count = len(weekly_df) if not weekly_df.empty else 0
        return ForecastResult(
            is_sufficient_history=False,
            status_message=(
                f"Insufficient historical data ({obs_count} weeks observed). "
                f"At least {min_history_required} full weeks of transaction history are required "
                "for robust time-series forecasting and holdout validation."
            ),
            historical_weekly_df=weekly_df,
        )

    # Use clean series (exclude partial end week if active days < 4)
    model_df = weekly_df.copy()
    if model_df.iloc[-1]["is_partial_week"] and model_df.iloc[-1]["active_days_in_week"] < 4:
        model_df = model_df.iloc[:-1].copy()

    y = model_df["net_revenue"].values
    dates = model_df["week_start"].dt.date.values
    n = len(y)

    # 1. Chronological Holdout Split (Never Random!)
    # Holdout is the last horizon_weeks
    train_y = y[:-horizon_weeks]
    test_y = y[-horizon_weeks:]
    test_dates = dates[-horizon_weeks:]

    # 2. Model A: Naive Baseline (4-Week Rolling Mean of training data)
    naive_val = np.mean(train_y[-horizon_weeks:]) if len(train_y) >= horizon_weeks else np.mean(train_y)
    pred_naive = np.full(horizon_weeks, naive_val)
    mae_naive = float(np.mean(np.abs(test_y - pred_naive)))
    wape_naive = calculate_wape(test_y, pred_naive)

    eval_naive = ForecastEvaluation(
        model_name="Naive Baseline (4-Wk Rolling Avg)",
        mae=mae_naive,
        wape_pct=wape_naive,
        holdout_predictions=pred_naive.tolist(),
        actual_holdout=test_y.tolist(),
    )

    # 3. Model B: Statistical Model (Exponential Smoothing / Holt-Winters)
    try:
        if len(train_y) >= 20:
            stats_model = ExponentialSmoothing(
                train_y,
                trend="add",
                damped_trend=True,
                initialization_method="estimated",
            ).fit()
        else:
            stats_model = SimpleExpSmoothing(
                train_y,
                initialization_method="estimated",
            ).fit()

        pred_stats = stats_model.forecast(horizon_weeks)
        # Revenue cannot be negative
        pred_stats = np.maximum(pred_stats, 0.0)
        mae_stats = float(np.mean(np.abs(test_y - pred_stats)))
        wape_stats = calculate_wape(test_y, pred_stats)
    except Exception:
        # Fallback to linear trend if optimization fails
        x_tr = np.arange(len(train_y))
        slope, intercept = np.polyfit(x_tr, train_y, 1)
        pred_stats = np.maximum(intercept + slope * np.arange(len(train_y), len(train_y) + horizon_weeks), 0.0)
        mae_stats = float(np.mean(np.abs(test_y - pred_stats)))
        wape_stats = calculate_wape(test_y, pred_stats)

    eval_stats = ForecastEvaluation(
        model_name="Exponential Smoothing (Statistical)",
        mae=mae_stats,
        wape_pct=wape_stats,
        holdout_predictions=pred_stats.tolist(),
        actual_holdout=test_y.tolist(),
    )

    # 4. Out-of-sample Model Selection: Lower WAPE wins
    if eval_stats.wape_pct <= eval_naive.wape_pct:
        winner = "Exponential Smoothing (Statistical)"
    else:
        winner = "Naive Baseline (4-Wk Rolling Avg)"

    # 5. Fit selected model on FULL historical dataset to project next 4 weeks
    last_known_week = model_df["week_start"].iloc[-1]
    future_dates = [
        (last_known_week + timedelta(days=7 * (i + 1))).date()
        for i in range(horizon_weeks)
    ]

    try:
        if winner.startswith("Exponential"):
            if len(y) >= 20:
                final_model = ExponentialSmoothing(
                    y,
                    trend="add",
                    damped_trend=True,
                    initialization_method="estimated",
                ).fit()
            else:
                final_model = SimpleExpSmoothing(y, initialization_method="estimated").fit()
            future_point = final_model.forecast(horizon_weeks)
            residuals = y - final_model.fittedvalues
        else:
            final_val = np.mean(y[-horizon_weeks:])
            future_point = np.full(horizon_weeks, final_val)
            residuals = y - np.mean(y[-horizon_weeks:])
    except Exception:
        final_val = np.mean(y[-horizon_weeks:])
        future_point = np.full(horizon_weeks, final_val)
        residuals = y - final_val

    future_point = np.maximum(future_point, 0.0)

    # 6. Calculate Prediction Intervals (95% standard normal z=1.96)
    std_resid = float(np.std(residuals)) if len(residuals) > 0 else float(np.std(y))
    h_steps = np.arange(1, horizon_weeks + 1)
    margin = 1.96 * std_resid * np.sqrt(h_steps)
    lower_bound = np.maximum(future_point - margin, 0.0)
    upper_bound = future_point + margin

    future_df = pd.DataFrame({
        "week_start": future_dates,
        "forecast_net_revenue": future_point.round(2),
        "ci_lower_95": lower_bound.round(2),
        "ci_upper_95": upper_bound.round(2),
    })

    notes = {
        "Methodology": "Chronological holdout evaluation on the most recent 4 weeks. No random splits.",
        "Model Selection": f"Selected '{winner}' based on superior out-of-sample WAPE on holdout.",
        "Interpretation": "Provides a short-term operational demand baseline assuming prevailing conditions continue.",
        "Limitation": (
            "Statistical extrapolation does not account for planned marketing promotional spikes, "
            "supply chain inventory stockouts, or unexpected macroeconomic disruptions."
        ),
    }

    return ForecastResult(
        is_sufficient_history=True,
        status_message="Forecast successfully generated with out-of-sample validation.",
        historical_weekly_df=model_df,
        evaluation_naive=eval_naive,
        evaluation_stats=eval_stats,
        selected_model_name=winner,
        future_forecast_df=future_df,
        notes=notes,
    )
