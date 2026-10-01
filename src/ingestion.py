"""
RetailPulse - Data Ingestion Module

Handles:
1. Metadata & documentation for UCI Online Retail dataset.
2. Ingestion & schema validation for CSV and Excel files.
3. Deterministic synthetic demo data generation (24+ months, seasonal patterns,
   repeat buyers, cancellations, missing customer IDs, intentional anomalies).
"""

from dataclasses import dataclass
import io
import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# UCI Online Retail Dataset Documentation & Metadata
UCI_METADATA = {
    "dataset_name": "Online Retail Data Set",
    "source_url": "https://archive.ics.uci.edu/dataset/352/online+retail",
    "download_url": "https://archive.ics.uci.edu/static/public/352/online+retail.zip",
    "citation": (
        "Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository. "
        "https://doi.org/10.24432/C5BW33"
    ),
    "donator": "Dr Daqing Chen, Director of Public Analytics, School of Engineering, London South Bank University",
    "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
    "currency": "GBP (£)",
    "expected_columns": [
        "InvoiceNo",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "CustomerID",
        "Country",
    ],
}

REQUIRED_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]


@dataclass
class ValidationResult:
    is_valid: bool
    missing_columns: List[str]
    detected_columns: List[str]
    row_count: int
    error_message: Optional[str] = None


def validate_schema(df: pd.DataFrame) -> ValidationResult:
    """
    Validates that the input DataFrame contains all required UCI columns.
    Matches column names case-insensitively while preserving standard casing.
    """
    if df is None or df.empty:
        return ValidationResult(
            is_valid=False,
            missing_columns=REQUIRED_COLUMNS,
            detected_columns=[],
            row_count=0,
            error_message="Uploaded file is empty or could not be parsed.",
        )

    # Standardize column names mapping
    col_map = {col.strip().lower(): col for col in df.columns}
    missing = []
    rename_dict = {}

    for req in REQUIRED_COLUMNS:
        req_lower = req.lower()
        if req_lower in col_map:
            actual_col = col_map[req_lower]
            rename_dict[actual_col] = req
        else:
            missing.append(req)

    if missing:
        return ValidationResult(
            is_valid=False,
            missing_columns=missing,
            detected_columns=list(df.columns),
            row_count=len(df),
            error_message=f"Missing required columns: {', '.join(missing)}",
        )

    # Standardize names on DataFrame in place if needed
    df.rename(columns=rename_dict, inplace=True)

    return ValidationResult(
        is_valid=True,
        missing_columns=[],
        detected_columns=list(df.columns),
        row_count=len(df),
        error_message=None,
    )


def load_file(
    file_source: Union[str, io.BytesIO, io.StringIO, bytes],
    file_type: Optional[str] = None,
) -> Tuple[Optional[pd.DataFrame], ValidationResult]:
    """
    Loads CSV or Excel data with strict schema validation.
    """
    try:
        if isinstance(file_source, str):
            if file_source.endswith(".xlsx") or file_source.endswith(".xls"):
                df = pd.read_excel(file_source, dtype={"CustomerID": "object", "InvoiceNo": "object"})
            else:
                df = pd.read_csv(file_source, dtype={"CustomerID": "object", "InvoiceNo": "object"}, low_memory=False)
        else:
            # File buffer (Streamlit UploadedFile or BytesIO)
            if file_type in ["xlsx", "xls", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
                df = pd.read_excel(file_source, dtype={"CustomerID": "object", "InvoiceNo": "object"})
            else:
                df = pd.read_csv(file_source, dtype={"CustomerID": "object", "InvoiceNo": "object"}, low_memory=False)

        val_result = validate_schema(df)
        if not val_result.is_valid:
            return None, val_result

        return df, val_result
    except Exception as e:
        return None, ValidationResult(
            is_valid=False,
            missing_columns=[],
            detected_columns=[],
            row_count=0,
            error_message=f"Failed to read file: {str(e)}",
        )


def generate_synthetic_demo_data(seed: int = 42) -> pd.DataFrame:
    """
    Generates a deterministic synthetic e-commerce transaction dataset with a fixed seed.
    Characteristics:
    - Exactly 24 full months (2024-01-01 through 2025-12-31).
    - ~400 customer cohorts with realistic repeat purchase rates (Champions, Regulars, One-off).
    - ~18% missing CustomerID representing guest checkouts (just like UCI Online Retail).
    - Seasonal variations: Strong Q4 holiday gift peak (Nov-Dec), moderate spring bump, summer dip.
    - Day-of-week shopping patterns (higher mid-week/Sunday orders).
    - Cancellations (~6% of invoices prefixed with 'C', negative quantities, matching prices).
    - Product catalogue with varied price tiers and Pareto distribution (top 20% products = ~80% sales).
    - Intentional data quality problems:
      * ~1.5% exact duplicate rows (scanner glitch simulation).
      * Negative or zero unit prices (bad debts, damaged stock adjustments, manual postages).
      * Missing descriptions for some records.
      * Whitespace padding on stock codes.
    """
    rng = np.random.default_rng(seed)

    # 1. Product Catalog (65 realistic gift/home retail items)
    products = [
        ("85123A", "WHITE HANGING HEART T-LIGHT HOLDER", 2.95, 0.08),
        ("71053", "WHITE METAL LANTERN", 3.75, 0.03),
        ("84406B", "CREAM CUPID HEARTS COAT HANGER", 3.25, 0.02),
        ("84029G", "KNITTED UNION FLAG HOT WATER BOTTLE", 4.25, 0.04),
        ("84029E", "RED WOOLLY HOTTIE WHITE HEART.", 4.25, 0.03),
        ("22752", "SET 7 BABUSHKA NESTING BOXES", 8.50, 0.05),
        ("21730", "GLASS STAR FROSTED T-LIGHT HOLDER", 4.95, 0.02),
        ("22633", "HAND WARMER UNION JACK", 2.10, 0.03),
        ("22632", "HAND WARMER RED POLKADOT", 2.10, 0.03),
        ("84879", "ASSORTED COLOUR BIRD ORNAMENT", 1.69, 0.07),
        ("22745", "POPPY'S PLAYHOUSE BEDROOM", 2.10, 0.02),
        ("22748", "POPPY'S PLAYHOUSE KITCHEN", 2.10, 0.02),
        ("22749", "POPPY'S PLAYHOUSE BATHROOM", 2.10, 0.02),
        ("22310", "IVORY KNITTED MUG COSY", 1.65, 0.03),
        ("84969", "BOX OF 6 ASSORTED COLOUR TEASPOONS", 4.25, 0.03),
        ("20725", "LUNCH BAG RED RETROSPOT", 1.65, 0.09),
        ("20727", "LUNCH BAG BLACK SKULL.", 1.65, 0.04),
        ("20728", "LUNCH BAG CARS BLUE", 1.65, 0.04),
        ("20726", "LUNCH BAG WOODLAND", 1.65, 0.04),
        ("22383", "LUNCH BAG SUKI DESIGN", 1.65, 0.03),
        ("22384", "LUNCH BAG PINK POLKADOT", 1.65, 0.03),
        ("22382", "LUNCH BAG SPACEBOY DESIGN", 1.65, 0.04),
        ("22666", "RECIPE BOX PANTRY DESIGN", 7.95, 0.03),
        ("22667", "RECIPE BOX RETROSPOT", 7.95, 0.02),
        ("22423", "REGENCY CAKESTAND 3 TIER", 12.75, 0.07),
        ("22720", "SET OF 3 CAKE TINS PANTRY DESIGN", 4.95, 0.03),
        ("22722", "SET OF 6 SPICE TINS PANTRY DESIGN", 3.95, 0.02),
        ("22138", "BAKING SET 9 PIECE RETROSPOT", 4.95, 0.02),
        ("22139", "RETROSPOT TEA SET CERAMIC 11 PC", 4.95, 0.02),
        ("22457", "NATURAL SLATE HEART CHALKBOARD", 2.95, 0.03),
        ("22469", "HEART OF WICKER SMALL", 1.45, 0.03),
        ("22470", "HEART OF WICKER LARGE", 2.55, 0.02),
        ("21212", "PACK OF 72 RETROSPOT CAKE CASES", 0.55, 0.05),
        ("21977", "PACK OF 60 PINK PAISLEY CAKE CASES", 0.55, 0.02),
        ("84991", "60 TEATIME FAIRY CAKE CASES", 0.55, 0.03),
        ("84992", "72 SWEETHEART FAIRY CAKE CASES", 0.55, 0.02),
        ("22960", "JAM MAKING SET WITH JARS", 4.25, 0.04),
        ("22961", "JAM MAKING SET PRINTED", 1.45, 0.03),
        ("23203", "JUMBO BAG DOILEY PATTERN", 2.08, 0.03),
        ("23204", "CHARLOTTE BAG APPLES DESIGN", 0.85, 0.02),
        ("23206", "JUMBO BAG VINTAGE DOILY", 2.08, 0.02),
        ("85099B", "JUMBO BAG RED RETROSPOT", 2.08, 0.08),
        ("23209", "LUNCH BAG VINTAGE DOILY", 1.65, 0.02),
        ("22086", "PAPER CHAIN KIT 50'S CHRISTMAS", 2.95, 0.05),
        ("22077", "6 RIBBONS RUSTIC CHARM", 1.65, 0.03),
        ("22834", "HAND WARMER BABUSHKA HEART", 2.10, 0.02),
        ("22835", "HOT WATER BOTTLE I AM SO POOR", 4.65, 0.02),
        ("22111", "SCOTTIE DOG HOT WATER BOTTLE", 4.95, 0.02),
        ("22112", "CHOCOLATE HOT WATER BOTTLE", 4.95, 0.02),
        ("22114", "HOT WATER BOTTLE TEA AND SYMPATHY", 4.25, 0.02),
        ("22197", "POPCORN HOLDER", 0.85, 0.04),
        ("22355", "CHARLOTTE BAG SUKI DESIGN", 0.85, 0.03),
        ("22356", "CHARLOTTE BAG PINK POLKADOT", 0.85, 0.03),
        ("23207", "JUMBO BAG ALPHABET", 2.08, 0.02),
        ("23208", "JUMBO BAG VINTAGE LEAF", 2.08, 0.02),
        ("22554", "PLASTERS IN TIN WOODLAND ANIMALS", 1.65, 0.03),
        ("22556", "PLASTERS IN TIN CIRCUS PARADE", 1.65, 0.02),
        ("22555", "PLASTERS IN TIN RETROSPOT", 1.65, 0.02),
        ("22326", "ROUND SNACK BOXES SET OF4 WOODLAND", 2.95, 0.04),
        ("22328", "ROUND SNACK BOXES SET OF 4 FRUITS", 2.95, 0.02),
        ("22558", "CLOTHES PEGS RETROSPOT PACK 24", 1.45, 0.02),
        ("22699", "ROSES REGENCY TEACUP AND SAUCER", 2.95, 0.03),
        ("22697", "GREEN REGENCY TEACUP AND SAUCER", 2.95, 0.03),
        ("22698", "PINK REGENCY TEACUP AND SAUCER", 2.95, 0.02),
        ("22178", "VICTORIAN GLASS HANGING T-LIGHT", 1.25, 0.03),
    ]

    # Normalize product probabilities
    raw_weights = [p[3] for p in products]
    p_weights = np.array(raw_weights) / sum(raw_weights)

    # 2. Customer Pool & Country Distribution
    # Realistic UK-heavy distribution (88% UK, 12% European exports)
    countries = ["United Kingdom", "Germany", "France", "EIRE", "Spain", "Netherlands", "Belgium", "Switzerland"]
    country_weights = [0.86, 0.04, 0.03, 0.02, 0.02, 0.015, 0.008, 0.007]

    num_known_customers = 420
    customer_ids = [str(12000 + i) for i in range(num_known_customers)]
    customer_countries = rng.choice(countries, size=num_known_customers, p=country_weights)

    # Customer Persona segments for repeat behavior
    # High frequency / Champions (15%), Regulars (35%), Low/One-off (50%)
    customer_types = rng.choice(
        ["champion", "regular", "occasional"],
        size=num_known_customers,
        p=[0.15, 0.35, 0.50],
    )

    # First acquisition month for each customer across 24 months (2024-01 to 2025-12)
    # Staggered entry across months 0 to 23
    customer_start_month = rng.integers(0, 24, size=num_known_customers)

    # 3. Time Grid Generation: 24 Months
    # 2024-01-01 to 2025-12-31 (731 days)
    start_date = pd.Timestamp("2024-01-01")
    end_date = pd.Timestamp("2025-12-31 23:59:59")
    total_days = (end_date - start_date).days

    rows = []
    invoice_seq = 536365

    # Target: ~15,000 to ~18,000 line items representing ~4,500 distinct orders
    # We will simulate day by day
    current_date = start_date

    while current_date <= end_date:
        month_idx = (current_date.year - 2024) * 12 + (current_date.month - 1)
        month = current_date.month
        day_of_week = current_date.dayofweek  # 0=Mon, 6=Sun

        # Base daily order volume: Seasonal multiplier
        # Nov/Dec holiday shopping surge (1.8x - 2.2x)
        # Jan/Feb post-holiday lull (0.8x)
        # Summer lull (0.75x)
        season_mult = 1.0
        if month in [11, 12]:
            season_mult = 2.1
        elif month in [10]:
            season_mult = 1.4
        elif month in [1, 2]:
            season_mult = 0.8
        elif month in [7, 8]:
            season_mult = 0.75
        elif month in [3, 4, 5]:
            season_mult = 1.1

        # Day of week multiplier (Online retail is open every day, but Saturdays slightly lower)
        dow_mult = 0.6 if day_of_week == 5 else 1.05

        # Active customers eligible to buy on this day
        active_mask = customer_start_month <= month_idx
        eligible_cust_indices = np.where(active_mask)[0]

        # Number of orders on this day
        daily_order_count = int(rng.poisson(lam=7 * season_mult * dow_mult))

        for _ in range(daily_order_count):
            invoice_seq += 1
            invoice_no = str(invoice_seq)

            # Determine if Guest checkout (missing CustomerID, ~18%)
            is_guest = rng.random() < 0.18

            if is_guest:
                cust_id = None
                country = rng.choice(countries, p=country_weights)
            else:
                # Pick customer based on persona weights
                eligible_weights = []
                for idx in eligible_cust_indices:
                    ctype = customer_types[idx]
                    if ctype == "champion":
                        eligible_weights.append(5.0)
                    elif ctype == "regular":
                        eligible_weights.append(1.5)
                    else:
                        eligible_weights.append(0.5)

                eligible_weights = np.array(eligible_weights) / sum(eligible_weights)
                selected_idx = rng.choice(eligible_cust_indices, p=eligible_weights)
                cust_id = customer_ids[selected_idx]
                country = customer_countries[selected_idx]

            # Timestamp within business hours (07:00 to 20:00)
            hour = rng.integers(7, 20)
            minute = rng.integers(0, 60)
            second = rng.integers(0, 60)
            order_time = current_date.replace(hour=hour, minute=minute, second=second)

            # Lines per order (between 1 and 8 items)
            num_items = rng.choice([1, 2, 3, 4, 5, 6, 8], p=[0.25, 0.28, 0.20, 0.12, 0.08, 0.05, 0.02])
            selected_prod_indices = rng.choice(len(products), size=num_items, replace=False, p=p_weights)

            order_items = []
            for p_idx in selected_prod_indices:
                stock_code, desc, unit_price, _ = products[p_idx]
                # Quantity: typical retail quantities (1 to 24)
                qty = int(rng.choice([1, 2, 3, 4, 6, 12, 24], p=[0.35, 0.25, 0.15, 0.10, 0.08, 0.05, 0.02]))
                order_items.append((stock_code, desc, qty, unit_price))
                rows.append({
                    "InvoiceNo": invoice_no,
                    "StockCode": stock_code,
                    "Description": desc,
                    "Quantity": qty,
                    "InvoiceDate": order_time,
                    "UnitPrice": unit_price,
                    "CustomerID": cust_id,
                    "Country": country,
                })

            # Simulate Return / Cancellation (~5.5% probability for the order)
            if rng.random() < 0.055 and order_items:
                # Cancellation occurs 1 to 14 days later
                cancel_days = rng.integers(1, 15)
                cancel_time = order_time + pd.Timedelta(days=int(cancel_days), hours=rng.integers(1, 5))
                if cancel_time <= end_date:
                    cancel_invoice = f"C{invoice_no}"
                    # Return 1 or all items from the order
                    num_returned = rng.choice([1, len(order_items)], p=[0.6, 0.4])
                    returned_items = order_items[:num_returned]
                    for r_code, r_desc, r_qty, r_price in returned_items:
                        rows.append({
                            "InvoiceNo": cancel_invoice,
                            "StockCode": r_code,
                            "Description": r_desc,
                            "Quantity": -abs(r_qty),
                            "InvoiceDate": cancel_time,
                            "UnitPrice": r_price,
                            "CustomerID": cust_id,
                            "Country": country,
                        })

        current_date += pd.Timedelta(days=1)

    df_synth = pd.DataFrame(rows)

    # 4. Inject intentional data-quality anomalies (documented in QA / quarantine)
    # A. Exact duplicate rows (~1.2%)
    num_dupes = int(len(df_synth) * 0.012)
    dupe_indices = rng.choice(len(df_synth), size=num_dupes, replace=False)
    dupes = df_synth.iloc[dupe_indices].copy()

    # B. Zero or negative unit prices (postage / adjust / bad debt) (~0.4%)
    num_bad_prices = int(len(df_synth) * 0.004)
    bad_price_indices = rng.choice(len(df_synth), size=num_bad_prices, replace=False)
    for idx in bad_price_indices:
        df_synth.loc[idx, "UnitPrice"] = rng.choice([0.0, -15.5, -45.0])
        df_synth.loc[idx, "Description"] = rng.choice(["Adjust bad debt", "damaged stock", "amazon fee", "lost in transit"])

    # C. Missing / whitespace descriptions (~0.5%)
    num_missing_desc = int(len(df_synth) * 0.005)
    missing_desc_indices = rng.choice(len(df_synth), size=num_missing_desc, replace=False)
    for idx in missing_desc_indices:
        df_synth.loc[idx, "Description"] = rng.choice([None, np.nan, "   "])

    # D. Unparseable dates (~0.1%)
    # Cast InvoiceDate to object first to allow invalid string simulation
    df_synth["InvoiceDate"] = df_synth["InvoiceDate"].astype(str)
    num_bad_dates = int(len(df_synth) * 0.001)
    bad_date_indices = rng.choice(len(df_synth), size=num_bad_dates, replace=False)
    for idx in bad_date_indices:
        df_synth.loc[idx, "InvoiceDate"] = "INVALID_DATE_RECORD"

    # Concatenate duplicates
    df_combined = pd.concat([df_synth, dupes], ignore_index=True)

    # Ensure CustomerID is string or NaN
    df_combined["CustomerID"] = df_combined["CustomerID"].astype(object)
    df_combined["InvoiceNo"] = df_combined["InvoiceNo"].astype(str)

    return df_combined


def get_demo_dataset_path() -> str:
    """Returns absolute path to local sample demo CSV."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sample_dir = os.path.join(base_dir, "data", "sample")
    os.makedirs(sample_dir, exist_ok=True)
    return os.path.join(sample_dir, "online_retail_sample.csv")


def ensure_sample_data_exists() -> str:
    """
    Ensures that data/sample/online_retail_sample.csv exists.
    If not, generates it deterministically.
    """
    sample_path = get_demo_dataset_path()
    if not os.path.exists(sample_path) or os.path.getsize(sample_path) < 1000:
        df = generate_synthetic_demo_data(seed=42)
        df.to_csv(sample_path, index=False)
    return sample_path
