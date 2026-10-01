"""
RetailPulse - DuckDB Analytical Database Layer

Manages DuckDB in-memory database, creates dimensional models (fct_transactions,
fct_orders, dim_customers, dim_products, dim_dates), and executes parameterized
SQL scripts.
"""

from datetime import date, datetime
import os
from typing import Any, Dict, List, Optional, Tuple
import duckdb
import pandas as pd


class DatabaseManager:
    """Encapsulates DuckDB connection, table loading, and parameterized SQL queries."""

    def __init__(self, db_path: str = ":memory:"):
        self.con = duckdb.connect(db_path)
        self._load_schema()

    def _load_schema(self):
        """Loads and executes DDL from sql/schema.sql."""
        sql_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sql")
        schema_path = os.path.join(sql_dir, "schema.sql")
        if os.path.exists(schema_path):
            with open(schema_path, "r", encoding="utf-8") as f:
                ddl = f.read()
            self.con.execute(ddl)

    def populate_database(self, clean_df: pd.DataFrame):
        """
        Builds and populates all dimensional and fact tables from clean transactions.
        """
        if clean_df is None or clean_df.empty:
            return

        # 1. Populate fct_transactions
        self.con.register("df_clean_stage", clean_df)
        self.con.execute("DELETE FROM fct_transactions;")
        self.con.execute("""
            INSERT INTO fct_transactions
            SELECT
                transaction_id,
                invoice_no,
                stock_code,
                description,
                quantity,
                unit_price,
                line_amount,
                gross_amount,
                return_amount,
                net_amount,
                invoice_date,
                CAST(date AS DATE) AS date,
                year_month,
                CAST(week_start AS DATE) AS week_start,
                customer_id,
                country,
                is_cancellation,
                is_return,
                is_qualifying_sale,
                has_customer_id
            FROM df_clean_stage;
        """)

        # 2. Populate dim_dates
        self.con.execute("DELETE FROM dim_dates;")
        self.con.execute("""
            INSERT INTO dim_dates
            SELECT DISTINCT
                date AS date_id,
                EXTRACT('year' FROM date) AS year,
                EXTRACT('month' FROM date) AS month,
                STRFTIME(date, '%Y-%m') AS year_month,
                EXTRACT('quarter' FROM date) AS quarter,
                week_start,
                STRFTIME(date, '%A') AS day_name,
                CASE WHEN DAYOFWEEK(date) IN (0, 6) THEN TRUE ELSE FALSE END AS is_weekend
            FROM fct_transactions;
        """)

        # 3. Populate dim_products
        self.con.execute("DELETE FROM dim_products;")
        self.con.execute("""
            INSERT INTO dim_products
            WITH prod_stats AS (
                SELECT
                    stock_code,
                    description,
                    unit_price,
                    COUNT(*) AS line_cnt,
                    ROW_NUMBER() OVER (PARTITION BY stock_code ORDER BY COUNT(*) DESC, LENGTH(description) DESC) AS rn
                FROM fct_transactions
                WHERE description <> '(No description)' AND description IS NOT NULL
                GROUP BY stock_code, description, unit_price
            )
            SELECT
                stock_code,
                COALESCE(description, '(Unknown Product)') AS description,
                COALESCE(unit_price, 0.0) AS typical_unit_price
            FROM prod_stats
            WHERE rn = 1;
        """)

        # 4. Populate dim_customers (known customers only)
        self.con.execute("DELETE FROM dim_customers;")
        self.con.execute("""
            INSERT INTO dim_customers
            WITH cust_summary AS (
                SELECT
                    customer_id,
                    FIRST(country) AS country,
                    MIN(invoice_date) AS first_purchase_date,
                    MAX(invoice_date) AS last_purchase_date,
                    COUNT(DISTINCT CASE WHEN is_qualifying_sale THEN invoice_no END) AS lifetime_orders,
                    COALESCE(SUM(CASE WHEN is_qualifying_sale THEN gross_amount ELSE 0.0 END), 0.0) AS lifetime_gross_spend
                FROM fct_transactions
                WHERE has_customer_id
                GROUP BY customer_id
            )
            SELECT
                customer_id,
                country,
                first_purchase_date,
                last_purchase_date,
                lifetime_orders,
                lifetime_gross_spend
            FROM cust_summary;
        """)

        # 5. Populate fct_orders
        self.con.execute("DELETE FROM fct_orders;")
        self.con.execute("""
            INSERT INTO fct_orders
            SELECT
                invoice_no,
                FIRST(customer_id) AS customer_id,
                FIRST(country) AS country,
                MIN(invoice_date) AS order_timestamp,
                MIN(date) AS order_date,
                FIRST(year_month) AS year_month,
                BOOL_OR(is_cancellation) AS is_cancellation,
                BOOL_OR(is_qualifying_sale) AS is_qualifying_sale,
                COUNT(*) AS line_count,
                SUM(quantity) AS total_quantity,
                COALESCE(SUM(CASE WHEN is_qualifying_sale THEN gross_amount ELSE 0.0 END), 0.0) AS gross_sales,
                COALESCE(SUM(CASE WHEN is_return THEN return_amount ELSE 0.0 END), 0.0) AS return_value,
                (COALESCE(SUM(CASE WHEN is_qualifying_sale THEN gross_amount ELSE 0.0 END), 0.0) -
                 COALESCE(SUM(CASE WHEN is_return THEN return_amount ELSE 0.0 END), 0.0)) AS net_revenue
            FROM fct_transactions
            GROUP BY invoice_no;
        """)

    def query(self, sql_query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Executes a safe parameterized SQL query and returns a pandas DataFrame.
        DuckDB accepts named parameters with $param_name.
        """
        if params:
            return self.con.execute(sql_query, params).df()
        return self.con.execute(sql_query).df()

    def get_countries(self) -> List[str]:
        """Returns sorted list of distinct countries present in the loaded dataset."""
        df = self.con.execute("SELECT DISTINCT country FROM fct_transactions ORDER BY country;").df()
        return df["country"].tolist() if not df.empty else []

    def get_date_bounds(self) -> Tuple[Optional[date], Optional[date]]:
        """Returns the minimum and maximum dates available in the dataset."""
        df = self.con.execute("SELECT MIN(date) AS min_d, MAX(date) AS max_d FROM fct_transactions;").df()
        if df.empty or df["min_d"].iloc[0] is None:
            return None, None
        min_d = df["min_d"].iloc[0]
        max_d = df["max_d"].iloc[0]
        if isinstance(min_d, pd.Timestamp):
            min_d = min_d.date()
        if isinstance(max_d, pd.Timestamp):
            max_d = max_d.date()
        return min_d, max_d


_GLOBAL_DB_MANAGER: Optional[DatabaseManager] = None


def get_database_manager(clean_df: Optional[pd.DataFrame] = None) -> DatabaseManager:
    """
    Singleton-style getter for DatabaseManager.
    If clean_df is provided or database not initialized, initializes with clean_df.
    """
    global _GLOBAL_DB_MANAGER
    if _GLOBAL_DB_MANAGER is None:
        _GLOBAL_DB_MANAGER = DatabaseManager()
        if clean_df is not None:
            _GLOBAL_DB_MANAGER.populate_database(clean_df)
    elif clean_df is not None:
        _GLOBAL_DB_MANAGER.populate_database(clean_df)
    return _GLOBAL_DB_MANAGER
