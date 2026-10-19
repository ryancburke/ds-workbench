import pandas as pd
import numpy as np
from config import config

# ── Sample dataset registry ───────────────────────────────────────────────────

SAMPLE_DATASETS = {
    "orders_dirty":        "Orders — Dirty           (nulls, mixed-case status & region, duplicate rows)",
    "sales_clean":         "Sales — Clean            (product, category, pricing, discounts, region)",
    "hr_attrition":        "HR Attrition             (salary, tenure, performance score, attrition flag)",
    "ecommerce":           "E-commerce               (sessions, page views, conversions, channel, device)",
    "ecommerce_leakage":   "E-commerce w/ Leakage    (same as ecommerce + conversion_score and order_id leakage features)",
    "transactions_fraud":  "Transactions — Fraud     (outlier amounts, mixed-case country, null user IDs, 3% fraud rate)",
    "sales_forecasting":   "Sales Forecasting        (promotions → revenue, seasonal pattern, regression target)",
    "subscription_churn":  "Subscription Churn       (disguised leakage: neutral-named status code, score and balance)",
    "customer_accounts":   "Customer Accounts        (hidden quality issues: placeholder nulls, sentinels, future dates, units)",
    "clinical_small_wide": "Clinical Small & Wide    (160 patients, 46 markers, few informative — overfitting trap)",
}

SAMPLE_JOIN_SCENARIOS = {
    "orders_customers_fanout": {
        "label":         "Orders ⋈ Customers  (fan-out — duplicate customer IDs on right)",
        "left_name":     "orders",
        "right_name":    "customers",
        "suggested_key": "customer_id",
    },
    "sales_products_clean": {
        "label":         "Sales ⋈ Products  (clean join — unique keys on both sides)",
        "left_name":     "sales",
        "right_name":    "products",
        "suggested_key": "product_id",
    },
    "transactions_users_many_to_many": {
        "label":         "Transactions ⋈ Users  (many-to-many — duplicate keys on both sides)",
        "left_name":     "transactions",
        "right_name":    "users",
        "suggested_key": "user_id",
    },
}


def load_sample_dataset(name: str) -> pd.DataFrame:
    generators = {
        "orders_dirty":       _gen_orders_dirty,
        "sales_clean":        _gen_sales_clean,
        "hr_attrition":       _gen_hr_attrition,
        "ecommerce":          _gen_ecommerce,
        "ecommerce_leakage":  _gen_ecommerce_leakage,
        "transactions_fraud": _gen_transactions_fraud,
        "sales_forecasting":  _gen_sales_forecasting,
        "subscription_churn": _gen_subscription_churn,
        "customer_accounts":  _gen_customer_accounts,
        "clinical_small_wide": _gen_clinical_small_wide,
    }
    if name not in generators:
        raise ValueError(f"Unknown sample dataset: {name}")
    return generators[name]()


def load_join_scenario(name: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Returns (left_df, right_df, suggested_join_key)."""
    if name == "orders_customers_fanout":
        return _gen_orders_dirty(), _gen_customers_with_dupes(), "customer_id"
    if name == "sales_products_clean":
        return _gen_sales_with_product_id(), _gen_products(), "product_id"
    if name == "transactions_users_many_to_many":
        return _gen_transactions_for_join(), _gen_users_with_dupes(), "user_id"
    raise ValueError(f"Unknown join scenario: {name}")


# ── Databricks connections ─────────────────────────────────────────────────────

def list_tables(catalog: str = "hive_metastore", schema: str = "default") -> list[str]:
    if config.local_dev_mode:
        return list(SAMPLE_DATASETS.keys())
    from databricks import sql
    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SHOW TABLES IN {catalog}.{schema}")
            return [row.tableName for row in cur.fetchall()]


def get_table_sample(table: str, limit: int = 1000) -> pd.DataFrame:
    if config.local_dev_mode:
        return load_sample_dataset(table) if table in SAMPLE_DATASETS else _gen_orders_dirty()
    from databricks import sql
    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT * FROM {table} LIMIT {limit}")
            return cur.fetchall_arrow().to_pandas()


def get_table_schema(table: str) -> list[dict]:
    if config.local_dev_mode:
        df = load_sample_dataset(table) if table in SAMPLE_DATASETS else _gen_orders_dirty()
        return [{"column_name": col, "data_type": str(dtype)} for col, dtype in df.dtypes.items()]
    from databricks import sql
    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"DESCRIBE {table}")
            return [{"column_name": r.col_name, "data_type": r.data_type} for r in cur.fetchall()]


def _get_connection():
    from databricks import sql
    return sql.connect(
        server_hostname=config.databricks_host,
        http_path=config.databricks_http_path,
        access_token=config.databricks_token,
    )


# ── Sample data generators ─────────────────────────────────────────────────────

def _gen_orders_dirty(n: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "order_id":    range(1001, n + 1001),
        "customer_id": [f"C{rng.integers(1, 200):04d}" if rng.random() > 0.04 else None for _ in range(n)],
        "amount":      np.where(rng.random(n) > 0.09, rng.exponential(120, n).round(2), None),
        "status":      rng.choice(
            ["completed", "pending", "Completed", "CANCELLED", "refunded", None],
            n, p=[0.5, 0.2, 0.1, 0.1, 0.05, 0.05],
        ),
        "region":      rng.choice(["NORTH", "South", "east", "West", None], n, p=[0.3, 0.3, 0.2, 0.15, 0.05]),
        "email":       [f"user{i}@example.com" if rng.random() > 0.05 else None for i in range(n)],
        "score":       rng.normal(65, 20, n).clip(0, 100).round(1),
        "created_at":  pd.date_range("2023-01-01", periods=n, freq="2h"),
    })
    dupes = df.sample(int(n * 0.03), random_state=7)
    return pd.concat([df, dupes], ignore_index=True).head(n)


def _gen_sales_clean(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    return pd.DataFrame({
        "sale_id":      range(1, n + 1),
        "product":      rng.choice(["Widget A", "Widget B", "Gadget X", "Gadget Y"], n),
        "category":     rng.choice(["Electronics", "Accessories", "Bundles"], n, p=[0.5, 0.3, 0.2]),
        "quantity":     rng.integers(1, 20, n),
        "unit_price":   rng.choice([9.99, 19.99, 49.99, 99.99, 199.99], n),
        "discount_pct": rng.choice([0, 5, 10, 15, 20], n, p=[0.5, 0.2, 0.15, 0.1, 0.05]),
        "region":       rng.choice(["North", "South", "East", "West"], n),
        "sale_date":    pd.date_range("2024-01-01", periods=n, freq="6h"),
    })


def _gen_hr_attrition(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(77)
    salary = rng.normal(85000, 22000, n).clip(30000, 180000).round(-2)
    outlier_idx = rng.choice(n, 8, replace=False)
    salary[outlier_idx] = rng.choice([5000, 8000, 500000, 750000], 8)
    return pd.DataFrame({
        "employee_id":        [f"E{i:04d}" for i in range(1, n + 1)],
        "department":         rng.choice(["Engineering", "Sales", "Marketing", "Operations", "Finance"], n,
                                         p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        "job_level":          rng.choice(["IC1", "IC2", "IC3", "IC4", "M1", "M2"], n,
                                         p=[0.2, 0.3, 0.2, 0.1, 0.15, 0.05]),
        "salary":             salary,
        "tenure_years":       rng.exponential(3.5, n).clip(0.1, 20).round(1),
        "performance_score":  np.where(rng.random(n) > 0.12, rng.integers(1, 6, n).astype(float), None),
        "satisfaction_score": np.where(rng.random(n) > 0.08, rng.integers(1, 11, n).astype(float), None),
        "attrition":          rng.choice([True, False], n, p=[0.18, 0.82]),
        "remote_pct":         rng.choice([0, 25, 50, 75, 100], n),
    })


def _gen_ecommerce(n: int = 700) -> pd.DataFrame:
    rng = np.random.default_rng(55)
    return pd.DataFrame({
        "session_id":     [f"S{i:06d}" for i in range(1, n + 1)],
        "user_id":        [f"U{rng.integers(1, 400):05d}" if rng.random() > 0.1 else None for _ in range(n)],
        "channel":        rng.choice(["organic", "paid_search", "social", "email", "direct", None],
                                     n, p=[0.3, 0.25, 0.2, 0.1, 0.1, 0.05]),
        "device":         rng.choice(["mobile", "desktop", "tablet"], n, p=[0.55, 0.38, 0.07]),
        "page_views":     rng.integers(1, 30, n),
        "time_on_site_s": np.where(rng.random(n) > 0.05, rng.exponential(180, n).round(0).clip(10, 1200), None),
        "converted":      rng.choice([True, False], n, p=[0.08, 0.92]),
        "cart_value":     np.where(rng.random(n) > 0.3, rng.exponential(75, n).round(2), 0.0),
        "session_date":   pd.date_range("2024-03-01", periods=n, freq="1h"),
    })


def _gen_customers_with_dupes(n: int = 150) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    base = pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(1, n + 1)],
        "name":        [f"Customer {i}" for i in range(1, n + 1)],
        "tier":        rng.choice(["gold", "silver", "bronze"], n, p=[0.2, 0.4, 0.4]),
        "country":     rng.choice(["US", "UK", "CA", "AU"], n),
    })
    dupes = base.sample(int(n * 0.10), random_state=42)
    return pd.concat([base, dupes], ignore_index=True)


def _gen_sales_with_product_id(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(33)
    return pd.DataFrame({
        "sale_id":    range(1, n + 1),
        "product_id": rng.integers(1, 21, n),
        "quantity":   rng.integers(1, 15, n),
        "unit_price": rng.choice([9.99, 29.99, 59.99, 99.99], n),
        "sale_date":  pd.date_range("2024-01-01", periods=n, freq="4h"),
    })


def _gen_products(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(33)
    return pd.DataFrame({
        "product_id": range(1, n + 1),
        "name":       [f"Product {i}" for i in range(1, n + 1)],
        "category":   rng.choice(["Electronics", "Accessories", "Bundles"], n),
        "cost_price": (rng.uniform(5, 80, n) * 100).round() / 100,
    })


def _gen_transactions_fraud(n: int = 1000) -> pd.DataFrame:
    """
    Financial transactions with injected quality issues:
    - ~8% null user_ids
    - amount: extreme outlier injections (simulated fraud amounts)
    - country: mixed-case values
    - is_fraud: ~3% True (severe class imbalance)
    """
    rng = np.random.default_rng(13)
    amount = rng.exponential(150, n).clip(1, 2000).round(2)
    fraud_idx = rng.choice(n, int(n * 0.03), replace=False)
    amount[fraud_idx] = rng.choice([8500.0, 12000.0, 25000.0, 47500.0], len(fraud_idx))
    is_fraud = np.zeros(n, dtype=bool)
    is_fraud[fraud_idx] = True

    countries_mixed = rng.choice(
        ["US", "us", "United States", "UK", "uk", "CA", "ca", "AU"],
        n, p=[0.35, 0.05, 0.05, 0.2, 0.05, 0.15, 0.05, 0.1],
    )
    return pd.DataFrame({
        "transaction_id":   [f"TXN{i:07d}" for i in range(1, n + 1)],
        "user_id":          [f"U{rng.integers(1, 300):05d}" if rng.random() > 0.08 else None for _ in range(n)],
        "merchant_id":      [f"M{rng.integers(1, 100):04d}" if rng.random() > 0.03 else None for _ in range(n)],
        "amount":           amount,
        "currency":         rng.choice(["USD", "GBP", "CAD", "AUD"], n, p=[0.6, 0.2, 0.1, 0.1]),
        "country":          countries_mixed,
        "device_type":      rng.choice(["mobile", "desktop", "tablet", "api"], n, p=[0.45, 0.35, 0.1, 0.1]),
        "transaction_type": rng.choice(["purchase", "refund", "transfer"], n, p=[0.85, 0.1, 0.05]),
        "is_fraud":         is_fraud,
        "created_at":       pd.date_range("2024-01-01", periods=n, freq="45min"),
    })


def _gen_sales_forecasting(n: int = 730) -> pd.DataFrame:
    """
    Daily sales data designed for regression EDA.
    Strong promotions_spend → revenue correlation (r ≈ 0.82, seeded).
    Seasonal weekly pattern. Clean — no injected quality issues.
    Answer key: revenue as regression target, promotions_spend↔revenue correlation.
    """
    rng = np.random.default_rng(61)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")

    promotions_spend = rng.uniform(500, 5000, n).round(2)
    seasonal_mult = 1.0 + 0.25 * np.sin(2 * np.pi * np.arange(n) / 365)
    weekly_mult   = np.where(dates.dayofweek < 5, 1.0, 0.7)  # weekends lower
    noise = rng.normal(0, 500, n)
    revenue = (promotions_spend * 12 * seasonal_mult * weekly_mult + noise + 8000).clip(0).round(2)

    return pd.DataFrame({
        "date":             dates,
        "region":           rng.choice(["North", "South", "East", "West"], n),
        "channel":          rng.choice(["online", "retail", "wholesale"], n, p=[0.5, 0.35, 0.15]),
        "promotions_spend": promotions_spend,
        "units_sold":       (revenue / rng.uniform(20, 80, n)).clip(0).round().astype(int),
        "returns_pct":      rng.uniform(0, 12, n).round(2),
        "new_customers":    rng.integers(5, 200, n),
        "revenue":          revenue,
    })


def _gen_clinical_small_wide(n: int = 160) -> pd.DataFrame:
    """
    Treatment response for 160 patients with 46 numeric features, of which only 4 carry
    (linear) signal; 6 are near-duplicates of others (r > 0.95). Target `responder` ~35%.
    The trap: with n/p ≈ 3.5, boosting or deep models overfit and a single 20% holdout is
    32 patients. Regularised linear models with repeated stratified CV are the sound choice.
    """
    rng = np.random.default_rng(31)
    X = rng.normal(0, 1, (n, 40))
    signal = 1.1 * X[:, 0] - 0.9 * X[:, 1] + 0.8 * X[:, 2] + 0.6 * X[:, 3]
    responder = (signal + rng.normal(0, 1.0, n)) > 0.45
    cols = {f"marker_{i + 1:02d}": X[:, i].round(4) for i in range(40)}
    for j, src in enumerate([0, 1, 4, 7, 9, 12]):   # near-duplicate assays of other markers
        cols[f"marker_{src + 1:02d}_rerun"] = (X[:, src] + rng.normal(0, 0.2, n)).round(4)
    return pd.DataFrame({
        "patient_id": [f"P{i:04d}" for i in range(n)],
        "age": rng.integers(25, 80, n),
        **cols,
        "responder": responder,
    })


def _gen_customer_accounts(n: int = 1500) -> pd.DataFrame:
    """
    Customer accounts with quality problems that null counts and IQR fences alone miss:
    - email: 7% empty strings and annual_income: 6% "N/A" — missing values that isnull() doesn't count
    - age: 4% sentinel 999 and 1% -1
    - signup_date: 3% in the future (2031)
    - country: the same codes with stray whitespace and casing ("US", "US ", " us")
    - annual_income: numbers stored as text with thousands separators ("52,000")
    - monthly_fee: 20% recorded in cents instead of dollars (×100)
    Clean on purpose: account_id (unique), region (consistent Title Case), tenure_days
    (skewed but plausible).
    """
    rng = np.random.default_rng(23)
    age = rng.integers(18, 80, n).astype(int)
    sentinel = rng.random(n)
    age = np.where(sentinel < 0.04, 999, np.where(sentinel < 0.05, -1, age))
    signup = pd.Timestamp("2019-01-01") + pd.to_timedelta(rng.integers(0, 6 * 365, n), unit="D")
    future = rng.random(n) < 0.03
    signup = signup.where(~future, pd.Timestamp("2031-06-01") + pd.to_timedelta(rng.integers(0, 200, n), unit="D"))
    country = rng.choice(["US", "US ", " us", "GB", "GB ", "DE", "FR"], n,
                         p=[0.40, 0.08, 0.04, 0.20, 0.04, 0.14, 0.10])
    income = (rng.lognormal(10.9, 0.45, n) // 1000 * 1000).astype(int)
    income_txt = np.array([f"{v:,}" for v in income], dtype=object)
    income_txt[rng.random(n) < 0.06] = "N/A"
    email = np.array([f"user{i}@example.com" for i in range(n)], dtype=object)
    email[rng.random(n) < 0.07] = ""
    fee = rng.choice([19.0, 49.0, 99.0], n, p=[0.5, 0.35, 0.15])
    fee = np.where(rng.random(n) < 0.20, fee * 100, fee)
    return pd.DataFrame({
        "account_id":    [f"A{i:07d}" for i in rng.permutation(n)],
        "email":         email,
        "age":           age,
        "country":       country,
        "region":        rng.choice(["North", "South", "East", "West"], n),
        "signup_date":   signup,
        "annual_income": income_txt,
        "monthly_fee":   fee,
        "tenure_days":   rng.gamma(1.5, 300, n).round(0).astype(int),
    })


def _gen_subscription_churn(n: int = 2000) -> pd.DataFrame:
    """
    Subscription customers, target `churned` (~20%). Leakage that column names don't give away:
    - status_code: categorical written by the cancellation flow ('C2' ≈ only churned customers)
    - engagement_index_v2: recomputed after the churn flag; separates the classes almost
      perfectly but non-linearly, so Pearson r stays well under 0.8
    - credit_balance: pro-rata refund credit, > 0 only for churned customers (zero, not null)
    Legitimate, genuinely predictive features: tenure_months, last_login_days,
    support_tickets_90d, monthly_spend, plan.
    """
    rng = np.random.default_rng(11)
    churned = rng.random(n) < 0.2
    tenure = np.where(churned, rng.gamma(2.0, 6.0, n), rng.gamma(3.0, 10.0, n)).round(0) + 1
    last_login = np.where(churned, rng.exponential(18, n), rng.exponential(6, n)).round(0)
    tickets = np.where(churned, rng.poisson(2.2, n), rng.poisson(0.9, n))
    plan = rng.choice(["basic", "pro", "team"], n, p=[0.5, 0.35, 0.15])
    spend = (np.select([plan == "basic", plan == "pro"], [19.0, 49.0], 99.0)
             * rng.uniform(0.9, 1.1, n)).round(2)
    status = np.where(churned, rng.choice(["C2", "A1"], n, p=[0.85, 0.15]),
                      rng.choice(["A1", "B4"], n, p=[0.97, 0.03]))
    # churned all below 0.03, retained above it with a heavy tail: near-perfect rank
    # separation, but the tail keeps Pearson r low
    engagement = np.where(churned, rng.uniform(0.0, 0.03, n),
                          0.03 + rng.pareto(1.3, n) * 0.08).round(4)
    credit = np.where(churned & (rng.random(n) < 0.6), rng.uniform(5, 90, n), 0.0).round(2)
    return pd.DataFrame({
        "customer_ref":        [f"CR{i:06d}" for i in rng.permutation(n)],
        "plan":                plan,
        "tenure_months":       tenure.astype(int),
        "monthly_spend":       spend,
        "last_login_days":     last_login.astype(int),
        "support_tickets_90d": tickets,
        "status_code":         status,
        "engagement_index_v2": engagement,
        "credit_balance":      credit,
        "churned":             churned,
    })


def _gen_ecommerce_leakage(n: int = 700) -> pd.DataFrame:
    """E-commerce sessions with two planted leakage features for leakage_detector benchmarks.

    Planted leakage:
    - conversion_score: float [0.75-1.0] for converted=True, [0.0-0.35] for False
      → point-biserial correlation with target ≈ 0.894 (direct leakage)
    - order_id: non-null only when converted=True
      → conditional_null_pct: {True: 0%, False: 100%} (proxy/post-event leakage)
    """
    rng = np.random.default_rng(55)
    converted = rng.choice([True, False], n, p=[0.08, 0.92])

    rng2 = np.random.default_rng(99)
    conversion_score = np.where(
        converted,
        rng2.uniform(0.75, 1.0, n),
        rng2.uniform(0.0, 0.35, n),
    ).round(4)

    order_counter = 0
    order_ids = []
    for c in converted:
        if c:
            order_counter += 1
            order_ids.append(f"ORD{order_counter:06d}")
        else:
            order_ids.append(None)

    return pd.DataFrame({
        "session_id":        [f"S{i:06d}" for i in range(1, n + 1)],
        "user_id":           [f"U{rng.integers(1, 400):05d}" if rng.random() > 0.1 else None
                              for _ in range(n)],
        "channel":           rng.choice(["organic", "paid_search", "social", "email", "direct", None],
                                        n, p=[0.3, 0.25, 0.2, 0.1, 0.1, 0.05]),
        "device":            rng.choice(["mobile", "desktop", "tablet"], n, p=[0.55, 0.38, 0.07]),
        "page_views":        rng.integers(1, 30, n),
        "time_on_site_s":    np.where(rng.random(n) > 0.05, rng.exponential(180, n).round(0).clip(10, 1200), None),
        "cart_value":        np.where(rng.random(n) > 0.3, rng.exponential(75, n).round(2), 0.0),
        "conversion_score":  conversion_score,   # LEAKAGE: derived from target
        "order_id":          order_ids,           # LEAKAGE: only exists post-conversion
        "converted":         converted,
        "session_date":      pd.date_range("2024-03-01", periods=n, freq="1h"),
    })


def _gen_transactions_for_join(n: int = 600) -> pd.DataFrame:
    """Left side of many-to-many join — transactions with duplicate user_ids on left."""
    rng = np.random.default_rng(17)
    base_users = [f"U{rng.integers(1, 80):04d}" for _ in range(n)]
    # Inject ~8% duplicate transaction rows (same user_id appears multiple times legitimately,
    # but combined with duplicate users on the right creates many-to-many)
    return pd.DataFrame({
        "transaction_id": [f"T{i:06d}" for i in range(1, n + 1)],
        "user_id":        base_users,
        "amount":         rng.exponential(200, n).round(2),
        "txn_date":       pd.date_range("2024-01-01", periods=n, freq="2h"),
    })


def _gen_users_with_dupes(n: int = 80) -> pd.DataFrame:
    """Right side of many-to-many join — users table with ~15% duplicate user_ids."""
    rng = np.random.default_rng(19)
    base = pd.DataFrame({
        "user_id":  [f"U{i:04d}" for i in range(1, n + 1)],
        "name":     [f"User {i}" for i in range(1, n + 1)],
        "segment":  rng.choice(["premium", "standard", "trial"], n, p=[0.2, 0.6, 0.2]),
        "country":  rng.choice(["US", "UK", "CA"], n),
    })
    dupes = base.sample(int(n * 0.15), random_state=99)
    return pd.concat([base, dupes], ignore_index=True)
