"""
Synthetic rows for the retail_marketing catalog (scaled down; deterministic).

Planted problems that only show up when the data is checked, not in the metadata:
  - campaign_exposure: ~10% of recipients aren't registered customers (their email_hash isn't in
    customers), and customers who got both the winter and spring sends appear twice.
  - customer_orders: ~20% of order lines have no customer_id (anonymous till purchases).
  - customer_features_daily: snapshots start on 2026-04-01, after the spring campaign (March
    2026) — so covariates measured before exposure don't exist in this table.
  - Targeting: the spring campaign went mostly to customers who already spent more
    (confounding a naive comparison).
Tables without a generator here get generic rows from their column types.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

N_CUSTOMERS = 5000


def _hash(i: int) -> str:
    return hashlib.sha256(f"user{i}@example.com".encode()).hexdigest()[:16]


def _customers() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(N_CUSTOMERS)],
        "email_hash": [_hash(i) for i in range(N_CUSTOMERS)],
        "signup_date": pd.Timestamp("2019-01-01") + pd.to_timedelta(rng.integers(0, 2600, N_CUSTOMERS), unit="D"),
        "loyalty_member": rng.random(N_CUSTOMERS) < 0.3,
        "region": rng.choice(["North", "South", "East", "West"], N_CUSTOMERS),
        "birth_year": rng.integers(1950, 2006, N_CUSTOMERS),
    })


def _propensity() -> np.ndarray:
    """Baseline spend level per customer; drives both targeting and spend (confounder)."""
    return np.random.default_rng(2).gamma(2.0, 1.0, N_CUSTOMERS)


def _exposure() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    base = _propensity()
    p = 1 / (1 + np.exp(-(base - 2.0) * 1.5))            # higher spenders were targeted
    spring = np.flatnonzero(rng.random(N_CUSTOMERS) < p * 0.9)
    winter = rng.choice(N_CUSTOMERS, 1500, replace=False)
    strangers = np.arange(N_CUSTOMERS, N_CUSTOMERS + 600)  # recipients with no customer record

    def sends(ids, campaign, start, days):
        return pd.DataFrame({
            "email_hash": [_hash(int(i)) for i in ids], "campaign_id": campaign,
            "sent_at": pd.Timestamp(start) + pd.to_timedelta(rng.integers(0, days * 24, len(ids)), unit="h"),
            "channel": rng.choice(["email", "paid_social", "display"], len(ids), p=[0.6, 0.25, 0.15]),
            "opened": rng.random(len(ids)) < 0.4, "clicked": rng.random(len(ids)) < 0.08})
    return pd.concat([sends(spring, "SPRING26", "2026-03-02", 28), sends(strangers, "SPRING26", "2026-03-02", 28),
                      sends(winter, "WINTER26", "2026-01-12", 14)], ignore_index=True)


def _orders() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    n = 40_000
    base = _propensity()
    cust = rng.choice(N_CUSTOMERS, n, p=base / base.sum())
    amount = rng.gamma(2.0, 18.0, n) * (0.6 + 0.4 * base[cust] / base.mean())
    anonymous = rng.random(n) < 0.2
    return pd.DataFrame({
        "order_id": [f"O{i:06d}" for i in range(n)],
        "customer_id": np.where(anonymous, None, [f"C{c:05d}" for c in cust]),
        "order_date": pd.Timestamp("2025-10-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D"),
        "store_id": np.where(rng.random(n) < 0.5, None, [f"S{s:03d}" for s in rng.integers(0, 40, n)]),
        "sku": [f"SKU{s:04d}" for s in rng.integers(0, 800, n)],
        "quantity": rng.integers(1, 5, n),
        "net_amount_eur": amount.round(2),
    })


def _features_daily() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    base = _propensity()
    snaps = pd.date_range("2026-04-01", "2026-10-01", freq="MS")     # starts after the campaign
    ids = np.tile(np.arange(N_CUSTOMERS), len(snaps))
    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in ids],
        "snapshot_date": np.repeat(snaps, N_CUSTOMERS),
        "spend_prior_90d_eur": (base[ids] * 60 * rng.uniform(0.7, 1.3, len(ids))).round(2),
        "orders_prior_90d": rng.poisson(base[ids] * 1.5),
        "tenure_days": rng.integers(30, 2600, len(ids)),
    })


def _store_weekly() -> pd.DataFrame:
    rng = np.random.default_rng(6)
    weeks = pd.date_range("2026-01-05", "2026-09-28", freq="W-MON")
    stores = [f"S{s:03d}" for s in range(40)]
    df = pd.DataFrame([(s, w) for s in stores for w in weeks], columns=["store_id", "week_start"])
    df["units_sold"] = rng.poisson(900, len(df))
    df["revenue_eur"] = (df["units_sold"] * rng.uniform(17, 23, len(df))).round(2)
    return df


def _promo_calendar() -> pd.DataFrame:
    return pd.DataFrame({"store_id": [f"S{s:03d}" for s in range(20)], "promo_id": "SPRINGPROMO",
                         "start_date": pd.Timestamp("2026-04-06"), "end_date": pd.Timestamp("2026-05-31"),
                         "promo_type": "price_cut"})


def _subscriptions() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 3000
    start = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 900, n), unit="D")
    cancelled = rng.random(n) < 0.25
    end = start + pd.to_timedelta(rng.integers(30, 400, n), unit="D")
    return pd.DataFrame({
        "subscription_id": [f"SUB{i:05d}" for i in range(n)],
        "customer_id": [f"C{c:05d}" for c in rng.choice(N_CUSTOMERS, n, replace=False)],
        "plan": rng.choice(["basic", "plus"], n), "start_date": start,
        "end_date": pd.Series(end).where(cancelled), "status": np.where(cancelled, "cancelled", "active")})


def _stores() -> pd.DataFrame:
    rng = np.random.default_rng(8)
    return pd.DataFrame({"store_id": [f"S{s:03d}" for s in range(40)],
                         "region": rng.choice(["North", "South", "East", "West"], 40),
                         "format": rng.choice(["mall", "high_street", "outlet"], 40),
                         "opened_date": pd.Timestamp("2010-01-01") + pd.to_timedelta(rng.integers(0, 5000, 40), unit="D")})


def _campaign_costs() -> pd.DataFrame:
    rows = [("SPRING26", ch, s, "2026-03-01", "2026-03-31") for ch, s in
            (("email", 18_000), ("paid_social", 42_000), ("display", 25_000))]
    rows += [("WINTER26", "email", 9_000, "2026-01-10", "2026-01-26")]
    df = pd.DataFrame(rows, columns=["campaign_id", "channel", "spend_eur", "period_start", "period_end"])
    df[["period_start", "period_end"]] = df[["period_start", "period_end"]].apply(pd.to_datetime)
    return df


def _product_margins() -> pd.DataFrame:
    """Effective-dated: two rows per SKU (a price review on 2026-04-01)."""
    rng = np.random.default_rng(9)
    skus = [f"SKU{s:04d}" for s in range(800)]
    return pd.DataFrame({"sku": skus * 2, "margin_pct": rng.uniform(0.15, 0.55, 1600).round(3),
                         "valid_from": [pd.Timestamp("2025-01-01")] * 800 + [pd.Timestamp("2026-04-01")] * 800})


_GENERATORS = {
    "demo.marketing.campaign_costs": _campaign_costs,
    "demo.finance.product_margins": _product_margins,
    "demo.crm.customers": _customers,
    "demo.marketing.campaign_exposure": _exposure,
    "demo.sales.customer_orders": _orders,
    "demo.crm.customer_features_daily": _features_daily,
    "demo.sales.store_weekly_sales": _store_weekly,
    "demo.marketing.promo_calendar": _promo_calendar,
    "demo.crm.subscriptions": _subscriptions,
    "demo.ops.stores": _stores,
}


def generate(table: str) -> pd.DataFrame | None:
    """Rows for a table, or None to fall back to generic rows from its column types."""
    fn = _GENERATORS.get(table)
    return fn() if fn else None
