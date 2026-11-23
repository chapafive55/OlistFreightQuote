"""
freight_pipeline.py - Olist freight quote model (code appendix)
Gavin Chapa

Predicts the freight charge (R$) for a delivered, single-seller Olist order and
compares two models on a time-based split:
    Model A: Ridge regression (log inputs + one-hot states)
    Model B: Histogram gradient boosting (scikit-learn HistGradientBoostingRegressor)

Every number in the report is printed by this script.

How to run (from the repository folder):
    pip install pandas numpy scikit-learn
    python freight_pipeline.py                  # Olist CSVs in ./data
    python freight_pipeline.py --data <folder>  # or point at the folder with the CSVs

Needs these Kaggle Olist CSVs (original file names, plain .csv or zipped .csv.zip):
olist_orders_dataset, olist_order_items_dataset, olist_products_dataset,
olist_customers_dataset, olist_sellers_dataset, olist_geolocation_dataset.
The repository already includes them, zipped, in ./data.
"""
import argparse
import json
import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
TABLES = ["orders", "order_items", "products", "customers", "sellers", "geolocation"]
SEED = 42


# ----------------------------------------------------------------------------
# 1. Data: load the CSVs into SQLite and run freight_pull.sql
# ----------------------------------------------------------------------------
def hav_km(lat1, lng1, lat2, lng2):
    """Haversine distance in km (registered as a SQL function)."""
    if None in (lat1, lng1, lat2, lng2):
        return None
    lat1, lng1, lat2, lng2 = map(math.radians, (lat1, lng1, lat2, lng2))
    a = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lng2 - lng1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(a))


def load_data(data_dir):
    con = sqlite3.connect(":memory:")
    for t in TABLES:
        name = f"olist_{t}_dataset"
        path = Path(data_dir) / f"{name}.csv"
        if not path.exists():                       # the repo stores them zipped (GitHub upload limit)
            path = Path(data_dir) / f"{name}.csv.zip"
        if not path.exists():
            raise SystemExit(f"Missing {name}.csv(.zip) in {data_dir}. Put the Olist CSVs in ./data or pass --data <folder>.")
        pd.read_csv(path).to_sql(name, con, index=False)
    con.create_function("hav_km", 4, hav_km)

    counts = con.execute("""
        SELECT COUNT(*), SUM(n_sellers = 1) FROM (
            SELECT o.order_id, COUNT(DISTINCT oi.seller_id) AS n_sellers
            FROM olist_orders_dataset o
            JOIN olist_order_items_dataset oi ON oi.order_id = o.order_id
            WHERE o.order_status = 'delivered'
            GROUP BY o.order_id)""").fetchone()
    df = pd.read_sql((HERE / "freight_pull.sql").read_text(), con)
    con.close()
    df["purchase_ts"] = pd.to_datetime(df["purchase_ts"])
    return df.sort_values("purchase_ts", kind="mergesort").reset_index(drop=True), counts


def time_split(df):
    """70/15/15 by purchase date: train on the past, validate and test on later orders."""
    n = len(df)
    n_tr, n_va = int(0.70 * n), int(0.15 * n)
    return df.iloc[:n_tr], df.iloc[n_tr:n_tr + n_va], df.iloc[n_tr + n_va:]


# ----------------------------------------------------------------------------
# 2. Features
# ----------------------------------------------------------------------------
LOG_COLS = ["weight_g", "volume_cm3", "price_sum", "distance_km"]   # skewed -> log1p
RAW_COLS = ["n_items", "same_state"]
STATE_COLS = ["seller_state", "customer_state"]


def ridge_features(d):
    X = d[RAW_COLS + STATE_COLS].copy()
    for c in LOG_COLS:
        X["log_" + c] = np.log1p(d[c])
    return X


def make_ridge(alpha):
    num = RAW_COLS + ["log_" + c for c in LOG_COLS]
    prep = ColumnTransformer([("num", StandardScaler(), num),
                              ("state", OneHotEncoder(handle_unknown="ignore"), STATE_COLS)])
    return make_pipeline(prep, Ridge(alpha=alpha))


def make_tree_features(all_states):
    def tree_features(d):
        X = d[LOG_COLS + RAW_COLS].copy()
        for c in STATE_COLS:   # native categorical handling, no one-hot needed
            X[c] = pd.Categorical(d[c], categories=all_states)
        return X
    return tree_features


def make_hgb(lr, depth, iters):
    return HistGradientBoostingRegressor(learning_rate=lr, max_depth=depth, max_iter=iters,
                                         min_samples_leaf=30, l2_regularization=1.0,
                                         categorical_features="from_dtype",
                                         early_stopping=False, random_state=SEED)


# ----------------------------------------------------------------------------
# 3. Metrics
# ----------------------------------------------------------------------------
def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = p - y                      # + = over-quote, - = under-quote
    return {"MAE": np.mean(np.abs(err)),
            "RMSE": math.sqrt(np.mean(err ** 2)),
            "R2": 1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2),
            "bias": np.mean(err),
            "within_5": np.mean(np.abs(err) <= 5),
            "under_5": np.mean(err < -5),
            "over_5": np.mean(err > 5)}


def psi(expected, actual, bins=10):
    """Population stability index using train deciles as bins."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def show(title, m):
    print(f"  {title:<22} MAE R${m['MAE']:.2f} | RMSE R${m['RMSE']:.2f} | R2 {m['R2']:.3f} | "
          f"bias R${m['bias']:+.2f} | within5 {m['within_5']:.1%} | under5 {m['under_5']:.1%} | over5 {m['over_5']:.1%}")


# ----------------------------------------------------------------------------
# 4. Run everything
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "data"), help="folder with the Olist CSVs")
    args = ap.parse_args()

    df, (n_delivered, n_single) = load_data(args.data)
    tr, va, te = time_split(df)
    y_tr, y_va, y_te = tr.freight_value, va.freight_value, te.freight_value
    out = {}

    print("\n== DATA ==")
    print(f"  delivered orders {n_delivered:,} | single-seller {n_single:,} ({n_single / n_delivered:.1%}) | "
          f"dropped {n_single - len(df):,} | final rows {len(df):,}")
    for name, d in [("train", tr), ("validation", va), ("test", te)]:
        print(f"  {name:<10} {len(d):>6,} rows  {d.purchase_ts.min():%Y-%m-%d} to {d.purchase_ts.max():%Y-%m-%d}")
    out["data"] = {"delivered": n_delivered, "single_seller": n_single, "rows": len(df),
                   "train": len(tr), "val": len(va), "test": len(te)}

    # ---- Model A: Ridge -------------------------------------------------------
    print("\n== MODEL A: RIDGE ==")
    Xa_tr, Xa_va, Xa_te = ridge_features(tr), ridge_features(va), ridge_features(te)
    ridge_val = {}
    for alpha in [0.01, 0.1, 1, 10, 100]:
        ridge_val[alpha] = metrics(y_va, make_ridge(alpha).fit(Xa_tr, y_tr).predict(Xa_va))["RMSE"]
        print(f"  alpha {alpha:<6} val RMSE R${ridge_val[alpha]:.2f}")
    best_alpha = min(ridge_val, key=ridge_val.get)
    model_a = make_ridge(best_alpha).fit(Xa_tr, y_tr)
    pa_va, pa_te = model_a.predict(Xa_va), model_a.predict(Xa_te)
    print(f"  best alpha {best_alpha}")
    show("validation", metrics(y_va, pa_va))
    show("TEST", metrics(y_te, pa_te))

    # ---- Model B: gradient boosting -----------------------------------------
    print("\n== MODEL B: GRADIENT BOOSTING (18-setting grid) ==")
    states = sorted(set(df.seller_state) | set(df.customer_state))
    tree_features = make_tree_features(states)
    Xb_tr, Xb_va, Xb_te = tree_features(tr), tree_features(va), tree_features(te)
    grid = {}
    for lr in [0.03, 0.06, 0.1]:
        for depth in [None, 8, 4]:
            for iters in [200, 400]:
                grid[(lr, depth, iters)] = metrics(y_va, make_hgb(lr, depth, iters).fit(Xb_tr, y_tr).predict(Xb_va))["RMSE"]
                print(f"  lr {lr:<5} depth {str(depth):<5} iters {iters:<4} val RMSE R${grid[(lr, depth, iters)]:.2f}")
    best = min(grid, key=grid.get)
    model_b = make_hgb(*best).fit(Xb_tr, y_tr)
    pb_va, pb_te = model_b.predict(Xb_va), model_b.predict(Xb_te)
    print(f"  best: learning rate {best[0]}, max depth {best[1]}, iterations {best[2]} "
          f"(min leaf 30, L2 1.0, max leaf nodes {model_b.max_leaf_nodes})")
    show("validation", metrics(y_va, pb_va))
    show("TEST", metrics(y_te, pb_te))

    # ---- Comparison ---------------------------------------------------------
    print("\n== COMPARISON (TEST) ==")
    base = np.full(len(te), y_tr.mean())          # mean-only baseline: train mean
    mA, mB, m0 = metrics(y_te, pa_te), metrics(y_te, pb_te), metrics(y_te, base)
    show("Model A (Ridge)", mA)
    show("Model B (GBM)", mB)
    show("Mean-only baseline", m0)
    print(f"  Model B MAE is {1 - mB['MAE'] / mA['MAE']:.0%} lower than Model A (R${mA['MAE'] - mB['MAE']:.2f} per order)")
    sq = (pb_te - y_te.values) ** 2
    worst = np.argsort(sq)[::-1][:int(round(0.01 * len(te)))]
    print(f"  worst 1% of test orders = {sq[worst].sum() / sq.sum():.0%} of Model B squared error; "
          f"{(te.weight_g.values[worst] > 10_000).mean():.0%} of them weigh over 10 kg")
    out["test"] = {"A": mA, "B": mB, "baseline": m0}

    # ---- Recommendation: safeguards, cushion, monitoring -------------------
    print("\n== RECOMMENDATION CHECKS (Model B, test) ==")
    err = np.abs(pb_te - y_te.values)
    heavy, multi = te.weight_g.values > 10_000, te.n_items.values > 1
    print(f"  orders over 10 kg: {heavy.mean():.1%} of test, MAE R${err[heavy].mean():.2f}")
    print(f"  multi-item orders: {multi.mean():.1%} of test, MAE R${err[multi].mean():.2f}")
    print(f"  everything else:   {(~heavy & ~multi).mean():.1%} of test, MAE R${err[~heavy & ~multi].mean():.2f}")
    cushion = float(np.mean(y_va - pb_va))       # validation shortfall
    mC = metrics(y_te, pb_te + cushion)
    print(f"  validation shortfall (cushion) R${cushion:.2f}")
    show("test + cushion", mC)

    month = te.purchase_ts.dt.to_period("M")
    va_month = va.purchase_ts.dt.to_period("M")
    print("  Model B signed bias by month (validation, then test):")
    for d, p, mo in [(va, pb_va, va_month), (te, pb_te, month)]:
        b = pd.Series(p - d.freight_value.values, index=d.index).groupby(mo.values).agg(["mean", "size"])
        for k, r in b.iterrows():
            print(f"    {k}  bias R${r['mean']:+.2f}  (n={int(r['size']):,})")

    print("  PSI, train vs test:")
    for c in ["weight_g", "volume_cm3", "price_sum", "distance_km", "n_items"]:
        print(f"    {c:<12} {psi(tr[c], te[c]):.3f}")
    print(f"    {'freight':<12} {psi(y_tr, y_te):.3f}")

    basket = df[(df.n_items == 1) & df.weight_g.between(500, 2000) & df.distance_km.between(300, 800)]
    bm = basket.groupby(basket.purchase_ts.dt.to_period("M")).freight_value.median()
    feb = bm.loc["2018-02"]
    summer = basket[basket.purchase_ts.dt.to_period("M").isin(pd.period_range("2018-06", "2018-08", freq="M"))].freight_value.median()
    print(f"  benchmark basket (1 item, 0.5-2 kg, 300-800 km) median freight: Feb 2018 R${feb:.2f} -> "
          f"Jun-Aug 2018 R${summer:.2f} ({summer / feb - 1:+.0%})")

    # ---- Limits: what a random split would have shown -----------------------
    print("\n== LIMITS: RANDOM SPLIT INSTEAD OF TIME SPLIT ==")
    shuffled = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    rtr, rva, rte = time_split(shuffled)          # same 70/15/15 sizes, random order
    rA = metrics(rte.freight_value, make_ridge(best_alpha).fit(ridge_features(rtr), rtr.freight_value).predict(ridge_features(rte)))
    rB = metrics(rte.freight_value, make_hgb(*best).fit(tree_features(rtr), rtr.freight_value).predict(tree_features(rte)))
    show("Model A random split", rA)
    show("Model B random split", rB)

    (HERE / "results").mkdir(exist_ok=True)
    out.update({"best_alpha": best_alpha, "best_hgb": list(best), "cushion": cushion, "test_with_cushion": mC,
                "random_split": {"A": rA, "B": rB}})
    (HERE / "results" / "metrics.json").write_text(json.dumps(out, indent=2, default=float))
    print("\nSaved results/metrics.json")


if __name__ == "__main__":
    main()
