"""
Stage 4: Account-level feature engineering.

One row per account per split. Each split's features use ONLY that split's
transactions (no peeking at the future). Counts and amounts are converted to
per-day rates because the splits have different lengths (6 / 2 / 2 days).

Feature groups
  volume        tx counts and distinct counterparties, in and out
  amount        USD-normalised sums / means / std / max, flow ratio
  behaviour     payment-format mix, cross-currency %, cross-bank %, self-transfer %
  timing        max tx per hour, active-day fraction, pass-through speed
  graph         PageRank, strongly / weakly connected component size
  neighbourhood mean of neighbours' features (a hand-built GNN-style aggregation)
  rules         Stage 3 rule scores

Outputs
  data/processed/features_train.parquet  (also _val, _test)

Usage (from project root):
    python src/features.py
"""
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import roc_auc_score

from config import PROCESSED

SPLIT_DAYS = {"train": 6, "val": 2, "test": 2}
USD = "US Dollar"


# ---------------------------------------------------------------- loading

def load():
    cols = ["ts", "src", "dst", "amt_paid", "cur_paid", "amt_recv", "cur_recv",
            "fmt", "label", "split", "cross_currency", "cross_bank"]
    df = pd.read_parquet(PROCESSED / "transactions.parquet", columns=cols)
    for c in ["cur_paid", "cur_recv", "fmt", "split"]:
        df[c] = df[c].astype(str)

    # Integer account codes: much faster and lighter than string keys
    codes, accounts = pd.factorize(pd.concat([df["src"], df["dst"]], ignore_index=True))
    n = len(df)
    df["s"], df["t"] = codes[:n].astype("int32"), codes[n:].astype("int32")
    df = df.drop(columns=["src", "dst"])
    return df, np.asarray(accounts)


def usd_rates(df):
    """Exchange rates estimated from the data itself, using train-split cross-currency
    transactions where one side is USD (median ratio, robust to outliers)."""
    tr = df[(df["split"] == "train") & (df["amt_paid"] > 0) & (df["amt_recv"] > 0)]
    a = tr[(tr["cur_recv"] == USD) & (tr["cur_paid"] != USD)]
    to_usd = (a["amt_recv"] / a["amt_paid"]).groupby(a["cur_paid"]).median()
    b = tr[(tr["cur_paid"] == USD) & (tr["cur_recv"] != USD)]
    from_usd = (b["amt_paid"] / b["amt_recv"]).groupby(b["cur_recv"]).median()

    rates = {USD: 1.0, **from_usd.to_dict(), **to_usd.to_dict()}
    all_cur = set(df["cur_paid"]) | set(df["cur_recv"])
    missing = all_cur - set(rates)
    if missing:
        print(f"  WARNING: no USD rate for {missing}; their amounts become NaN")
    return rates


# ---------------------------------------------------------------- features

def tabular_features(d, days):
    no_self = d[d["s"] != d["t"]]
    out = pd.DataFrame({"acct": no_self["s"], "cp": no_self["t"], "usd": no_self["usd_paid"],
                        "ts": no_self["ts"]})
    inn = pd.DataFrame({"acct": no_self["t"], "cp": no_self["s"], "usd": no_self["usd_recv"],
                        "ts": no_self["ts"]})

    feats = []
    for name, side in [("out", out), ("in", inn)]:
        g = side.groupby("acct")
        f = pd.DataFrame({
            f"{name}_cnt": g.size() / days,
            f"{name}_uniq": g["cp"].nunique() / days,
            f"{name}_sum": np.log1p(g["usd"].sum() / days),
            f"{name}_mean": np.log1p(g["usd"].mean()),
            f"{name}_std": np.log1p(g["usd"].std().fillna(0)),
            f"{name}_max": np.log1p(g["usd"].max()),
        })
        feats.append(f)
    F = pd.concat(feats, axis=1)
    for name in ["out", "in"]:
        F[[f"{name}_cnt", f"{name}_uniq"]] = F[[f"{name}_cnt", f"{name}_uniq"]].fillna(0)
    F["log_flow_ratio"] = F["out_sum"].fillna(0) - F["in_sum"].fillna(0)   # log(out/in)
    F["uniq_ratio"] = F["out_uniq"] / (F["in_uniq"] + F["out_uniq"] + 1e-9)

    # Behaviour over every tx touching the account (both directions, incl. self)
    both = pd.concat([
        d[["s", "fmt", "cross_currency", "cross_bank", "ts"]].rename(columns={"s": "acct"}),
        d[["t", "fmt", "cross_currency", "cross_bank", "ts"]].rename(columns={"t": "acct"}),
    ], ignore_index=True)
    fmt_share = pd.crosstab(both["acct"], both["fmt"], normalize="index")
    fmt_share.columns = [f"fmt_{c.lower().replace(' ', '_')}" for c in fmt_share.columns]
    g = both.groupby("acct")
    B = pd.DataFrame({
        "cross_currency_share": g["cross_currency"].mean(),
        "cross_bank_share": g["cross_bank"].mean(),
    })
    B["active_day_frac"] = both.assign(day=both["ts"].dt.normalize()).groupby("acct")["day"].nunique() / days
    self_cnt = d[d["s"] == d["t"]].groupby("s").size()
    B["self_share"] = (self_cnt.reindex(B.index).fillna(0) / g.size()).astype(float)
    hourly = both.assign(h=both["ts"].dt.floor("h")).groupby(["acct", "h"]).size()
    B["max_tx_per_hour"] = hourly.groupby(level=0).max()

    # Pass-through speed: for each outgoing tx, hours since the latest incoming tx
    o = out[["acct", "ts"]].sort_values("ts")
    i = inn[["acct", "ts"]].rename(columns={"ts": "in_ts"}).sort_values("in_ts")
    m = pd.merge_asof(o, i, left_on="ts", right_on="in_ts", by="acct", direction="backward")
    m["gap_h"] = (m["ts"] - m["in_ts"]).dt.total_seconds() / 3600
    m["fast"] = (m["gap_h"] <= 24).astype(float)
    pt = m.groupby("acct").agg(pt_median_h=("gap_h", "median"), pt_frac24=("fast", "mean"))

    return F.join([B, fmt_share, pt], how="outer")


def pagerank(A, alpha=0.85, iters=60, tol=1e-10):
    n = A.shape[0]
    outdeg = np.asarray(A.sum(axis=1)).ravel()
    inv = np.divide(1.0, outdeg, out=np.zeros_like(outdeg, dtype=float), where=outdeg > 0)
    PT = (sp.diags(inv) @ A).T.tocsr()
    r = np.full(n, 1.0 / n)
    dangling = outdeg == 0
    for _ in range(iters):
        r_new = alpha * (PT @ r + r[dangling].sum() / n) + (1 - alpha) / n
        if np.abs(r_new - r).sum() < tol:
            r = r_new
            break
        r = r_new
    return r * n   # scale so the average node is ~1


def graph_features(d, n_nodes):
    e = d[d["s"] != d["t"]]
    A = sp.csr_matrix((np.ones(len(e), dtype=np.float32), (e["s"], e["t"])),
                      shape=(n_nodes, n_nodes))
    A.data[:] = 1.0                                   # unique edges only
    U = ((A + A.T) > 0).astype(np.float32).tocsr()    # undirected neighbours

    _, scc = connected_components(A, directed=True, connection="strong")
    _, wcc = connected_components(A, directed=True, connection="weak")
    G = pd.DataFrame({
        "pagerank": pagerank(A),
        "scc_size": np.bincount(scc)[scc],
        "wcc_size": np.log1p(np.bincount(wcc)[wcc]),
        "n_neighbours": np.asarray(U.sum(axis=1)).ravel(),
    })
    G["log_scc_size"] = np.log1p(G.pop("scc_size"))
    G["in_cycle_scc"] = (G["log_scc_size"] > np.log1p(1)).astype(int)
    return G, U


NBR_COLS = ["log_flow_ratio", "pt_frac24", "out_uniq", "in_uniq",
            "cross_currency_share", "pagerank", "max_tx_per_hour", "rule_any"]


def neighbour_features(F, U):
    """Mean of each neighbour's feature value: the aggregation step a GNN learns."""
    deg = np.asarray(U.sum(axis=1)).ravel()
    out = {}
    for c in NBR_COLS:
        v = np.zeros(U.shape[0], dtype=np.float32)
        col = F[c]
        v[col.index.to_numpy()] = col.fillna(col.median()).to_numpy()
        s = U @ v
        out[f"nbr_mean_{c}"] = np.divide(s, deg, out=np.zeros_like(s), where=deg > 0)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- main

def main():
    t0 = time.time()
    print("Loading transactions...")
    df, accounts = load()
    n_nodes = len(accounts)
    print(f"  {len(df):,} tx, {n_nodes:,} accounts")

    print("Estimating USD exchange rates from the data...")
    rates = usd_rates(df)
    df["usd_paid"] = df["amt_paid"] * df["cur_paid"].map(rates)
    df["usd_recv"] = df["amt_recv"] * df["cur_recv"].map(rates)

    rules = pd.read_parquet(PROCESSED / "rule_flags.parquet")
    rule_cols = [c for c in rules.columns if c not in ("account", "split")]
    acct_index = pd.Series(np.arange(n_nodes), index=accounts)
    all_fmt_cols = None

    for split, days in SPLIT_DAYS.items():
        ts = time.time()
        d = df[df["split"] == split]
        print(f"\n=== {split.upper()} ({len(d):,} tx, {days} days) ===")

        F = tabular_features(d, days)
        G, U = graph_features(d, n_nodes)
        F = F.join(G, how="left")

        r = rules[rules["split"] == split].copy()
        r["acct"] = acct_index.reindex(r["account"]).to_numpy()
        r = r.dropna(subset=["acct"]).set_index(r["acct"].astype(int).rename(None))[rule_cols]
        F = F.join(r, how="left")
        F[rule_cols] = F[rule_cols].fillna(0)
        F["rule_any"] = (F[rule_cols] > 0).sum(axis=1)

        N = neighbour_features(F, U)
        F = F.join(N, how="left")

        # Keep payment-format columns identical across splits
        fmt_cols = sorted(c for c in F.columns if c.startswith("fmt_"))
        if all_fmt_cols is None:
            all_fmt_cols = fmt_cols
        for c in all_fmt_cols:
            if c not in F:
                F[c] = 0.0
        F = F.drop(columns=[c for c in fmt_cols if c not in all_fmt_cols])
        F[all_fmt_cols] = F[all_fmt_cols].fillna(0)

        lab = d[d["label"] == 1]
        pos = set(lab["s"]) | set(lab["t"])
        F["label"] = F.index.isin(pos).astype("int8")
        F.insert(0, "account", accounts[F.index.to_numpy()])
        F = F.reset_index(drop=True)

        F.to_parquet(PROCESSED / f"features_{split}.parquet", index=False)
        n_feat = F.shape[1] - 2
        print(f"  {len(F):,} accounts x {n_feat} features, "
              f"{F['label'].sum():,} laundering ({F['label'].mean():.3%})  [{time.time() - ts:.0f}s]")

        if split == "train":
            X = F.drop(columns=["account", "label"])
            aucs = {c: roc_auc_score(F["label"], X[c].fillna(X[c].median()))
                    for c in X.columns if X[c].nunique() > 1}
            top = pd.Series(aucs).sub(0.5).abs().add(0.5).sort_values(ascending=False)
            print("\n  Top 15 single-feature AUCs on train (0.5 = useless, 1.0 = perfect):")
            for c, v in top.head(15).items():
                print(f"    {c:32s} {v:.3f}")

    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
