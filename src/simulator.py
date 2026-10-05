"""
Stage 6: Stream simulator.

Replays test-period transactions (Sep 9-10) into the API in timestamp order,
as if they were arriving live.

--reset first removes test-period edges from Neo4j and clears the API's live window,
so the graph only contains what has "happened" so far. To restore the full graph
afterwards (e.g. before re-running patterns.py), redo the neo4j-admin bulk import.

Usage (from project root, Neo4j + API running):
    python src/simulator.py --reset --n 5000 --rate 20          # one tx per request
    python src/simulator.py --reset --all --batch-size 10000    # full test window, micro-batches

In batch mode the run ends with an evaluation: the stream clock is moved to the end of
the test window, every active account is rescored, and live scores are compared with
the offline test results on the same accounts and labels.
"""
import argparse
import time

import numpy as np

import pandas as pd
import requests
from neo4j import GraphDatabase

from sklearn.metrics import average_precision_score, roc_auc_score

from config import PROCESSED, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, TEST_START, DATA_END

Q_DELETE = """
MATCH ()-[t:TX]->() WHERE t.ts >= $start
CALL (t) { DELETE t } IN TRANSACTIONS OF 50000 ROWS
"""


def reset(api):
    start = int(pd.Timestamp(TEST_START).value // 10**9)
    print(f"Removing test-period edges (ts >= {TEST_START}) from Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as s:
        s.run(Q_DELETE, start=start).consume()
        left = s.run("MATCH ()-[t:TX]->() RETURN count(t) AS n").single()["n"]
    driver.close()
    print(f"  graph now has {left:,} transactions")
    requests.post(f"{api}/admin/reset", timeout=300).raise_for_status()
    print("  API live window cleared and re-warmed with the previous 48h")


def load_test(n):
    cols = ["tx_id", "ts", "src", "dst", "amt_paid", "cur_paid", "amt_recv",
            "cur_recv", "fmt", "label", "split"]
    df = pd.read_parquet(PROCESSED / "transactions.parquet", columns=cols)
    df = df[df["split"] == "test"].sort_values("tx_id")
    df = df if n is None else df.head(n)
    for c in ["cur_paid", "cur_recv", "fmt"]:
        df[c] = df[c].astype(str)
    return df.drop(columns="split")


def metrics(y, s, ks):
    order = np.argsort(-s, kind="stable")
    row = {"PR-AUC": average_precision_score(y, s), "ROC-AUC": roc_auc_score(y, s)}
    for k in ks:
        row[f"recall@{k}"] = y[order[:k]].sum() / y.sum()
    return row


def evaluate(api, session):
    print(f"\nEnd-of-window rescore (clock -> {DATA_END}) ...")
    r = session.post(f"{api}/admin/rescore-all", params={"at": DATA_END}, timeout=3600).json()
    print(f"  rescored {r['accounts']:,} accounts in {r['seconds']}s")
    live = session.get(f"{api}/scores", timeout=600).json()
    live = pd.DataFrame({"account": live["accounts"], "live": live["scores"]})

    feats = pd.read_parquet(PROCESSED / "features_test.parquet",
                            columns=["account", "label", "rule_any"])
    off = pd.read_parquet(PROCESSED / "test_scores.parquet", columns=["account", "score"])
    m = feats.merge(off, on="account", how="left").merge(live, on="account", how="left")
    missing = m["live"].isna().sum()
    m["live"] = m["live"].fillna(0.0)
    y = m["label"].to_numpy()
    k_rules = int((m["rule_any"] > 0).sum())
    ks = [100, 1000, k_rules]
    rows = {"rules": metrics(y, m["rule_any"].to_numpy(dtype=float), ks),
            "offline (test-window graph features)": metrics(y, m["score"].to_numpy(), ks),
            "live (batch graph features from Sep 7-8)": metrics(y, m["live"].to_numpy(), ks)}
    table = pd.DataFrame(rows).T
    print(f"\n=== OFFLINE vs LIVE: {len(m):,} test accounts, {int(y.sum()):,} launderers "
          f"({missing:,} without a live score) ===")
    print(table.to_string(float_format="{:.3f}".format))
    out = PROCESSED.parent.parent / "reports" / "live_vs_offline.md"
    out.write_text("# Offline vs live (end-of-window rescore)\n\n```\n"
                   + table.to_string(float_format="{:.3f}".format) + "\n```\n")
    print(f"\nSaved {out}")


def main_batch(n, batch_size, api, do_reset):
    session = requests.Session()
    session.get(f"{api}/health", timeout=10).raise_for_status()
    if do_reset:
        reset(api)
    df = load_test(n)
    df["ts"] = df["ts"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    records = df.to_dict("records")
    print(f"Replaying {len(records):,} transactions in batches of {batch_size:,} ...")
    t0 = time.time()
    for i in range(0, len(records), batch_size):
        info = session.post(f"{api}/transactions/batch",
                            json={"transactions": records[i:i + batch_size]},
                            timeout=3600).json()
        done = min(i + batch_size, len(records))
        print(f"  {done:,}/{len(records):,} tx | batch: {info['total_ms']:,} ms "
              f"(write {info['write_ms']:,}, score {info['score_ms']:,}), "
              f"{info['accounts_scored']:,} accounts rescored | "
              f"overall {done / (time.time() - t0):,.0f} tx/s")
    print(f"Replay done in {time.time() - t0:.0f}s")
    evaluate(api, session)


def main(n, rate, api, do_reset):
    requests.get(f"{api}/health", timeout=10).raise_for_status()
    if do_reset:
        reset(api)

    df = load_test(n)
    print(f"Replaying {len(df):,} transactions at up to {rate}/s ...")

    session = requests.Session()
    t0, errors = time.time(), 0
    for i, r in enumerate(df.itertuples(index=False), 1):
        payload = {
            "tx_id": int(r.tx_id), "ts": r.ts.isoformat(), "src": r.src, "dst": r.dst,
            "amt_paid": float(r.amt_paid), "cur_paid": r.cur_paid,
            "amt_recv": float(r.amt_recv), "cur_recv": r.cur_recv,
            "fmt": r.fmt, "label": int(r.label),
        }
        try:
            session.post(f"{api}/transaction", json=payload, timeout=30).raise_for_status()
        except requests.RequestException as e:
            errors += 1
            if errors <= 5:
                print(f"  error on tx {r.tx_id}: {e}")
        target = t0 + i / rate
        if (wait := target - time.time()) > 0:
            time.sleep(wait)
        if i % 500 == 0 or i == len(df):
            st = session.get(f"{api}/stats", timeout=10).json()
            print(f"  {i:,} tx | {i / (time.time() - t0):.1f} tx/s | "
                  f"p50 {st['latency_ms_p50']} ms, p95 {st['latency_ms_p95']} ms | "
                  f"{st['accounts_scored']:,} accounts scored")

    print(f"\nDone: {len(df):,} tx in {time.time() - t0:.0f}s, {errors} errors")
    top = session.get(f"{api}/alerts", params={"n": 10}, timeout=10).json()
    print("\nTop 10 alerts (demo_label 1 = known launderer, never seen by the model):")
    for a in top:
        reason = a["reasons"][0]["description"] if a["reasons"] else "-"
        print(f"  {a['score']:.3f}  {a['account']:20s} label={a['demo_label']}  "
              f"({a['graph_features']})  top reason: {reason}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000, help="transactions to replay")
    ap.add_argument("--all", action="store_true", help="replay the whole test window")
    ap.add_argument("--batch-size", type=int, default=1,
                    help=">1 = micro-batch mode (e.g. 10000), ends with offline-vs-live evaluation")
    ap.add_argument("--rate", type=float, default=20, help="max transactions per second")
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--reset", action="store_true",
                    help="remove test-period edges from Neo4j and clear the live window first")
    a = ap.parse_args()
    n = None if a.all else a.n
    if a.batch_size > 1:
        main_batch(n, a.batch_size, a.api, a.reset)
    else:
        main(n if n is not None else 10**9, a.rate, a.api, a.reset)
