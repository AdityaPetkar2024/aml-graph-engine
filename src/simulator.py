"""
Stage 6: Stream simulator.

Replays test-period transactions (Sep 9-10) into the API in timestamp order,
as if they were arriving live.

--reset first removes test-period edges from Neo4j and clears the API's live window,
so the graph only contains what has "happened" so far. To restore the full graph
afterwards (e.g. before re-running patterns.py), redo the neo4j-admin bulk import.

Usage (from project root, Neo4j + API running):
    python src/simulator.py --reset --n 5000 --rate 20
"""
import argparse
import time

import pandas as pd
import requests
from neo4j import GraphDatabase

from config import PROCESSED, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, TEST_START

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


def main(n, rate, api, do_reset):
    requests.get(f"{api}/health", timeout=10).raise_for_status()
    if do_reset:
        reset(api)

    cols = ["tx_id", "ts", "src", "dst", "amt_paid", "cur_paid", "amt_recv",
            "cur_recv", "fmt", "label", "split"]
    df = pd.read_parquet(PROCESSED / "transactions.parquet", columns=cols)
    df = df[df["split"] == "test"].sort_values("tx_id").head(n)
    for c in ["cur_paid", "cur_recv", "fmt"]:
        df[c] = df[c].astype(str)
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
    ap.add_argument("--rate", type=float, default=20, help="max transactions per second")
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--reset", action="store_true",
                    help="remove test-period edges from Neo4j and clear the live window first")
    a = ap.parse_args()
    main(a.n, a.rate, a.api, a.reset)
