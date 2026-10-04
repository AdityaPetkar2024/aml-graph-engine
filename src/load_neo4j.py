"""
Stage 2: Post-import setup and verification.

Run AFTER the neo4j-admin bulk import and `docker compose up -d`.
- Creates indexes
- Checks node/edge counts against transactions.parquet
- Prints a quick look at laundering edges

Usage (from project root):
    python src/load_neo4j.py
"""
import time

import pandas as pd
from neo4j import GraphDatabase

from config import PROCESSED, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

INDEXES = [
    "CREATE INDEX acct_id IF NOT EXISTS FOR (a:Account) ON (a.id)",
    "CREATE INDEX tx_ts IF NOT EXISTS FOR ()-[t:TX]-() ON (t.ts)",
    "CREATE INDEX tx_label IF NOT EXISTS FOR ()-[t:TX]-() ON (t.label)",
]


def wait_for_neo4j(driver, retries=30):
    for i in range(retries):
        try:
            driver.verify_connectivity()
            return
        except Exception:
            print(f"  waiting for Neo4j... ({i + 1}/{retries})")
            time.sleep(5)
    raise RuntimeError("Neo4j not reachable. Is `docker compose up -d` running?")


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    print("Connecting to Neo4j...")
    wait_for_neo4j(driver)

    with driver.session() as s:
        print("Creating indexes...")
        for q in INDEXES:
            s.run(q)
        s.run("CALL db.awaitIndexes(600)")

        n_nodes = s.run("MATCH (a:Account) RETURN count(a) AS n").single()["n"]
        rec = s.run("MATCH ()-[t:TX]->() RETURN count(t) AS n, sum(t.label) AS l").single()
        n_edges, n_laund = rec["n"], rec["l"]

        sample = s.run("""
            MATCH (a:Account)-[t:TX {label: 1}]->(b:Account)
            RETURN a.id AS src, b.id AS dst, t.amt_paid AS amt, t.fmt AS fmt
            LIMIT 5
        """).data()
    driver.close()

    # Compare against the source of truth
    df = pd.read_parquet(PROCESSED / "transactions.parquet", columns=["src", "dst", "label"])
    exp_nodes = pd.unique(pd.concat([df["src"], df["dst"]])).size
    exp_edges, exp_laund = len(df), int(df["label"].sum())

    print("\n=== VERIFICATION ===")
    print(f"{'':12}{'Neo4j':>12}{'expected':>12}")
    for name, got, exp in [("accounts", n_nodes, exp_nodes),
                           ("tx", n_edges, exp_edges),
                           ("laundering", n_laund, exp_laund)]:
        flag = "OK" if got == exp else "MISMATCH"
        print(f"{name:12}{got:>12,}{exp:>12,}  {flag}")

    print("\nSample laundering edges:")
    for r in sample:
        print(f"  {r['src']} -> {r['dst']}  {r['amt']:,.2f}  ({r['fmt']})")


if __name__ == "__main__":
    main()
