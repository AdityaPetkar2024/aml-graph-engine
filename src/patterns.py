"""
Stage 3: Rule-based detectors (the baseline the ML model must beat).

Five classic AML typology rules, written in Cypher and run per split
(train / val / test) so each split only sees its own time window:

  fan_out         one account pays >= K distinct accounts in a single day
  fan_in          one account is paid by >= K distinct accounts in a single day
  gather_scatter  an account receives from >= K and pays out to >= K within D days
  scatter_gather  A -> >= K low-activity intermediaries -> C, within a time window
  cycle           money returns to its origin in 2..MAX_HOPS hops, timestamps increasing

Thresholds are fixed domain heuristics, NOT tuned on the test split.

Outputs:
  data/processed/rule_flags.parquet   one row per flagged (account, split), rule scores
  reports/rule_baseline.md            precision / recall per rule per split

Usage (from project root, Neo4j running):
    python src/patterns.py
    python src/patterns.py --splits val test          # faster while iterating
    python src/patterns.py --skip-cycles              # if cycles are too slow
    python src/patterns.py --max-hops 3               # cheaper cycle search
"""
import argparse
import time
from collections import defaultdict

import pandas as pd
from neo4j import GraphDatabase

from config import (ROOT, PROCESSED, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
                    DATA_END, VAL_START, TEST_START)

DATA_START = "2022-09-01"
SPLITS = {
    "train": (DATA_START, VAL_START),
    "val": (VAL_START, TEST_START),
    "test": (TEST_START, DATA_END),
}
DAY = 86400

# Fixed heuristics (document these in the README)
P = dict(
    fan_k=5,                  # distinct counterparties in one day
    gs_k=3,                   # gather-scatter: >= k senders and >= k receivers
    gs_days=2,                # ... within this many days
    sg_k=3,                   # scatter-gather: >= k distinct intermediaries
    sg_window=3 * DAY,        # intermediary must forward within this window
    cycle_window=5 * DAY,     # whole cycle must complete within this window
    hub_cap=30,               # skip intermediaries with more tx than this in the split
)
BATCH = 1000


def epoch(d: str) -> int:
    return int(pd.Timestamp(d).value // 10**9)   # same convention as ingest.py


# ---------------------------------------------------------------- Cypher

def q_degree(prop):
    # Per-split activity count, used only to skip high-volume hubs during traversal
    return f"""
    MATCH (a:Account)
    CALL {{
      WITH a
      SET a.{prop} = COUNT {{ (a)-[t:TX]-() WHERE t.ts >= $start AND t.ts < $end }}
    }} IN TRANSACTIONS OF 20000 ROWS
    """


Q_FAN_OUT = """
MATCH (a:Account)-[t:TX]->(b:Account)
WHERE t.ts >= $start AND t.ts < $end AND a <> b
WITH a, t.ts / 86400 AS day, count(DISTINCT b) AS n
WHERE n >= $fan_k
RETURN a.id AS account, max(n) AS score
"""

Q_FAN_IN = """
MATCH (a:Account)-[t:TX]->(b:Account)
WHERE t.ts >= $start AND t.ts < $end AND a <> b
WITH b, t.ts / 86400 AS day, count(DISTINCT a) AS n
WHERE n >= $fan_k
RETURN b.id AS account, max(n) AS score
"""

Q_GATHER_SCATTER = """
MATCH (s:Account)-[t:TX]->(h:Account)
WHERE t.ts >= $start AND t.ts < $end AND s <> h
WITH h, t.ts / 86400 AS day, count(DISTINCT s) AS n_in
WHERE n_in >= $gs_k
MATCH (h)-[t2:TX]->(r:Account)
WHERE t2.ts >= day * 86400 AND t2.ts < (day + $gs_days) * 86400
  AND t2.ts < $end AND r <> h
WITH h, day, n_in, count(DISTINCT r) AS n_out
WHERE n_out >= $gs_k
RETURN h.id AS account, max(n_in + n_out) AS score
"""


def q_sg_candidates(deg):
    return f"""
    MATCH (a:Account)-[t:TX]->(m:Account)
    WHERE t.ts >= $start AND t.ts < $end AND a <> m AND m.{deg} <= $hub_cap
    WITH a, count(DISTINCT m) AS n
    WHERE n >= $sg_k
    RETURN a.id AS id
    """


def q_scatter_gather(deg):
    return f"""
    UNWIND $ids AS id
    MATCH (a:Account {{id: id}})-[t1:TX]->(m:Account)-[t2:TX]->(c:Account)
    WHERE t1.ts >= $start AND t1.ts < $end AND m.{deg} <= $hub_cap
      AND a <> m AND m <> c AND a <> c
      AND t2.ts >= t1.ts AND t2.ts <= t1.ts + $sg_window AND t2.ts < $end
    WITH a, c, collect(DISTINCT m.id) AS mids
    WHERE size(mids) >= $sg_k
    RETURN a.id AS src, c.id AS dst, mids
    """


def q_cycle_candidates(deg):
    return f"""
    MATCH (a:Account)
    WHERE a.{deg} >= 2 AND a.{deg} <= $hub_cap
      AND EXISTS {{ (a)-[t:TX]->(b) WHERE t.ts >= $start AND t.ts < $end AND b <> a }}
      AND EXISTS {{ (b)-[t:TX]->(a) WHERE t.ts >= $start AND t.ts < $end AND b <> a }}
    RETURN a.id AS id
    """


def q_cycle(deg, max_hops):
    # Quantified path pattern: prunes during expansion (time window, no self-loops,
    # no hubs), then checks timestamps increase hop by hop. LIMIT 1 = stop at first hit.
    return f"""
    UNWIND $ids AS id
    MATCH (a:Account {{id: id}})
    CALL {{
      WITH a
      MATCH (a) ((x)-[r:TX]->(y)
                 WHERE r.ts >= $start AND r.ts < $end AND x <> y
                   AND y.{deg} <= $hub_cap){{2,{max_hops}}} (a)
      WHERE all(i IN range(0, size(r) - 2) WHERE r[i].ts <= r[i + 1].ts)
        AND r[size(r) - 1].ts - r[0].ts <= $cycle_window
      RETURN size(r) AS hops
      LIMIT 1
    }}
    RETURN a.id AS account, hops AS score
    """


# ---------------------------------------------------------------- runner

def batched(session, query, ids, params, label):
    rows, t0 = [], time.time()
    for i in range(0, len(ids), BATCH):
        rows += session.run(query, ids=ids[i:i + BATCH], **params).data()
        done = min(i + BATCH, len(ids))
        if (i // BATCH) % 20 == 0 or done == len(ids):
            print(f"      {label}: {done:,}/{len(ids):,} anchors  ({time.time() - t0:.0f}s)")
    return rows


def run_split(driver, split, skip_cycles, max_hops):
    start, end = (epoch(d) for d in SPLITS[split])
    deg = f"deg_{split}"
    params = dict(start=start, end=end, **P)
    flags = defaultdict(dict)        # account -> {rule: score}
    timings = {}

    with driver.session() as s:
        t = time.time()
        s.run(q_degree(deg), start=start, end=end).consume()
        print(f"    degree property {deg}: {time.time() - t:.0f}s")

        for rule, q in [("fan_out", Q_FAN_OUT), ("fan_in", Q_FAN_IN),
                        ("gather_scatter", Q_GATHER_SCATTER)]:
            t = time.time()
            for r in s.run(q, **params).data():
                flags[r["account"]][rule] = r["score"]
            timings[rule] = time.time() - t
            print(f"    {rule}: {timings[rule]:.0f}s")

        t = time.time()
        ids = [r["id"] for r in s.run(q_sg_candidates(deg), **params).data()]
        print(f"    scatter_gather: {len(ids):,} candidate sources")
        for r in batched(s, q_scatter_gather(deg), ids, params, "scatter_gather"):
            n = len(r["mids"])
            for acct in [r["src"], r["dst"], *r["mids"]]:
                flags[acct]["scatter_gather"] = max(flags[acct].get("scatter_gather", 0), n)
        timings["scatter_gather"] = time.time() - t

        if not skip_cycles:
            t = time.time()
            ids = [r["id"] for r in s.run(q_cycle_candidates(deg), **params).data()]
            print(f"    cycle: {len(ids):,} candidate anchors (max {max_hops} hops)")
            for r in batched(s, q_cycle(deg, max_hops), ids, params, "cycle"):
                flags[r["account"]]["cycle"] = r["score"]
            timings["cycle"] = time.time() - t

    return flags, timings


# ---------------------------------------------------------------- evaluation

def ground_truth():
    df = pd.read_parquet(PROCESSED / "transactions.parquet",
                         columns=["src", "dst", "label", "split"])
    truth = {}
    for sp in SPLITS:
        d = df[df["split"] == sp]
        lab = d[d["label"] == 1]
        truth[sp] = (set(d["src"]) | set(d["dst"]),      # active accounts
                     set(lab["src"]) | set(lab["dst"]))  # laundering accounts
    return truth


def evaluate(split, flags, timings, truth, rules):
    active, pos = truth[split]
    rows = []
    for rule in rules + ["ANY_RULE"]:
        flagged = set(flags) if rule == "ANY_RULE" else {a for a, f in flags.items() if rule in f}
        tp = len(flagged & pos)
        rows.append(dict(
            split=split, rule=rule, alerts=len(flagged), true_pos=tp,
            precision=tp / len(flagged) if flagged else 0.0,
            recall=tp / len(pos) if pos else 0.0,
            seconds=round(timings.get(rule, sum(timings.values())), 1),
        ))
    print(f"\n  {split}: {len(active):,} active accounts, {len(pos):,} laundering accounts")
    return rows


def main(splits, skip_cycles, max_hops):
    rules = ["fan_out", "fan_in", "gather_scatter", "scatter_gather"] + ([] if skip_cycles else ["cycle"])
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("Loading ground truth...")
    truth = ground_truth()

    all_flags, results = [], []
    for sp in splits:
        print(f"\n=== {sp.upper()} ===")
        flags, timings = run_split(driver, sp, skip_cycles, max_hops)
        results += evaluate(sp, flags, timings, truth, rules)
        for acct, f in flags.items():
            all_flags.append({"account": acct, "split": sp, **{r: f.get(r, 0) for r in rules}})
    driver.close()

    flags_df = pd.DataFrame(all_flags)
    flags_df.to_parquet(PROCESSED / "rule_flags.parquet", index=False)

    res = pd.DataFrame(results)
    pd.set_option("display.width", 120)
    table = res.to_string(index=False, formatters={
        "precision": "{:.3f}".format, "recall": "{:.3f}".format})
    print("\n=== RULE BASELINE ===")
    print(table)

    out = ROOT / "reports" / "rule_baseline.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# Rule-based baseline\n\nThresholds: " + str(P)
                   + "\n\nAccount-level metrics per split.\n\n```\n" + table + "\n```\n")
    print(f"\nSaved {PROCESSED / 'rule_flags.parquet'} and {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    ap.add_argument("--skip-cycles", action="store_true")
    ap.add_argument("--max-hops", type=int, default=4)
    a = ap.parse_args()
    main(a.splits, a.skip_cycles, a.max_hops)
