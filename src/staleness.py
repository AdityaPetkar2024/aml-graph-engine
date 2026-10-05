"""
Stage 6b: How often must the graph features be refreshed?

Replays the whole test window (Sep 9-10) through the SAME serving code as the API
(api.scoring.Scorer), in-process so no HTTP or Neo4j is needed, under different
graph-refresh schedules. At the end of the window every account is rescored and
compared with the offline test labels.

  never       graph features stay those of the Sep 7-8 batch (48h+ stale)
  every 24h   refreshed at Sep 10 00:00 (24h stale at the end)
  every 12h   ... last refresh Sep 10 12:00
  every 6h    ... last refresh Sep 10 18:00
  at end      refreshed at Sep 11 00:00 (fresh; should equal offline)

Usage (from project root):
    python src/staleness.py
    python src/staleness.py --schedules never 24 end      # fewer runs
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                 # for the api package

import numpy as np                            # noqa: E402
import pandas as pd                           # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from api.scoring import Scorer               # noqa: E402
from config import PROCESSED, TEST_START, DATA_END  # noqa: E402

CHUNK = 50_000


def load_test_tx():
    cols = ["tx_id", "ts", "src", "dst", "amt_paid", "cur_paid", "amt_recv",
            "cur_recv", "fmt", "label", "split"]
    df = pd.read_parquet(PROCESSED / "transactions.parquet", columns=cols)
    df = df[df["split"] == "test"].sort_values("tx_id").drop(columns="split")
    for c in ["cur_paid", "cur_recv", "fmt"]:
        df[c] = df[c].astype(str)
    return df


def boundaries(schedule):
    start, end = pd.Timestamp(TEST_START), pd.Timestamp(DATA_END)
    if schedule == "never":
        return []
    if schedule == "end":
        return [end]
    step = pd.Timedelta(hours=int(schedule))
    out, b = [], start + step
    while b < end:                            # refreshes strictly before the end
        out.append(b)
        b += step
    return out


def replay(sc, df, schedule):
    sc.reset()
    ts = df["ts"].to_numpy()
    records = df.to_dict("records")
    pos, refresh_s = 0, 0.0
    for b in boundaries(schedule) + [None]:
        stop = len(records) if b is None else int(np.searchsorted(ts, np.datetime64(b), "left"))
        for i in range(pos, stop, CHUNK):
            sc.ingest_many(records[i:min(i + CHUNK, stop)], score=False)
        pos = stop
        if b is not None:
            sc.now = b                       # the job runs on the clock boundary
            refresh_s += sc.refresh_graph()["seconds"]
    sc.rescore_all(at=DATA_END)
    s = sc.all_scores()
    return pd.DataFrame({"account": s["accounts"], "score": s["scores"]}), refresh_s


def metrics(y, s, ks):
    order = np.argsort(-s, kind="stable")
    row = {"PR-AUC": average_precision_score(y, s), "ROC-AUC": roc_auc_score(y, s)}
    for k in ks:
        row[f"recall@{k}"] = y[order[:k]].sum() / y.sum()
    return row


def main(schedules):
    t0 = time.time()
    print("Loading model, warm-start window and test transactions...")
    sc = Scorer()
    df = load_test_tx()
    feats = pd.read_parquet(PROCESSED / "features_test.parquet",
                            columns=["account", "label", "rule_any"])
    off = pd.read_parquet(PROCESSED / "test_scores.parquet", columns=["account", "score"])
    y = feats["label"].to_numpy()
    k_rules = int((feats["rule_any"] > 0).sum())
    ks = [100, 1000, k_rules]

    names = {"never": "never (Sep 7-8 batch only)", "end": "at end of window (fresh)"}
    rows = {"rules": metrics(y, feats["rule_any"].to_numpy(dtype=float), ks),
            "offline": metrics(y, feats[["account"]].merge(off, on="account", how="left")
                               ["score"].fillna(0).to_numpy(), ks)}
    for sch in schedules:
        t = time.time()
        live, refresh_s = replay(sc, df, sch)
        s = feats[["account"]].merge(live, on="account", how="left")["score"].fillna(0)
        name = names.get(sch, f"every {sch}h")
        rows[f"live, refresh {name}"] = metrics(y, s.to_numpy(), ks)
        print(f"  {name:28s} PR-AUC {rows[f'live, refresh {name}']['PR-AUC']:.3f}  "
              f"({len(boundaries(sch))} refreshes, {refresh_s:.0f}s refreshing, "
              f"{time.time() - t:.0f}s total)")

    table = pd.DataFrame(rows).T
    text = table.to_string(float_format="{:.3f}".format)
    print(f"\n=== GRAPH REFRESH SCHEDULE vs DETECTION ({len(y):,} test accounts, "
          f"{int(y.sum()):,} launderers) ===\n{text}")
    out = ROOT / "reports" / "graph_staleness.md"
    out.write_text("# Graph-feature refresh schedule vs detection (test window)\n\n```\n"
                   + text + "\n```\n")
    print(f"\nSaved {out}  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedules", nargs="+", default=["never", "24", "12", "6", "end"],
                    help="never | end | hours between refreshes (e.g. 24 12 6)")
    main(ap.parse_args().schedules)
