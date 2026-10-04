"""
Stage 1: Ingest
- Loads raw transactions with memory-efficient dtypes
- Parses the patterns file and tags laundering transactions with their pattern type
- Builds account IDs (bank_account) and simple flags
- Drops the post-Sep-10 tail, date-based train/val/test split
- Writes parquet (for Python) and CSVs (for Neo4j bulk import)

Usage (from project root):
    python src/ingest.py                    # HI-Small (default)
    python src/ingest.py --dataset HI-Medium
"""
import argparse
import time

import pandas as pd

from config import RAW, PROCESSED, DATA_END, VAL_START, TEST_START

COLS = ["ts", "from_bank", "from_acct", "to_bank", "to_acct", "amt_recv",
        "cur_recv", "amt_paid", "cur_paid", "fmt", "label"]
KEY = ["ts", "from_bank", "from_acct", "to_bank", "to_acct", "amt_key"]
TS_FORMAT = "%Y/%m/%d %H:%M"


def add_key(df: pd.DataFrame) -> pd.DataFrame:
    """Amount as integer cents so float noise can't break the merge."""
    df["amt_key"] = (df["amt_paid"].astype(float) * 100).round().astype("int64")
    return df


def load_transactions(dataset: str) -> pd.DataFrame:
    df = pd.read_csv(
        RAW / f"{dataset}_Trans.csv", header=0, names=COLS,
        dtype={"from_bank": "int32", "to_bank": "int32",
               "from_acct": "string", "to_acct": "string",
               "amt_recv": "float64", "amt_paid": "float64",
               "cur_recv": "category", "cur_paid": "category",
               "fmt": "category", "label": "int8"},
    )
    df["ts"] = pd.to_datetime(df["ts"], format=TS_FORMAT)
    return add_key(df)


def load_patterns(dataset: str) -> pd.DataFrame:
    rows, pattern, attempt = [], None, -1
    with open(RAW / f"{dataset}_Patterns.txt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("BEGIN"):
                # "BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 5 hops" -> "CYCLE"
                pattern = line.split(" - ", 1)[1].split(":")[0].strip()
                attempt += 1
            elif line.startswith("END"):
                pattern = None
            elif pattern:
                rows.append(line.split(",")[:11] + [pattern, attempt])

    pat = pd.DataFrame(rows, columns=COLS + ["pattern", "attempt_id"])
    pat["ts"] = pd.to_datetime(pat["ts"], format=TS_FORMAT)
    pat["from_bank"] = pat["from_bank"].astype("int32")   # strips leading zeros
    pat["to_bank"] = pat["to_bank"].astype("int32")
    pat["from_acct"] = pat["from_acct"].astype("string")
    pat["to_acct"] = pat["to_acct"].astype("string")
    pat["amt_paid"] = pat["amt_paid"].astype(float)
    pat = add_key(pat)
    return pat[KEY + ["pattern", "attempt_id"]].drop_duplicates(subset=KEY)


def write_neo4j_csvs(df: pd.DataFrame) -> int:
    accts = pd.unique(pd.concat([df["src"], df["dst"]]))
    pd.DataFrame({"id:ID": accts, ":LABEL": "Account"}).to_csv(
        PROCESSED / "nodes.csv", index=False)

    pd.DataFrame({
        ":START_ID": df["src"],
        ":END_ID": df["dst"],
        "tx_id:long": df["tx_id"],
        "ts:long": df["ts"].astype("int64") // 10**9,   # unix seconds
        "amt_paid:double": df["amt_paid"],
        "fmt": df["fmt"].astype(str),
        "label:int": df["label"],
        ":TYPE": "TX",
    }).to_csv(PROCESSED / "edges.csv", index=False)
    return len(accts)


def main(dataset: str) -> None:
    t0 = time.time()
    PROCESSED.mkdir(parents=True, exist_ok=True)

    print(f"Loading {dataset} transactions...")
    df = load_transactions(dataset)
    print(f"  {len(df):,} rows")

    print("Parsing patterns...")
    pat = load_patterns(dataset)
    print(f"  {len(pat):,} pattern transactions")

    n_before = len(df)
    df = df.merge(pat, on=KEY, how="left")
    assert len(df) == n_before, "Merge duplicated rows, check the keys"

    df["pattern"] = df["pattern"].fillna("NONE")
    df.loc[(df["label"] == 1) & (df["pattern"] == "NONE"), "pattern"] = "OTHER"
    df["pattern"] = df["pattern"].astype("category")
    df["attempt_id"] = df["attempt_id"].fillna(-1).astype("int32")

    # Account IDs and flags
    df["src"] = df["from_bank"].astype(str) + "_" + df["from_acct"]
    df["dst"] = df["to_bank"].astype(str) + "_" + df["to_acct"]
    df["cross_currency"] = (df["cur_recv"].astype(str)
                            != df["cur_paid"].astype(str)).astype("int8")
    df["cross_bank"] = (df["from_bank"] != df["to_bank"]).astype("int8")

    # Drop the near-empty tail after Sep 10 (data-generation artifact)
    n_all = len(df)
    df = df[df["ts"] < DATA_END]
    n_dropped = n_all - len(df)

    # Time-ordered IDs and date-based split (never random: that leaks the future)
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    df["tx_id"] = df.index.astype("int64")
    df["split"] = "train"
    df.loc[df["ts"] >= VAL_START, "split"] = "val"
    df.loc[df["ts"] >= TEST_START, "split"] = "test"
    df["split"] = pd.Categorical(df["split"], categories=["train", "val", "test"])

    df = df.drop(columns=["amt_key"])
    df.to_parquet(PROCESSED / "transactions.parquet", index=False)
    n_accts = write_neo4j_csvs(df)

    # Sanity report
    print("\n=== REPORT ===")
    print(f"Date range:      {df['ts'].min()}  ->  {df['ts'].max()}")
    print(f"Dropped tail:    {n_dropped:,} tx after {DATA_END}")
    print(f"Split dates:     val >= {VAL_START}, test >= {TEST_START}, end < {DATA_END}")
    print(f"Unique accounts: {n_accts:,}")
    print(f"Laundering rate: {df['label'].mean():.4%}")
    print("\nBy split (count / laundering / rate):")
    print(df.groupby("split", observed=True)["label"].agg(["count", "sum", "mean"]))
    print("\nPattern counts (laundering only):")
    print(df.loc[df["label"] == 1, "pattern"].value_counts())
    matched = (df.loc[df["label"] == 1, "pattern"] != "OTHER").mean()
    print(f"\nLaundering tx matched to a pattern: {matched:.1%}")
    print("\nTransactions per day:")
    print(df.set_index("ts").resample("D")["label"].agg(["count", "sum"]))
    print(f"\nDone in {time.time() - t0:.0f}s. Files in {PROCESSED}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="HI-Small",
                        help="File prefix, e.g. HI-Small, LI-Small, HI-Medium")
    main(parser.parse_args().dataset)
