"""
Scoring engine: speed layer + batch layer.

  speed layer  the account's own (tabular) features are recomputed on every
               transaction over a ROLLING 48h window, using the SAME
               tabular_features() function as offline training (no train/serve skew).
               The window is warm-started with the 48h before the stream begins,
               so every account has the same amount of history the model was
               trained on (otherwise an account's first ACH payment = "100% ACH").
  batch layer  graph + neighbourhood features come from the last completed batch
               run (features_val.parquet = the Sep 7-8 window), like a nightly job

In production the in-memory window would live in a feature store such as Redis.
"""
import json
import threading
from collections import defaultdict, deque
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb

from config import MODELS, PROCESSED
from features import tabular_features, usd_rates

WINDOW_DAYS = 2                      # same length as the val/test windows
WINDOW = pd.Timedelta(days=WINDOW_DAYS)
WARM_SPLIT = "val"                   # the 48h immediately before the stream (Sep 7-8)
COLS = ["s", "t", "ts", "usd_paid", "usd_recv", "fmt", "cross_currency", "cross_bank"]
BATCH_FILE = "features_val.parquet"  # most recent completed window before the stream
GRAPH_COLS = {"pagerank", "wcc_size", "n_neighbours", "log_scc_size", "in_cycle_scc"}

DESCRIPTIONS = {
    "fmt_ach": "share of transactions paid by ACH",
    "fmt_cheque": "share of transactions paid by cheque",
    "fmt_credit_card": "share of transactions paid by credit card",
    "fmt_cash": "share of transactions paid in cash",
    "fmt_wire": "share of transactions paid by wire",
    "fmt_bitcoin": "share of transactions paid in bitcoin",
    "fmt_reinvestment": "share of reinvestment (self) transactions",
    "out_cnt": "outgoing payments per day",
    "in_cnt": "incoming payments per day",
    "out_uniq": "distinct payees per day",
    "in_uniq": "distinct payers per day",
    "out_sum": "total sent per day (log USD)",
    "in_sum": "total received per day (log USD)",
    "out_mean": "average payment sent (log USD)",
    "in_mean": "average payment received (log USD)",
    "out_max": "largest payment sent (log USD)",
    "in_max": "largest payment received (log USD)",
    "log_flow_ratio": "money out vs money in (log ratio)",
    "uniq_ratio": "share of counterparties that are payees",
    "pt_frac24": "share of payments forwarded within 24h of receiving money",
    "pt_median_h": "median hours between receiving and forwarding money",
    "max_tx_per_hour": "burstiness: most transactions in one hour",
    "active_day_frac": "fraction of days active",
    "cross_currency_share": "share of cross-currency transactions",
    "cross_bank_share": "share of cross-bank transactions",
    "self_share": "share of self-transfers",
    "pagerank": "centrality in the transaction network",
    "n_neighbours": "number of distinct counterparties in the graph",
    "log_scc_size": "size of the cycle-connected group it belongs to",
    "in_cycle_scc": "sits on a money cycle",
}


def describe(feature: str) -> str:
    if feature.startswith("nbr_mean_"):
        base = feature[len("nbr_mean_"):]
        return "neighbours' average: " + DESCRIPTIONS.get(base, base)
    return DESCRIPTIONS.get(feature, feature)


def load_rates() -> dict:
    path = MODELS / "usd_rates.json"
    if path.exists():
        return json.loads(path.read_text())
    cols = ["split", "amt_paid", "amt_recv", "cur_paid", "cur_recv"]
    df = pd.read_parquet(PROCESSED / "transactions.parquet", columns=cols)
    for c in ["split", "cur_paid", "cur_recv"]:
        df[c] = df[c].astype(str)
    rates = {k: float(v) for k, v in usd_rates(df).items()}
    path.write_text(json.dumps(rates, indent=2))
    return rates


class Scorer:
    def __init__(self):
        self.booster = xgb.Booster()
        self.booster.load_model(MODELS / "xgb.json")
        try:
            self.n_trees = self.booster.best_iteration + 1
        except AttributeError:
            self.n_trees = 0                       # 0 = use all trees
        self.feats = json.loads((MODELS / "feature_columns.json").read_text())
        self.batch_cols = [c for c in self.feats if c in GRAPH_COLS or c.startswith("nbr_")]
        self.live_cols = [c for c in self.feats if c not in self.batch_cols]

        batch = pd.read_parquet(PROCESSED / BATCH_FILE).set_index("account")
        self.batch = batch[self.batch_cols]
        self.rates = load_rates()
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        with self.lock:
            self.window = defaultdict(list)        # account -> tx tuples (rolling 48h)
            self.latest = {}                       # account -> AccountScore dict
            self.demo_label = {}
            self.latencies = deque(maxlen=2000)
            self.n_tx = 0
            self.now = None                        # stream clock = latest tx timestamp
            self._warm_start()

    def _warm_start(self):
        """Pre-load the 48h before the stream so features start mature."""
        cols = ["ts", "src", "dst", "amt_paid", "cur_paid", "amt_recv", "cur_recv",
                "fmt", "cross_currency", "cross_bank", "split"]
        df = pd.read_parquet(PROCESSED / "transactions.parquet", columns=cols)
        df = df[df["split"] == WARM_SPLIT]
        usd_p = df["amt_paid"].to_numpy() * df["cur_paid"].astype(str).map(self.rates).to_numpy(dtype=float)
        usd_r = df["amt_recv"].to_numpy() * df["cur_recv"].astype(str).map(self.rates).to_numpy(dtype=float)
        rows = zip(df["src"], df["dst"], df["ts"], usd_p, usd_r, df["fmt"].astype(str),
                   df["cross_currency"].astype(int), df["cross_bank"].astype(int))
        for r in rows:
            self.window[r[0]].append(r)
            if r[1] != r[0]:
                self.window[r[1]].append(r)
        self.now = df["ts"].max()
        self.warm_tx = len(df)

    # ------------------------------------------------------------ features
    def _row(self, tx: dict) -> dict:
        usd_p = tx["amt_paid"] * self.rates.get(tx["cur_paid"], np.nan)
        usd_r = tx["amt_recv"] * self.rates.get(tx["cur_recv"], np.nan)
        ts = pd.Timestamp(tx["ts"])
        ts = ts.tz_convert(None) if ts.tzinfo else ts
        return (tx["src"], tx["dst"], ts, usd_p, usd_r, tx["fmt"],
                int(tx["cur_paid"] != tx["cur_recv"]),
                int(tx["src"].split("_")[0] != tx["dst"].split("_")[0]))

    def feature_vector(self, account: str):
        rows = self.window.get(account, [])
        if rows and self.now is not None:                 # drop tx older than 48h
            cutoff = self.now - WINDOW
            if rows[0][2] <= cutoff:
                rows = [r for r in rows if r[2] > cutoff]
                self.window[account] = rows
        live = pd.Series(np.nan, index=self.live_cols, dtype=float)
        if rows:
            F = tabular_features(pd.DataFrame(rows, columns=COLS), WINDOW_DAYS)
            if account in F.index:
                r = F.loc[account]
                for c in self.live_cols:
                    if c in r.index:
                        live[c] = r[c]
                    elif c.startswith("fmt_"):
                        live[c] = 0.0              # format never used by this account
        if account in self.batch.index:
            graph, source = self.batch.loc[account], "batch"
        else:
            graph, source = pd.Series(np.nan, index=self.batch_cols), "missing"
        x = pd.concat([live, graph])[self.feats].astype(float)
        return x, source, len(rows)

    # ------------------------------------------------------------ scoring
    def score(self, account: str, top_k: int = 3) -> dict:
        x, source, n = self.feature_vector(account)
        dm = xgb.DMatrix(x.to_frame().T, feature_names=self.feats)
        rng = (0, self.n_trees)
        p = float(self.booster.predict(dm, iteration_range=rng)[0])
        contrib = self.booster.predict(dm, pred_contribs=True, iteration_range=rng)[0][:-1]
        order = np.argsort(-contrib)[:top_k]
        reasons = [{
            "feature": self.feats[i], "description": describe(self.feats[i]),
            "value": None if pd.isna(x.iloc[i]) else float(x.iloc[i]),
            "contribution": float(contrib[i]),
        } for i in order if contrib[i] > 0]
        return {"account": account, "score": p, "reasons": reasons, "tx_in_window": n,
                "graph_features": source, "demo_label": self.demo_label.get(account),
                "scored_at": datetime.now(timezone.utc)}

    def ingest(self, tx: dict) -> list[dict]:
        with self.lock:
            row = self._row(tx)
            self.now = row[2] if self.now is None else max(self.now, row[2])
            self.window[tx["src"]].append(row)
            if tx["dst"] != tx["src"]:
                self.window[tx["dst"]].append(row)
            if tx.get("label") == 1:
                self.demo_label[tx["src"]] = self.demo_label[tx["dst"]] = 1
            self.n_tx += 1
            out = []
            for acct in dict.fromkeys([tx["src"], tx["dst"]]):
                s = self.score(acct)
                self.latest[acct] = s
                out.append(s)
            return out

    def alerts(self, n: int) -> list[dict]:
        with self.lock:
            return sorted(self.latest.values(), key=lambda s: -s["score"])[:n]

    def stats(self) -> dict:
        lat = np.array(self.latencies)
        pct = (lambda q: round(float(np.percentile(lat, q)), 2)) if len(lat) else (lambda q: None)
        return {"transactions": self.n_tx, "accounts_scored": len(self.latest),
                "latency_ms_p50": pct(50), "latency_ms_p95": pct(95),
                "model_trees": self.n_trees, "features": len(self.feats),
                "batch_features_from": BATCH_FILE,
                "warm_start_tx": getattr(self, "warm_tx", 0),
                "stream_clock": str(self.now)}
