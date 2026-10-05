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
import heapq
import json
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb

from config import MODELS, PROCESSED
from features import graph_features, neighbour_features, tabular_features, usd_rates

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
        self.initial_batch = batch[self.batch_cols]
        # PageRank is computed over the full account universe offline, so the live
        # refresh uses the same node count to give identical values.
        ids = pd.read_parquet(PROCESSED / "transactions.parquet", columns=["src", "dst"])
        self.n_accounts = int(pd.unique(pd.concat([ids["src"], ids["dst"]])).size)
        del ids
        self.rates = load_rates()
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        with self.lock:
            self.window = defaultdict(list)        # account -> tx tuples (rolling 48h)
            self.latest = {}                       # account -> AccountScore dict
            self.demo_label = {}
            self.latencies = deque(maxlen=2000)
            self.batches = deque(maxlen=200)       # micro-batch timings
            self.n_tx = 0
            self.now = None                        # stream clock = latest tx timestamp
            self.batch = self.initial_batch
            self.batch_info = f"{BATCH_FILE} (Sep 7-8 window)"
            self.refreshes = []
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
            if rows[0][2] < cutoff:
                rows = [r for r in rows if r[2] >= cutoff]
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
            # Calendar days in a rolling 48h window can be 3 (e.g. Sep 7, 8, 9), giving
            # values up to 1.5 that training never saw. Count 24h blocks back from the
            # stream clock instead, so the value stays in [0, 1] as in training.
            if "active_day_frac" in live.index and self.now is not None:
                blocks = {int((self.now - r[2]).total_seconds() // 86400) for r in rows}
                live["active_day_frac"] = min(len(blocks), WINDOW_DAYS) / WINDOW_DAYS
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

    # ------------------------------------------------------------ micro-batch
    def _rows_for(self, accounts):
        """Window rows touching any of `accounts`, pruned to the rolling window and
        de-duplicated (each tx tuple sits in both its payer's and payee's list)."""
        cutoff = self.now - WINDOW
        seen, counts = {}, {}
        for a in accounts:
            rows = self.window.get(a)
            if not rows:
                counts[a] = 0
                continue
            if rows[0][2] < cutoff:
                rows = [r for r in rows if r[2] >= cutoff]
                self.window[a] = rows
            counts[a] = len(rows)
            for r in rows:
                seen[id(r)] = r
        return list(seen.values()), counts

    def features_many(self, accounts):
        """Vectorised version of feature_vector() for many accounts at once."""
        rows, counts = self._rows_for(accounts)
        idx = pd.Index([a for a in accounts if counts.get(a, 0) > 0])
        live = pd.DataFrame(np.nan, index=idx, columns=self.live_cols, dtype=float)
        if len(idx):
            d = pd.DataFrame(rows, columns=COLS)
            F = tabular_features(d, WINDOW_DAYS).reindex(idx)
            for c in self.live_cols:
                if c in F.columns:
                    live[c] = F[c].astype(float).to_numpy()
                elif c.startswith("fmt_"):
                    live[c] = 0.0
            fmt = [c for c in self.live_cols if c.startswith("fmt_")]
            live[fmt] = live[fmt].fillna(0.0)
            if "active_day_frac" in live.columns:      # same 24h-block rule as feature_vector
                both = pd.concat([d[["s", "ts"]].rename(columns={"s": "acct"}),
                                  d[["t", "ts"]].rename(columns={"t": "acct"})])
                both = both[both["acct"].isin(idx)]
                blk = ((self.now - both["ts"]).dt.total_seconds() // 86400).astype(int)
                adf = blk.groupby(both["acct"]).nunique().clip(upper=WINDOW_DAYS) / WINDOW_DAYS
                live["active_day_frac"] = adf.reindex(idx).to_numpy()
        graph = self.batch.reindex(idx)
        X = pd.concat([live, graph], axis=1)[self.feats].astype(float)
        n = np.array([counts[a] for a in idx])
        return X, n

    def _score_many(self, accounts, reasons_for=1000):
        X, n = self.features_many(accounts)
        if X.empty:
            return 0
        rng = (0, self.n_trees)
        p = self.booster.predict(xgb.DMatrix(X.to_numpy(), feature_names=self.feats),
                                 iteration_range=rng)
        in_batch = X.index.isin(self.batch.index)
        now = datetime.now(timezone.utc)
        for i, acct in enumerate(X.index):
            self.latest[acct] = {"account": acct, "score": float(p[i]), "reasons": None,
                                 "tx_in_window": int(n[i]),
                                 "graph_features": "batch" if in_batch[i] else "missing",
                                 "demo_label": self.demo_label.get(acct), "scored_at": now}
        top = np.argsort(-p)[:reasons_for]           # SHAP only where analysts will look
        sub = X.iloc[top]
        contrib = self.booster.predict(xgb.DMatrix(sub.to_numpy(), feature_names=self.feats),
                                       pred_contribs=True, iteration_range=rng)[:, :-1]
        for j, i in enumerate(top):
            c, x = contrib[j], sub.iloc[j]
            self.latest[X.index[i]]["reasons"] = [{
                "feature": self.feats[k], "description": describe(self.feats[k]),
                "value": None if pd.isna(x.iloc[k]) else float(x.iloc[k]),
                "contribution": float(c[k]),
            } for k in np.argsort(-c)[:3] if c[k] > 0]
        return len(X)

    def refresh_graph(self) -> dict:
        """Batch layer refresh: recompute graph + neighbourhood features from the current
        rolling window with the same functions features.py uses offline."""
        with self.lock:
            t0 = time.perf_counter()
            rows, _ = self._rows_for(list(self.window))
            d = pd.DataFrame(rows, columns=COLS)
            codes, accts = pd.factorize(pd.concat([d["s"], d["t"]], ignore_index=True))
            n = len(d)
            d["s"], d["t"] = codes[:n].astype("int32"), codes[n:].astype("int32")
            F = tabular_features(d, WINDOW_DAYS)
            G, U = graph_features(d, max(self.n_accounts, len(accts)))
            F = F.join(G, how="left")
            F["rule_any"] = 0.0                    # not used by the model (see train.py)
            F = F.join(neighbour_features(F, U), how="left")
            B = F[self.batch_cols].copy()
            B.index = pd.Index(np.asarray(accts)[B.index.to_numpy()], name="account")
            self.batch = B
            self.batch_info = f"refreshed at {self.now} ({len(B):,} accounts)"
            info = {"accounts": len(B), "window_tx": n, "stream_clock": str(self.now),
                    "seconds": round(time.perf_counter() - t0, 1)}
            self.refreshes.append(info)
            return info

    def ingest_many(self, txs: list[dict], score: bool = True) -> int:
        with self.lock:
            touched = {}
            for tx in txs:
                row = self._row(tx)
                self.now = row[2] if self.now is None else max(self.now, row[2])
                self.window[tx["src"]].append(row)
                if tx["dst"] != tx["src"]:
                    self.window[tx["dst"]].append(row)
                if tx.get("label") == 1:
                    self.demo_label[tx["src"]] = self.demo_label[tx["dst"]] = 1
                touched[tx["src"]] = touched[tx["dst"]] = None
            self.n_tx += len(txs)
            return self._score_many(list(touched)) if score else 0

    def rescore_all(self, at=None) -> dict:
        """Set the stream clock (e.g. to the end of the test window) and rescore every
        account active in the rolling window: a clean end-of-window snapshot."""
        with self.lock:
            t0 = time.perf_counter()
            if at is not None:
                at = pd.Timestamp(at)
                self.now = at.tz_convert(None) if at.tzinfo else at
            self.latest = {}
            n = self._score_many(list(self.window), reasons_for=1000)
            return {"accounts": n, "seconds": round(time.perf_counter() - t0, 1),
                    "stream_clock": str(self.now)}

    def all_scores(self) -> dict:
        with self.lock:
            return {"stream_clock": str(self.now),
                    "accounts": list(self.latest),
                    "scores": [v["score"] for v in self.latest.values()]}

    def alerts(self, n: int) -> list[dict]:
        with self.lock:
            top = heapq.nlargest(n, self.latest.values(), key=lambda s: s["score"])
            for i, s in enumerate(top):
                if s["reasons"] is None:               # filled lazily for batch-scored accounts
                    top[i] = self.latest[s["account"]] = self.score(s["account"])
            return top

    def stats(self) -> dict:
        lat = np.array(self.latencies)
        pct = (lambda q: round(float(np.percentile(lat, q)), 2)) if len(lat) else (lambda q: None)
        return {"transactions": self.n_tx, "accounts_scored": len(self.latest),
                "latency_ms_p50": pct(50), "latency_ms_p95": pct(95),
                "model_trees": self.n_trees, "features": len(self.feats),
                "batch_features_from": self.batch_info,
                "graph_refreshes": len(self.refreshes),
                "warm_start_tx": getattr(self, "warm_tx", 0),
                "batches": len(self.batches),
                "last_batch": self.batches[-1] if self.batches else None,
                "stream_clock": str(self.now)}
