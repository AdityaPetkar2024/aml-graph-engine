"""
AML Screening API.

Run (from project root, Neo4j running):
    uvicorn api.main:app --port 8000
Docs:
    http://localhost:8000/docs
"""
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd                                    # noqa: E402
from fastapi import FastAPI, HTTPException, Query      # noqa: E402

from config import VAL_START                           # noqa: E402
from api.db import Graph                               # noqa: E402
from api.schemas import AccountScore, Subgraph, Transaction, TxBatch, TxResult  # noqa: E402
from api.scoring import Scorer                         # noqa: E402

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    t = time.time()
    state["scorer"] = Scorer()
    state["graph"] = Graph()
    print(f"Model and feature store loaded in {time.time() - t:.1f}s; "
          f"Neo4j reachable: {state['graph'].ping()}")
    yield
    state["graph"].close()


app = FastAPI(title="AML Graph-Screening API", version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "neo4j": state["graph"].ping(), **state["scorer"].stats()}


@app.get("/stats")
def stats():
    return state["scorer"].stats()


@app.post("/transaction", response_model=TxResult)
def transaction(tx: Transaction):
    """Ingest one transaction: write it to the graph, update both accounts' live
    features, rescore them. Returns the new scores and the processing latency."""
    t0 = time.perf_counter()
    d = tx.model_dump()
    state["graph"].write_tx(d)
    scores = state["scorer"].ingest(d)
    ms = (time.perf_counter() - t0) * 1000
    state["scorer"].latencies.append(ms)
    return {"tx_id": tx.tx_id, "scores": scores, "latency_ms": round(ms, 2)}


@app.post("/transactions/batch")
def transactions_batch(b: TxBatch):
    """Micro-batch ingest: one graph write for the whole batch, then every touched
    account is rescored in a single vectorised pass."""
    t0 = time.perf_counter()
    txs = [t.model_dump() for t in b.transactions]
    state["graph"].write_batch(txs)
    t1 = time.perf_counter()
    scored = state["scorer"].ingest_many(txs, score=b.score)
    t2 = time.perf_counter()
    info = {"transactions": len(txs), "accounts_scored": scored,
            "write_ms": round((t1 - t0) * 1000), "score_ms": round((t2 - t1) * 1000),
            "total_ms": round((t2 - t0) * 1000),
            "tx_per_s": round(len(txs) / max(t2 - t0, 1e-9), 1)}
    state["scorer"].batches.append(info)
    return info


@app.post("/admin/rescore-all")
def rescore_all(at: Optional[str] = None):
    """Rescore every account in the rolling window, optionally moving the stream clock
    first (the simulator uses the end of the test window for its final evaluation)."""
    return state["scorer"].rescore_all(at)


@app.post("/admin/refresh-graph")
def refresh_graph():
    """Recompute graph and neighbourhood features from the current 48h window
    (what a scheduled batch job would do)."""
    return state["scorer"].refresh_graph()


@app.get("/scores")
def all_scores():
    """Latest score for every scored account (used for offline-vs-live evaluation)."""
    return state["scorer"].all_scores()


@app.get("/score/{account}", response_model=AccountScore)
def score(account: str):
    sc = state["scorer"]
    if account not in sc.window and account not in sc.batch.index:
        raise HTTPException(404, f"Unknown account {account}")
    with sc.lock:
        return sc.score(account, top_k=5)


@app.get("/alerts", response_model=list[AccountScore])
def alerts(n: int = Query(50, ge=1, le=1000)):
    """Highest-risk accounts scored so far in the live window."""
    return state["scorer"].alerts(n)


@app.get("/subgraph/{account}", response_model=Subgraph)
def subgraph(account: str, hop1: int = Query(40, le=200), per: int = Query(5, le=20)):
    """Two-hop neighbourhood (batch window + live window) for the investigation view."""
    since = int(pd.Timestamp(VAL_START).value // 10**9)
    edges = state["graph"].subgraph(account, since, hop1, per)
    latest = state["scorer"].latest
    ids = {account} | {e["src"] for e in edges} | {e["dst"] for e in edges}
    nodes = [{"id": i, "center": i == account,
              "score": latest[i]["score"] if i in latest else None} for i in ids]
    return {"account": account, "nodes": nodes, "edges": edges}


@app.post("/admin/reset")
def reset():
    """Clear the live window (used by the simulator before a fresh replay)."""
    state["scorer"].reset()
    return {"status": "reset"}
