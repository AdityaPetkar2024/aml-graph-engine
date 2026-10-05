from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Transaction(BaseModel):
    tx_id: int
    ts: datetime
    src: str = Field(description="bank_account of the payer, e.g. 12_800063D30")
    dst: str = Field(description="bank_account of the payee")
    amt_paid: float = Field(ge=0)
    cur_paid: str
    amt_recv: float = Field(ge=0)
    cur_recv: str
    fmt: str = Field(description="Payment format: ACH, Wire, Cheque, ...")
    label: Optional[int] = Field(
        default=None, description="Ground truth, demo only. NEVER used for scoring.")


class Reason(BaseModel):
    feature: str
    description: str
    value: Optional[float]
    contribution: float          # SHAP value (log-odds); positive = raises risk


class AccountScore(BaseModel):
    account: str
    score: float                 # model probability, 0..1
    reasons: list[Reason]        # top factors pushing the score up
    tx_in_window: int            # live transactions seen for this account
    graph_features: str          # "batch" = from last batch run, "missing" = new account
    demo_label: Optional[int]    # 1 if a known laundering tx was seen (demo only)
    scored_at: datetime


class TxResult(BaseModel):
    tx_id: int
    scores: list[AccountScore]
    latency_ms: float


class Edge(BaseModel):
    src: str
    dst: str
    amount: float
    ts: int
    fmt: str


class Subgraph(BaseModel):
    account: str
    nodes: list[dict]            # {"id", "score" (or None), "center"}
    edges: list[Edge]
