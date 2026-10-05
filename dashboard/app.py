"""
AML Graph-Screening Engine: analyst dashboard.

Run (from project root, API running):
    streamlit run dashboard/app.py
"""
import streamlit as st

from common import API, get

st.set_page_config(page_title="AML Graph-Screening Engine", page_icon="🕸️", layout="wide")

st.title("AML Graph-Screening Engine")
st.write(
    "Ranks bank accounts by money-laundering risk using their own transaction behaviour "
    "and the behaviour of the accounts they are connected to. Every score comes with "
    "the factors behind it."
)

h = get("/health")
c1, c2, c3, c4 = st.columns(4)
c1.metric("API", h["status"].upper())
c2.metric("Neo4j", "connected" if h["neo4j"] else "DOWN")
c3.metric("Transactions streamed", f"{h['transactions']:,}")
c4.metric("Accounts scored", f"{h['accounts_scored']:,}")
if not h["neo4j"]:
    st.warning("Neo4j is not reachable: run `docker compose up -d`. Scoring still works; "
               "the investigation graph will not.")

st.subheader("Pages")
st.markdown(
    "- **Alert Queue**: the highest-risk accounts right now, with the top reason for each.\n"
    "- **Investigate**: one account's risk factors and its two-hop transaction network.\n"
    "- **Model Performance**: offline results against the rule baseline."
)

st.subheader("How scoring works")
st.markdown(
    "- **Speed layer:** each account's own features (volumes, amounts, payment mix, timing) "
    "are recomputed on every transaction over a rolling 48-hour window.\n"
    "- **Batch layer:** graph features (PageRank, neighbours, cycles, neighbourhood averages) "
    "come from the latest batch run.\n"
    "- **Model:** XGBoost, explained with TreeSHAP. Scores are a reason to investigate, "
    "not proof of laundering."
)
st.caption(f"API: {API} · model: {h['model_trees']} trees, {h['features']} features · "
           f"stream clock: {h.get('stream_clock', 'n/a')}")
