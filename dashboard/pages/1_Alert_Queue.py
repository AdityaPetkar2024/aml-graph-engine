import pandas as pd
import streamlit as st

from common import get

st.set_page_config(page_title="Alert Queue", page_icon="🚨", layout="wide")
st.title("Alert queue")

n = st.sidebar.slider("Alerts to show", 10, 500, 50, step=10)
auto = st.sidebar.toggle("Auto-refresh every 5 s", value=True,
                         help="Turn off before selecting an account, so the list stops moving.")


@st.fragment(run_every=5 if auto else None)
def queue():
    s = get("/stats")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions streamed", f"{s['transactions']:,}")
    c2.metric("Accounts scored", f"{s['accounts_scored']:,}")
    c3.metric("Latency p50", f"{s['latency_ms_p50']} ms" if s["latency_ms_p50"] else "–")
    c4.metric("Latency p95", f"{s['latency_ms_p95']} ms" if s["latency_ms_p95"] else "–")

    alerts = get("/alerts", n=n)
    if not alerts:
        st.info("No alerts yet. Start the stream:  python src/simulator.py --reset")
        return

    df = pd.DataFrame([{
        "Risk": a["score"],
        "Account": a["account"],
        "Top reason": a["reasons"][0]["description"] if a["reasons"] else "",
        "Live tx (48h)": a["tx_in_window"],
        "Graph features": a["graph_features"],
        "Known launderer (demo)": "yes" if a["demo_label"] == 1 else "",
    } for a in alerts])

    event = st.dataframe(
        df, hide_index=True, on_select="rerun", selection_mode="single-row",
        key="alerts_table",
        column_config={"Risk": st.column_config.ProgressColumn(
            "Risk", min_value=0.0, max_value=1.0, format="%.3f")},
    )
    st.caption("'Known launderer (demo)' marks accounts already seen in a labelled laundering "
               "transaction during this stream. The model never sees labels.")

    rows = event.selection.rows
    if rows:
        acct = df.iloc[rows[0]]["Account"]
        if st.button(f"Investigate {acct}", type="primary"):
            st.session_state["account"] = acct
            st.switch_page("pages/2_Investigate.py")


queue()
