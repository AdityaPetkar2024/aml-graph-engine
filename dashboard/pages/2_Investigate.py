import math
from datetime import datetime, timezone

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

from common import get, risk_colour

st.set_page_config(page_title="Investigate", page_icon="🔎", layout="wide")
st.title("Investigate an account")

acct = st.text_input("Account ID", value=st.session_state.get("account", ""),
                     placeholder="e.g. 12_800063D30  (bank_account)").strip()
if not acct:
    st.info("Pick an account from the Alert Queue, or paste an account ID.")
    st.stop()
st.session_state["account"] = acct

s = get(f"/score/{acct}")
if s is None:
    st.warning(f"Unknown account: {acct}")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Risk score", f"{s['score']:.3f}")
c2.metric("Transactions in last 48h", s["tx_in_window"])
c3.metric("Graph features", s["graph_features"])
c4.metric("Known launderer (demo)", "yes" if s["demo_label"] == 1 else "not seen")
if s["graph_features"] == "missing":
    st.caption("New account: no graph features until the next batch run, so the score "
               "rests on its own transactions only.")

left, right = st.columns([2, 3], gap="large")

with left:
    st.subheader("Why this score")
    if not s["reasons"]:
        st.write("No factor pushes this account's risk up.")
    else:
        r = pd.DataFrame([{
            "Factor": x["description"],
            "Value": None if x["value"] is None else round(x["value"], 3),
            "Contribution": round(x["contribution"], 3),
        } for x in s["reasons"]])
        chart = alt.Chart(r).mark_bar(color="#d62728").encode(
            x=alt.X("Contribution:Q", title="SHAP contribution (log-odds)"),
            y=alt.Y("Factor:N", sort="-x", title=None, axis=alt.Axis(labelLimit=340)),
            tooltip=["Factor", "Value", "Contribution"],
        ).properties(height=36 * len(r) + 40)
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(r, hide_index=True)
        st.caption("Contribution = SHAP value in log-odds: how much each factor raised the "
                   "score. It shows what the model relied on, not proof of laundering.")

with right:
    st.subheader("Transaction network (2 hops)")
    g = get(f"/subgraph/{acct}")
    if not g or not g["edges"]:
        st.write("No transactions in the window for this account.")
    else:
        net = Network(height="560px", width="100%", directed=True,
                      cdn_resources="remote", bgcolor="#ffffff")
        net.barnes_hut(gravity=-4000, spring_length=120)
        for node in g["nodes"]:
            score = node["score"]
            tip = f"{node['id']}\nrisk: {'not scored' if score is None else f'{score:.3f}'}"
            net.add_node(node["id"], label=node["id"], title=tip,
                         color=risk_colour(score), size=28 if node["center"] else 14,
                         borderWidth=4 if node["center"] else 1, font={"size": 11})
        for e in g["edges"]:
            when = datetime.fromtimestamp(e["ts"], tz=timezone.utc).strftime("%b %d %H:%M")
            net.add_edge(e["src"], e["dst"], width=1 + math.log10(e["amount"] + 1) / 2,
                         title=f"{e['amount']:,.2f} via {e['fmt']} on {when}",
                         color="#9aa0a6", arrows="to")
        components.html(net.generate_html(), height=580, scrolling=False)
        st.caption(f"{len(g['nodes'])} accounts, {len(g['edges'])} transactions since Sep 7. "
                   "Red ≥ 0.5 risk, orange 0.2–0.5, green < 0.2, grey = not scored this "
                   "session. Arrows show payment direction; thickness grows with amount. "
                   "Hover for details; busy hubs are capped.")
