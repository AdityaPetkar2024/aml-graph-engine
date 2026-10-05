import os
from pathlib import Path

import requests
import streamlit as st

API = os.environ.get("AML_API", "http://localhost:8000")
ROOT = Path(__file__).resolve().parents[1]


def get(path: str, **params):
    """GET from the scoring API. Returns None on 404; stops the page if the API is down."""
    try:
        r = requests.get(f"{API}{path}", params=params, timeout=15)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        st.error(f"Scoring API not reachable at {API} ({e.__class__.__name__}). "
                 "Start it with:  uvicorn api.main:app --port 8000")
        st.stop()


def risk_colour(score):
    if score is None:
        return "#b4b4b4"      # not scored in this session
    if score >= 0.5:
        return "#d62728"      # high
    if score >= 0.2:
        return "#ff9f1c"      # medium
    return "#2ca02c"          # low
