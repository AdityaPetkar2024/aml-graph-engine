import streamlit as st

from common import ROOT

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")
st.title("Model performance")
st.write("Offline results on the held-out test window (Sep 9 to 10), produced by "
         "`src/train.py` and `src/patterns.py`.")

reports = ROOT / "reports"
figures = reports / "figures"

results = reports / "results.md"
if results.exists():
    text = results.read_text()
    st.markdown("\n".join(l for l in text.splitlines()
                          if not l.startswith("![") and not l.startswith("# ")))
else:
    st.warning("reports/results.md not found: run  python src/train.py")

c1, c2 = st.columns(2)
for col, name, caption in [
    (c1, "pr_curve.png", "Precision-recall curves: rules vs XGBoost variants"),
    (c2, "feature_importance.png", "Top 20 features by mean |SHAP| (production model)"),
]:
    path = figures / name
    if path.exists():
        col.image(str(path), caption=caption)
    else:
        col.info(f"{name} not found: run  python src/train.py")

rules = reports / "rule_baseline.md"
if rules.exists():
    with st.expander("Rule baseline details (all splits)"):
        st.markdown("\n".join(l for l in rules.read_text().splitlines()
                              if not l.startswith("# ")))
