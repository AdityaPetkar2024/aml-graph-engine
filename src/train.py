"""
Stage 5: Train and evaluate.

Experiments (all evaluated on the TEST split, never used for any decision):
  rules_only      Stage 3 rule count as a score (the bank's baseline)
  xgb_tabular     XGBoost on the account's own transaction features only
  xgb_graph       + graph and neighbourhood features
  xgb_all         + rule scores as features
The production model is whichever variant scores best on VALIDATION PR-AUC.

Training uses train; early stopping uses val; test is touched once at the end.

Metrics
  PR-AUC          main metric: robust to the 0.4% positive rate (accuracy is useless here)
  recall@K        of the top-K riskiest accounts, how many launderers are caught
  K = rule alert count on test, so ML and rules are compared at the same analyst workload

Outputs
  models/xgb.json, models/feature_columns.json
  data/processed/test_scores.parquet       (for the API / dashboard)
  reports/results.md, reports/figures/pr_curve.png, reports/figures/feature_importance.png

Usage (from project root):
    python src/train.py
"""
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from config import PROCESSED, MODELS, FIGURES, ROOT

GRAPH = ["pagerank", "wcc_size", "n_neighbours", "log_scc_size", "in_cycle_scc"]
RULES = ["fan_out", "fan_in", "gather_scatter", "scatter_gather", "cycle", "rule_any"]
# Excluded from every model.
#   wcc_size: 99% of launderers sit in the giant component (likely a generator artifact);
#             retraining without it showed no loss on val.
#   nbr_mean_rule_any: needs the Cypher rule engine at serving time; dropped so the live
#             graph refresh depends only on the transaction window.
EXCLUDE = ["wcc_size", "nbr_mean_rule_any"]
KS = [100, 500, 1000, 5000]
# Only models whose features the live API can compute may become the production model.
# xgb_all needs rule scores (the Cypher rule engine) at serving time, so it stays a
# comparison only.
SERVABLE = ("xgb_tabular", "xgb_graph")


def load(split):
    return pd.read_parquet(PROCESSED / f"features_{split}.parquet")


def groups(cols):
    nbr = [c for c in cols if c.startswith("nbr_")]
    tab = [c for c in cols if c not in GRAPH + RULES + nbr]
    feature_sets = {
        "xgb_tabular": tab,
        "xgb_graph": tab + GRAPH + nbr,
        "xgb_all": tab + GRAPH + nbr + RULES,
    }
    return {k: [c for c in v if c not in EXCLUDE] for k, v in feature_sets.items()}


def fit(Xtr, ytr, Xva, yva):
    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=2000, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        scale_pos_weight=spw, tree_method="hist", eval_metric="aucpr",
        early_stopping_rounds=100, n_jobs=-1, random_state=42,
    )
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return model


def predict(model, X):
    return model.predict_proba(X, iteration_range=(0, model.best_iteration + 1))[:, 1]


def recall_at_k(y, score, k):
    k = min(k, len(y))
    top = np.argsort(-score, kind="stable")[:k]
    return y[top].sum() / max(y.sum(), 1), y[top].sum() / k


def metrics(name, y, score, ks):
    row = {"model": name,
           "pr_auc": average_precision_score(y, score),
           "roc_auc": roc_auc_score(y, score)}
    for k in ks:
        rec, prec = recall_at_k(y, score, k)
        row[f"recall@{k}"] = rec
        row[f"prec@{k}"] = prec
    return row


def pattern_recall(test, scores, k):
    """Recall at K broken down by laundering pattern type (FAN-OUT, CYCLE, ...)."""
    tx = pd.read_parquet(PROCESSED / "transactions.parquet",
                         columns=["src", "dst", "label", "pattern", "split"])
    tx = tx[(tx["split"] == "test") & (tx["label"] == 1)]
    tx["pattern"] = tx["pattern"].astype(str)
    acct_pat = pd.concat([tx[["src", "pattern"]].rename(columns={"src": "account"}),
                          tx[["dst", "pattern"]].rename(columns={"dst": "account"})]).drop_duplicates()
    rows = []
    for name, s in scores.items():
        top = set(test["account"].to_numpy()[np.argsort(-s, kind="stable")[:k]])
        for pat, g in acct_pat.groupby("pattern"):
            accts = set(g["account"])
            rows.append({"model": name, "pattern": pat, "accounts": len(accts),
                         "recall": len(accts & top) / len(accts)})
    return (pd.DataFrame(rows)
            .pivot_table(index=["pattern", "accounts"], columns="model", values="recall")
            .reset_index().sort_values("accounts", ascending=False))


def main():
    t0 = time.time()
    tr, va, te = load("train"), load("val"), load("test")
    cols = [c for c in tr.columns if c not in ("account", "label")]
    ytr, yva, yte = tr["label"].to_numpy(), va["label"].to_numpy(), te["label"].to_numpy()

    k_rules = int((te["rule_any"] > 0).sum())
    ks = KS + [k_rules]
    print(f"Test: {len(te):,} accounts, {yte.sum():,} laundering; "
          f"rules raise {k_rules:,} alerts -> comparing at K={k_rules:,}")

    scores = {"rules_only": te["rule_any"].to_numpy().astype(float)}
    results = [metrics("rules_only", yte, scores["rules_only"], ks)]
    models, val_ap = {}, {}
    for name, feats in groups(cols).items():
        t = time.time()
        m = fit(tr[feats], ytr, va[feats], yva)
        models[name] = (m, feats)
        val_ap[name] = average_precision_score(yva, predict(m, va[feats]))
        scores[name] = predict(m, te[feats])
        results.append(metrics(name, yte, scores[name], ks))
        print(f"  {name:12s} {len(feats):2d} features, {m.best_iteration + 1:4d} trees, "
              f"val PR-AUC {val_ap[name]:.3f}, test PR-AUC {results[-1]['pr_auc']:.3f}  "
              f"[{time.time() - t:.0f}s]")

    # Choose the production model on VALIDATION only (picking on test would leak),
    # among the models the live API can serve
    chosen = max((m for m in val_ap if m in SERVABLE), key=val_ap.get)
    print(f"\nProduction model chosen on val PR-AUC (servable models only): {chosen}")

    res = pd.DataFrame(results)
    show = ["model", "pr_auc", "roc_auc", "recall@100", "recall@1000",
            f"recall@{k_rules}", f"prec@{k_rules}"]
    table = res[show].to_string(index=False, float_format="{:.3f}".format)
    print("\n=== TEST RESULTS ===")
    print(table)

    pat = pattern_recall(te, scores, k_rules)
    pat_table = pat.to_string(index=False, float_format="{:.2f}".format)
    print(f"\n=== RECALL BY PATTERN (top {k_rules:,} alerts) ===")
    print(pat_table)

    # ---- save production model and scores
    best, feats = models[chosen]
    MODELS.mkdir(parents=True, exist_ok=True)
    best.save_model(MODELS / "xgb.json")
    (MODELS / "feature_columns.json").write_text(json.dumps(feats, indent=2))
    pd.DataFrame({"account": te["account"], "score": scores[chosen], "label": yte}) \
        .sort_values("score", ascending=False) \
        .to_parquet(PROCESSED / "test_scores.parquet", index=False)

    # ---- figures
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    for name, s in scores.items():
        p, r, _ = precision_recall_curve(yte, s)
        plt.plot(r, p, label=f"{name} (PR-AUC {average_precision_score(yte, s):.3f})")
    plt.axhline(yte.mean(), ls="--", c="grey", lw=1, label=f"random ({yte.mean():.3%})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-recall on test split")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(FIGURES / "pr_curve.png", dpi=150); plt.close()

    # Per-feature contributions (TreeSHAP built into XGBoost) on a test sample
    sample = te[feats].sample(min(20000, len(te)), random_state=0)
    contrib = best.get_booster().predict(
        xgb.DMatrix(sample), pred_contribs=True,
        iteration_range=(0, best.best_iteration + 1))[:, :-1]
    imp = pd.Series(np.abs(contrib).mean(axis=0), index=feats).sort_values()
    imp.tail(20).plot.barh(figsize=(7, 6))
    plt.xlabel("mean |SHAP value|"); plt.title(f"Top 20 features ({chosen})")
    plt.tight_layout(); plt.savefig(FIGURES / "feature_importance.png", dpi=150); plt.close()
    print("\nTop 10 features by mean |SHAP|:")
    for c, v in imp.sort_values(ascending=False).head(10).items():
        print(f"    {c:32s} {v:.3f}")

    (ROOT / "reports" / "results.md").write_text(
        "# Results (test split, Sep 9-10)\n\n"
        f"{len(te):,} accounts, {yte.sum():,} laundering ({yte.mean():.2%}). "
        f"Rules raise {k_rules:,} alerts, so ML is compared at the same K.\n\n"
        "```\n" + table + "\n```\n\n"
        f"Production model (chosen on validation PR-AUC): **{chosen}**\n\n"
        f"## Recall by pattern (top {k_rules:,} alerts)\n\n```\n" + pat_table + "\n```\n\n"
        "![PR curve](figures/pr_curve.png)\n\n![Feature importance](figures/feature_importance.png)\n")
    print(f"\nSaved model, scores, reports/results.md and figures  [{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()
