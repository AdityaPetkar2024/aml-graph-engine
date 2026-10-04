# Results (test split, Sep 9-10)

354,447 accounts, 1,397 laundering (0.39%). Rules raise 10,711 alerts, so ML is compared at the same K.

```
      model  pr_auc  roc_auc  recall@100  recall@1000  recall@10711  prec@10711
 rules_only   0.013    0.567       0.014        0.028         0.162       0.021
xgb_tabular   0.223    0.899       0.070        0.208         0.528       0.069
  xgb_graph   0.290    0.928       0.069        0.291         0.568       0.074
    xgb_all   0.267    0.922       0.069        0.283         0.550       0.072
```

Production model (chosen on validation PR-AUC): **xgb_graph**

## Recall by pattern (top 10,711 alerts)

```
       pattern  accounts  rules_only  xgb_all  xgb_graph  xgb_tabular
         OTHER       635        0.17     0.25       0.29         0.36
GATHER-SCATTER       154        0.14     0.87       0.88         0.79
         STACK       151        0.18     0.77       0.76         0.68
SCATTER-GATHER       130        0.20     0.78       0.81         0.63
     BIPARTITE        84        0.27     0.71       0.73         0.60
       FAN-OUT        81        0.21     0.85       0.84         0.65
         CYCLE        78        0.23     0.77       0.81         0.65
        FAN-IN        75        0.21     0.80       0.77         0.76
        RANDOM        59        0.14     0.88       0.85         0.73
```

![PR curve](figures/pr_curve.png)

![Feature importance](figures/feature_importance.png)
