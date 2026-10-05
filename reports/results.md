# Results (test split, Sep 9-10)

354,447 accounts, 1,397 laundering (0.39%). Rules raise 10,711 alerts, so ML is compared at the same K.

```
      model  pr_auc  roc_auc  recall@100  recall@1000  recall@10711  prec@10711
 rules_only   0.013    0.567       0.014        0.028         0.162       0.021
xgb_tabular   0.223    0.899       0.070        0.208         0.528       0.069
  xgb_graph   0.267    0.925       0.069        0.274         0.540       0.070
    xgb_all   0.262    0.928       0.069        0.270         0.554       0.072
```

Production model (chosen on validation PR-AUC): **xgb_graph**

## Recall by pattern (top 10,711 alerts)

```
       pattern  accounts  rules_only  xgb_all  xgb_graph  xgb_tabular
         OTHER       635        0.17     0.27       0.26         0.36
GATHER-SCATTER       154        0.14     0.86       0.85         0.79
         STACK       151        0.18     0.74       0.72         0.68
SCATTER-GATHER       130        0.20     0.80       0.78         0.63
     BIPARTITE        84        0.27     0.67       0.73         0.60
       FAN-OUT        81        0.21     0.88       0.83         0.65
         CYCLE        78        0.23     0.79       0.77         0.65
        FAN-IN        75        0.21     0.75       0.73         0.76
        RANDOM        59        0.14     0.88       0.83         0.73
```

![PR curve](figures/pr_curve.png)

![Feature importance](figures/feature_importance.png)
