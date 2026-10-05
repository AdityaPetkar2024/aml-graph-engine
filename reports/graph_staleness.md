# Graph-feature refresh schedule vs detection (test window)

```
                                          PR-AUC  ROC-AUC  recall@100  recall@1000  recall@10711
rules                                      0.013    0.567       0.014        0.028         0.162
offline                                    0.267    0.925       0.069        0.274         0.540
live, refresh never (Sep 7-8 batch only)   0.127    0.879       0.046        0.178         0.413
live, refresh every 24h                    0.193    0.896       0.064        0.205         0.452
live, refresh every 12h                    0.233    0.910       0.067        0.244         0.481
live, refresh every 6h                     0.251    0.919       0.068        0.262         0.515
live, refresh at end of window (fresh)     0.267    0.925       0.069        0.274         0.541
```
