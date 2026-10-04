# Rule-based baseline

Thresholds: {'fan_k': 5, 'gs_k': 3, 'gs_days': 2, 'sg_k': 3, 'sg_window': 259200, 'cycle_window': 432000, 'hub_cap': 30}

Account-level metrics per split.

```
split           rule  alerts  true_pos precision recall  seconds
train        fan_out   12258       189     0.015  0.054      6.4
train         fan_in    3805       298     0.078  0.086      5.6
train gather_scatter    4041       100     0.025  0.029      7.7
train scatter_gather     177        90     0.508  0.026     24.6
train          cycle    1162       104     0.090  0.030     32.0
train       ANY_RULE   19546       643     0.033  0.185     76.3
  val        fan_out    5378       103     0.019  0.069      1.5
  val         fan_in    1235        70     0.057  0.047      1.7
  val gather_scatter     670        34     0.051  0.023      2.0
  val scatter_gather       5         5     1.000  0.003     14.4
  val          cycle     197        22     0.112  0.015     83.3
  val       ANY_RULE    7252       215     0.030  0.144    102.8
 test        fan_out    8022       100     0.012  0.072      1.3
 test         fan_in    1478       109     0.074  0.078      1.3
 test gather_scatter    1598        48     0.030  0.034      3.1
 test scatter_gather       0         0     0.000  0.000     14.3
 test          cycle     180        19     0.106  0.014     57.0
 test       ANY_RULE   10711       227     0.021  0.162     77.1
```
