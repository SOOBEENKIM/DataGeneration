# ARGN label-first order control

Completed fits: 4/4.
Two generation draws per completed fit. This is a development diagnostic, not a final test result.

| Arm | Fit seed | Fraud % | Mean run | Termination % | Personal amount W1 | Normal amount W1 |
|---|---:|---:|---:|---:|---:|---:|
| Real development | — | 0.6438 | 10.053 | 8.914 | 0 | 0 |
| B | 20260930 | 21.7985 | 59.076 | 1.599 | 1.0020 | 0.0491 |
| B | 20261001 | 49.4989 | 101.579 | 0.901 | 1.0855 | 0.0270 |
| B_event_label_first | 20260930 | 0.5214 | 1.014 | 98.593 | 1.1224 | 0.0849 |
| B_event_label_first | 20261001 | 0.6057 | 1.011 | 98.947 | 1.1545 | 0.0291 |
| B_event_weighted | 20260930 | 0.4153 | 1.022 | 97.881 | 1.0610 | 0.1049 |
| B_event_weighted | 20261001 | 0.5759 | 1.011 | 98.902 | 1.0787 | 0.0402 |
| B_label_first | 20260930 | 8.6708 | 28.451 | 3.583 | 0.9890 | 0.0284 |
| B_label_first | 20261001 | 48.3758 | 59.095 | 1.611 | 1.0708 | 0.0253 |

Teacher diagnostic uses real past and real length/position tokens. Free generation uses generated length.
Label-first and label-last probabilities have different current-field conditioning sets; NLL is not a standalone ranking.

| Arm | Fit seed | Condition | Events | Actual fraud % | Predicted fraud % |
|---|---:|---|---:|---:|---:|
| B | 20260930 | transition/first | 147 | 8.1633 | 68.4044 |
| B | 20260930 | transition/onset | 102 | 100.0000 | 0.2246 |
| B | 20260930 | transition/termination | 101 | 0.0000 | 68.9453 |
| B | 20260930 | previous_label/1 | 1133 | 91.0856 | 71.3099 |
| B | 20261001 | transition/first | 147 | 8.1633 | 77.3391 |
| B | 20261001 | transition/onset | 102 | 100.0000 | 0.3955 |
| B | 20261001 | transition/termination | 101 | 0.0000 | 69.5722 |
| B | 20261001 | previous_label/1 | 1133 | 91.0856 | 67.6993 |
| B_event_weighted | 20260930 | transition/first | 147 | 8.1633 | 21.0274 |
| B_event_weighted | 20260930 | transition/onset | 102 | 100.0000 | 1.2800 |
| B_event_weighted | 20260930 | transition/termination | 101 | 0.0000 | 2.2150 |
| B_event_weighted | 20260930 | previous_label/1 | 1133 | 91.0856 | 4.3716 |
| B_event_weighted | 20261001 | transition/first | 147 | 8.1633 | 32.5983 |
| B_event_weighted | 20261001 | transition/onset | 102 | 100.0000 | 0.8416 |
| B_event_weighted | 20261001 | transition/termination | 101 | 0.0000 | 0.7154 |
| B_event_weighted | 20261001 | previous_label/1 | 1133 | 91.0856 | 1.1608 |
| B_label_first | 20260930 | transition/first | 147 | 8.1633 | 77.6977 |
| B_label_first | 20260930 | transition/onset | 102 | 100.0000 | 0.1396 |
| B_label_first | 20260930 | transition/termination | 101 | 0.0000 | 76.7524 |
| B_label_first | 20260930 | previous_label/1 | 1133 | 91.0856 | 69.6330 |
| B_label_first | 20261001 | transition/first | 147 | 8.1633 | 77.6134 |
| B_label_first | 20261001 | transition/onset | 102 | 100.0000 | 0.3952 |
| B_label_first | 20261001 | transition/termination | 101 | 0.0000 | 70.2129 |
| B_label_first | 20261001 | previous_label/1 | 1133 | 91.0856 | 56.3825 |
| B_event_label_first | 20260930 | transition/first | 147 | 8.1633 | 30.1667 |
| B_event_label_first | 20260930 | transition/onset | 102 | 100.0000 | 0.8058 |
| B_event_label_first | 20260930 | transition/termination | 101 | 0.0000 | 1.5307 |
| B_event_label_first | 20260930 | previous_label/1 | 1133 | 91.0856 | 1.7418 |
| B_event_label_first | 20261001 | transition/first | 147 | 8.1633 | 38.5913 |
| B_event_label_first | 20261001 | transition/onset | 102 | 100.0000 | 0.5831 |
| B_event_label_first | 20261001 | transition/termination | 101 | 0.0000 | 0.6227 |
| B_event_label_first | 20261001 | previous_label/1 | 1133 | 91.0856 | 0.7257 |

See `evaluation/all_draws.csv`, `fit_means.csv`, `paired_fit_differences.csv`, `all_teacher_summaries.csv` and per-fit hazard/age-amount/numeric files for all results.

Negative absolute_error_change means closer to the development reference. Do not select the best draw.
Order changes are an existing ARGN capability. Improvement is baseline refinement, not new architectural novelty.
Joint prevalence, transition, amount, normal-quality and diversity assessment is required before the next structure.
CPAR adequacy and task-matched TabDiT/other recent generator comparisons remain outstanding.
