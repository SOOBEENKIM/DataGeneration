# 초기 출력·후속 보정 2×2 대조 결과

새 초기 통계 mixture1개+2계수 보정2개,24개 전체 생성. 기존 ARGN 재학습0회. 낮을수록 좋은 W1.

| arm | normal_gap_seconds_log_w1 | onset_personal_gap_w1 | fraud_gap_seconds_log_w1 | normal_stay_personal_gap_w1 | onset_gap_w1 | fraud_ratio_log_w1 | normal_amount_log_w1 | class_0_merchant_tv | median_unique_merchants_per_customer | early_time | time100_199 | late1024_time | equal_customer_normal_gap_w1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clock_joint | 0.137295 | 0.11678 | 0.099784 | 0.012279 | 0.513995 | 0.104591 | 0.061807 | 0.06776 | 538.25 | 0.843384 | 0.547257 | 0.175211 | 0.485626 |
| history_only | 0.093009 | 0.14334 | 0.139139 | 0.053987 | 0.232068 | 0.109529 | 0.061505 | 0.067558 | 538.25 | 0.427376 | 0.291716 | 0.094736 | 0.287098 |
| clock_prefix | 0.083968 | 0.09717 | 0.096087 | 0.006773 | 0.320574 | 0.110373 | 0.061573 | 0.06737 | 538.375 | 0.236051 | 0.188436 | 0.333324 | 0.267697 |
| current | 0.052838 | 0.100208 | 0.103274 | 0.007663 | 0.290504 | 0.114237 | 0.061892 | 0.067718 | 537.75 | 0.331035 | 0.126845 | 0.079196 | 0.220775 |
| init_only | 0.069124 | 0.099303 | 0.105201 | 0.005342 | 0.2626 | 0.110805 | 0.061695 | 0.067725 | 537.375 | 0.068592 | 0.162779 | 0.079196 | 0.194762 |
| handoff_only | 0.069641 | 0.098448 | 0.105929 | 0.006949 | 0.266386 | 0.111043 | 0.061674 | 0.067698 | 537.375 | 0.096917 | 0.20065 | 0.079196 | 0.192135 |
| init_handoff | 0.064193 | 0.096961 | 0.104905 | 0.005435 | 0.263925 | 0.112353 | 0.06169 | 0.067726 | 537.125 | 0.092173 | 0.146383 | 0.079196 | 0.203189 |

| arm | fit_seed | early_pass | early_improved_draws | costs | passes |
|---|---|---|---|---|---|
| init_only | 20260930 | True | 4 | whole_normal_time,position_50_200,position_100_200 | False |
| handoff_only | 20260930 | True | 4 | whole_normal_time,position_50_200,position_100_200 | False |
| init_handoff | 20260930 | True | 4 | whole_normal_time,position_50_200 | False |
| init_only | 20261001 | True | 4 | whole_normal_time,position_50_200,position_100_200 | False |
| handoff_only | 20261001 | True | 4 | whole_normal_time,position_50_200,position_100_200 | False |
| init_handoff | 20261001 | True | 4 | whole_normal_time,position_50_200 | False |

| arm | fit_seed | normal_pass | improved_draws | late_pass | customer_pass | costs | passes |
|---|---|---|---|---|---|---|---|
| init_only | 20260930 | True | 4 | True | True |  | True |
| handoff_only | 20260930 | True | 4 | True | True | fraud_gap_seconds_log_w1 | False |
| init_handoff | 20260930 | True | 4 | True | True |  | True |
| init_only | 20261001 | True | 4 | True | True |  | True |
| handoff_only | 20261001 | True | 4 | True | True |  | True |
| init_handoff | 20261001 | True | 4 | True | True | fraud_ratio_log_w1 | False |

전체 사기 라벨/길이는 matched draw와 정확히 같음. 원본 생성/모델 보존. 아직 자동교체나 push 없음.
