# 개인 속도 반영 강도/보정 강도 결과

작은 계수4회 학습,2개 기존 부모×4draw×4팔=32개 전체 생성. native ARGN 신규 학습0회. 새 head은 공유하므로2개 독립 학습 seed가 아니다.

| arm | normal_gap_seconds_log_w1 | onset_personal_gap_w1 | fraud_gap_seconds_log_w1 | normal_stay_personal_gap_w1 | onset_gap_w1 | fraud_ratio_log_w1 | normal_amount_log_w1 | class_0_merchant_tv | median_unique_merchants_per_customer | early_time | time100_199 | late1024_time | equal_customer_normal_gap_w1 | mid_personal_gap_w1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clock_joint | 0.137295 | 0.11678 | 0.099784 | 0.012279 | 0.513995 | 0.104591 | 0.061807 | 0.06776 | 538.25 | 0.843384 | 0.547257 | 0.175211 | 0.485626 | 0.024688 |
| history_only | 0.093009 | 0.14334 | 0.139139 | 0.053987 | 0.232068 | 0.109529 | 0.061505 | 0.067558 | 538.25 | 0.427376 | 0.291716 | 0.094736 | 0.287098 | 0.086562 |
| clock_prefix | 0.083968 | 0.09717 | 0.096087 | 0.006773 | 0.320574 | 0.110373 | 0.061573 | 0.06737 | 538.375 | 0.236051 | 0.188436 | 0.333324 | 0.267697 | 0.013691 |
| current | 0.052838 | 0.100208 | 0.103274 | 0.007663 | 0.290504 | 0.114237 | 0.061892 | 0.067718 | 537.75 | 0.331035 | 0.126845 | 0.079196 | 0.220775 | 0.018713 |
| init_only | 0.069124 | 0.099303 | 0.105201 | 0.005342 | 0.2626 | 0.110805 | 0.061695 | 0.067725 | 537.375 | 0.068592 | 0.162779 | 0.079196 | 0.194762 | 0.022165 |
| init_handoff | 0.064193 | 0.096961 | 0.104905 | 0.005435 | 0.263925 | 0.112353 | 0.06169 | 0.067726 | 537.125 | 0.092173 | 0.146383 | 0.079196 | 0.203189 | 0.020732 |
| correction_gain | 0.083713 | 0.096571 | 0.106104 | 0.00535 | 0.24455 | 0.110612 | 0.061713 | 0.067705 | 537.875 | 0.068592 | 0.270885 | 0.079196 | 0.171038 | 0.027406 |
| regression_gain | 0.069124 | 0.099303 | 0.105201 | 0.005342 | 0.2626 | 0.110805 | 0.061695 | 0.067725 | 537.375 | 0.068592 | 0.162779 | 0.079196 | 0.194762 | 0.022165 |
| joint_gain | 0.084016 | 0.098137 | 0.106093 | 0.00535 | 0.244796 | 0.111056 | 0.061668 | 0.067718 | 538.25 | 0.068592 | 0.270976 | 0.079196 | 0.170204 | 0.027306 |
| output_gain_shift | 0.084526 | 0.096883 | 0.106167 | 0.005359 | 0.244592 | 0.109145 | 0.061653 | 0.067755 | 537.875 | 0.068592 | 0.276235 | 0.079196 | 0.169478 | 0.027566 |

| arm | fit_seed | mid_pass | mid_wins | whole_pass | whole_wins | costs | passes |
|---|---|---|---|---|---|---|---|
| correction_gain | 20260930 | False | 0 | False | 0 | position_200_500,mid_personal | False |
| regression_gain | 20260930 | False | 0 | False | 0 |  | False |
| joint_gain | 20260930 | False | 0 | False | 0 | position_200_500,mid_personal | False |
| output_gain_shift | 20260930 | False | 0 | False | 0 | position_200_500,mid_personal | False |
| correction_gain | 20261001 | False | 0 | False | 0 | mid_personal | False |
| regression_gain | 20261001 | False | 0 | False | 0 |  | False |
| joint_gain | 20261001 | False | 0 | False | 0 | mid_personal | False |
| output_gain_shift | 20261001 | False | 0 | False | 0 | position_200_500,mid_personal | False |

| arm | fit_seed | normal_pass | improved_draws | late_pass | customer_pass | costs | passes |
|---|---|---|---|---|---|---|---|
| correction_gain | 20260930 | False | 2 | True | True | fraud_gap_seconds_log_w1 | False |
| regression_gain | 20260930 | True | 4 | True | True |  | True |
| joint_gain | 20260930 | False | 2 | True | True | fraud_gap_seconds_log_w1 | False |
| output_gain_shift | 20260930 | False | 2 | True | True | fraud_gap_seconds_log_w1 | False |
| correction_gain | 20261001 | False | 2 | True | True |  | False |
| regression_gain | 20261001 | True | 4 | True | True |  | True |
| joint_gain | 20261001 | False | 2 | True | True |  | False |
| output_gain_shift | 20261001 | False | 2 | True | True |  | False |

초반50개 거래간격, labels/lengths 정확 일치. 기존 모델 보존. 자동 교체/push 없음.
