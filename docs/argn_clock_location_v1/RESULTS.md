# 기존 GMR 관계를 유지한 시간 위치 학습

같은2개 native 부모×4GPU draw. 본 대조는 두 위치 계수 적합2개이며 원래 GMR 가중치는 정확히 같다.
앞선 진행 GMM3팔도 숨기지 않고 함께 비교한다. 모든 값은 development이며 최종 test는 미사용이다.

| arm | normal_gap_seconds_log_w1 | onset_personal_gap_w1 | fraud_gap_seconds_log_w1 | normal_stay_personal_gap_w1 | onset_gap_w1 | fraud_ratio_log_w1 | normal_amount_log_w1 |
|---|---|---|---|---|---|---|---|
| frozen | 0.12587 | 0.45554 | 0.13676 | 0.03104 | 1.49623 | 0.11413 | 0.06302 |
| phase_mixture | 0.0743 | 0.22868 | 0.13291 | 0.06766 | 0.31413 | 0.1063 | 0.06087 |
| history_only | 0.09301 | 0.14334 | 0.13914 | 0.05399 | 0.23207 | 0.10953 | 0.0615 |
| clock_joint | 0.13729 | 0.11678 | 0.09978 | 0.01228 | 0.51399 | 0.10459 | 0.06181 |
| progress_joint | 0.1758 | 0.14373 | 0.13255 | 0.02578 | 0.44666 | 0.11063 | 0.06182 |
| progress_prefix | 0.05895 | 0.11878 | 0.1299 | 0.02463 | 0.40506 | 0.10742 | 0.06082 |
| progress_rollout | 0.06467 | 0.10749 | 0.13325 | 0.02395 | 0.36099 | 0.11058 | 0.061 |
| clock_prefix | 0.08397 | 0.09717 | 0.09609 | 0.00677 | 0.32057 | 0.11037 | 0.06157 |
| clock_rollout | 0.12558 | 0.097 | 0.10302 | 0.006 | 0.28162 | 0.10787 | 0.06178 |

부모별 사전 비용 판정:

| fit_seed | arm | normal_better_than_GMR_and_history | costs | passes_screen | rollout_added_effect |
|---|---|---|---|---|---|
| 20260930 | clock_prefix | True |  | True | False |
| 20260930 | clock_rollout | False | phase_GMM:normal_gap_seconds_log_w1 | False | False |
| 20261001 | clock_prefix | False |  | False | False |
| 20261001 | clock_rollout | False | phase_GMM:normal_gap_seconds_log_w1 | False | False |
