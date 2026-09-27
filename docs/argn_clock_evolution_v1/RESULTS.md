# 개인 속도 진행과 반복 입력 보정 결과

같은2개 frozen ARGN 부모×4개 GPU sampling draw. 새 GMM4개와2계수 보정2개는 공유 적합이며 독립 ARGN6학습이 아니다.
development 평가이며 최종 test는 사용하지 않았다. W1은 낮을수록 좋다.

| arm | normal_gap_seconds_log_w1 | onset_personal_gap_w1 | fraud_gap_seconds_log_w1 | normal_stay_personal_gap_w1 | onset_gap_w1 | fraud_ratio_log_w1 | normal_amount_log_w1 |
|---|---|---|---|---|---|---|---|
| frozen | 0.12587 | 0.45554 | 0.13676 | 0.03104 | 1.49623 | 0.11413 | 0.06302 |
| phase_mixture | 0.0743 | 0.22868 | 0.13291 | 0.06766 | 0.31413 | 0.1063 | 0.06087 |
| history_only | 0.09301 | 0.14334 | 0.13914 | 0.05399 | 0.23207 | 0.10953 | 0.0615 |
| clock_joint | 0.13729 | 0.11678 | 0.09978 | 0.01228 | 0.51399 | 0.10459 | 0.06181 |
| progress_joint | 0.1758 | 0.14373 | 0.13255 | 0.02578 | 0.44666 | 0.11063 | 0.06182 |
| progress_prefix | 0.05895 | 0.11878 | 0.1299 | 0.02463 | 0.40506 | 0.10742 | 0.06082 |
| progress_rollout | 0.06467 | 0.10749 | 0.13325 | 0.02395 | 0.36099 | 0.11058 | 0.061 |

사전에 고정한 부모별 비용 판정:

| fit_seed | arm | normal_better_than_GMR_and_history | costs | passes_screen | rollout_added_effect |
|---|---|---|---|---|---|
| 20260930 | progress_joint | False | GMR:onset_personal_gap_w1,GMR:normal_stay_personal_gap_w1,GMR:fraud_gap_seconds_log_w1,phase_GMM:normal_gap_seconds_log_w1 | False | False |
| 20260930 | progress_prefix | True | GMR:normal_stay_personal_gap_w1,GMR:fraud_gap_seconds_log_w1 | False | False |
| 20260930 | progress_rollout | True | GMR:normal_stay_personal_gap_w1,GMR:fraud_gap_seconds_log_w1 | False | False |
| 20261001 | progress_joint | False | GMR:onset_personal_gap_w1,GMR:normal_stay_personal_gap_w1,GMR:fraud_gap_seconds_log_w1,phase_GMM:normal_gap_seconds_log_w1 | False | False |
| 20261001 | progress_prefix | True | GMR:normal_stay_personal_gap_w1,GMR:fraud_gap_seconds_log_w1 | False | False |
| 20261001 | progress_rollout | True | GMR:normal_stay_personal_gap_w1,GMR:fraud_gap_seconds_log_w1 | False | False |
