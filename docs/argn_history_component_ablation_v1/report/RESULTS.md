# 이력 모델 초기 처리/보정 분리 결과

기존 이력망6개 고정. 새 학습0, 새 전체 생성24. 기존36출력과 합쳐60개 평가. 낮을수록 좋은 오류.
I=초기 처리만, C=보정만. 기존 neither/both와 구성 요소를 제외한 모든 가중치를 공유한다.

| arm | normal_gap_seconds_log_w1 | onset_personal_gap_w1 | normal_stay_personal_gap_w1 | early_time | time100_199 | late1024_time |
|---|---|---|---|---|---|---|
| history_only | 0.123082 | 0.099052 | 0.033969 | 0.316533 | 0.182406 | 0.082123 |
| history_initial_only | 0.122188 | 0.100582 | 0.03364 | 0.310201 | 0.18252 | 0.081822 |
| history_calibration_only | 0.052605 | 0.079223 | 0.036049 | 0.10501 | 0.089122 | 0.077265 |
| history_init_cal | 0.052395 | 0.080343 | 0.035996 | 0.097851 | 0.088986 | 0.077573 |
| init_only | 0.063958 | 0.120572 | 0.006468 | 0.084596 | 0.099827 | 0.074017 |

| contrast | metric | delta | lower_fits | lower_parent_fit_means | lower_draws | equal_draws | draws |
|---|---|---|---|---|---|---|---|
| I_minus_H | normal_gap_seconds_log_w1 | -0.000894 | 2 | 4 | 9 | 0 | 12 |
| I_minus_H | onset_personal_gap_w1 | 0.00153 | 0 | 3 | 6 | 0 | 12 |
| I_minus_H | normal_stay_personal_gap_w1 | -0.000329 | 2 | 4 | 9 | 0 | 12 |
| IC_minus_C | normal_gap_seconds_log_w1 | -0.000211 | 1 | 3 | 5 | 0 | 12 |
| IC_minus_C | onset_personal_gap_w1 | 0.00112 | 0 | 2 | 5 | 0 | 12 |
| IC_minus_C | normal_stay_personal_gap_w1 | -5.3e-05 | 2 | 3 | 7 | 0 | 12 |
| C_minus_H | normal_gap_seconds_log_w1 | -0.070477 | 3 | 6 | 12 | 0 | 12 |
| C_minus_H | onset_personal_gap_w1 | -0.019829 | 3 | 4 | 9 | 0 | 12 |
| C_minus_H | normal_stay_personal_gap_w1 | 0.00208 | 0 | 2 | 5 | 0 | 12 |
| IC_minus_I | normal_gap_seconds_log_w1 | -0.069793 | 3 | 6 | 12 | 0 | 12 |
| IC_minus_I | onset_personal_gap_w1 | -0.020239 | 3 | 5 | 10 | 0 | 12 |
| IC_minus_I | normal_stay_personal_gap_w1 | 0.002356 | 0 | 1 | 3 | 0 | 12 |
| IC_minus_H | normal_gap_seconds_log_w1 | -0.070687 | 3 | 6 | 12 | 0 | 12 |
| IC_minus_H | onset_personal_gap_w1 | -0.018709 | 3 | 4 | 10 | 0 | 12 |
| IC_minus_H | normal_stay_personal_gap_w1 | 0.002027 | 0 | 2 | 5 | 0 | 12 |
| interaction_IC_minus_I_minus_C_plus_H | normal_gap_seconds_log_w1 | 0.000683 | 1 | 2 | 5 | 0 | 12 |
| interaction_IC_minus_I_minus_C_plus_H | onset_personal_gap_w1 | -0.00041 | 2 | 4 | 5 | 0 | 12 |
| interaction_IC_minus_I_minus_C_plus_H | normal_stay_personal_gap_w1 | 0.000276 | 0 | 1 | 3 | 0 | 12 |

모든 fit/parent/draw 차이는 FIT_EFFECTS.csv와 DRAW_EFFECTS.csv에 보존한다.
W1 차이의 상호작용은 오차 척도상의 기술 통계이며 새로운 구조의 필연성/신규성 증명이 아니다.
이전 개발 자료를 재사용했다. 최종 test/CPAR/최신 전체 생성기 실험은 이번에 하지 않았다.
기존 모델 교체·commit·push 없음.
