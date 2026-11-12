# init_only 시간 모듈 독립 재학습 검증

학습 난수3개 × 기존 부모2개 × 생성 난수2개. 각 팔12생성. 새 ARGN 본체 재학습은 아니다.
모든 결과는 반복 사용한 development이며 최종 test는 읽지 않았다. 낮을수록 좋은 W1.

| arm | normal_gap_seconds_log_w1 | onset_personal_gap_w1 | early_time | time100_199 | late1024_time | fraud_ratio_log_w1 |
|---|---|---|---|---|---|---|
| init_only | 0.063958 | 0.120572 | 0.084596 | 0.099827 | 0.074017 | 0.10197 |
| no_initial | 0.05052 | 0.118626 | 0.342188 | 0.080059 | 0.074017 | 0.10318 |
| no_calibration | 0.110182 | 0.117591 | 0.441744 | 0.550003 | 0.168004 | 0.099478 |
| gmr | 0.132234 | 0.119846 | 0.845354 | 0.579494 | 0.168004 | 0.099074 |
| phase_gmm | 0.075265 | 0.248234 | 0.781787 | 0.512851 | 0.240215 | 0.101677 |
| history_only | 0.123082 | 0.099052 | 0.316533 | 0.182406 | 0.082123 | 0.102648 |
| history_init_cal | 0.052395 | 0.080343 | 0.097851 | 0.088986 | 0.077573 | 0.105156 |

| comparator | normal_target | onset_target | costs | lower_normal_fits | lower_onset_fits | descriptive_screen |
|---|---|---|---|---|---|---|
| no_initial | False | False |  | 0 | 1 | False |
| no_calibration | True | False | time500_1023 | 3 | 1 | False |
| gmr | True | False | time500_1023 | 3 | 2 | False |
| phase_gmm | True | True | time500_1023 | 3 | 3 | False |
| history_only | False | False |  | 3 | 0 | False |
| history_init_cal | False | False |  | 0 | 0 | False |

강한 이력 대조 history_init_cal에도 같은 초기 간격 분포와 같은6계수/목적/예산 보정을 제공했다.
각 시간 fit의 차이는 PAIRED_EFFECTS.csv, 부모별 값은 PARENT_FIT_MEANS.csv에 기록했다.
희귀 사건의 표본 수는 SUPPORT.csv, 수렴/선택 정보는 FIT_AUDIT.csv에 보존했다.
이 결과를 최신 전체 생성기 대비 우월성이나 새로운 방법의 기여 확정으로 해석하지 않는다.
기존 모델과 실패 결과 모두 보존; 자동 교체 및 push 없음.
