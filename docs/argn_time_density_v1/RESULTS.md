# 단일 조건부 시간 출력 비교

같은 GPU의2부모×4생성 평균. W1은 작을수록 좋다. CPU 대조는 섞지 않았다.
새 checkpoint 선택은 내부 검증만 사용했다. 이 표는 development이며 최종 test는 미사용이다.

| 모델 | 개인 대비 시작 간격 | 시작 간격 | 전체 사기 간격 | 정상 간격 | 정상 개인 간격 | 개인 대비 사기 금액 | 정상 금액 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 고정 개선본 | 0.4555 | 1.4962 | 0.1368 | 0.1259 | 0.0310 | 0.1141 | 0.0630 |
| 기존 경계 head | 0.1302 | 0.3952 | 0.2088 | 0.1250 | 0.0305 | 0.1148 | 0.0630 |
| 단순 phase GMM | 0.2287 | 0.3141 | 0.1329 | 0.0743 | 0.0677 | 0.1063 | 0.0609 |
| 일반 이력 밀도 | 0.1433 | 0.2321 | 0.1391 | 0.0930 | 0.0540 | 0.1095 | 0.0615 |
| 이력 + 개인 속도 입력 | 0.1473 | 0.2389 | 0.1405 | 0.0944 | 0.0539 | 0.1114 | 0.0612 |
| 개인 속도 대비 출력 | 0.1468 | 2.4529 | 1.9004 | 2.5970 | 0.0285 | 0.1097 | 0.0741 |

## 사전 screen

| fit_seed | arm | onset_better_than_simple | onset_better_than_clock | costs | passes_quality_screen | supports_relative_added_effect |
|---|---|---|---|---|---|---|
| 20260930 | history_only | True | True | frozen:normal_stay_personal_gap_w1,mixture:normal_gap_seconds_log_w1 | False | False |
| 20260930 | history_clock | True | False | frozen:normal_stay_personal_gap_w1,mixture:normal_gap_seconds_log_w1 | False | False |
| 20260930 | history_relative | True | True | frozen:normal_gap_seconds_log_w1,frozen:fraud_gap_seconds_log_w1,mixture:fraud_gap_seconds_log_w1,mixture:normal_gap_seconds_log_w1 | False | False |
| 20261001 | history_only | True | True |  | True | False |
| 20261001 | history_clock | True | False |  | True | False |
| 20261001 | history_relative | True | False | frozen:normal_gap_seconds_log_w1,frozen:fraud_gap_seconds_log_w1,mixture:fraud_gap_seconds_log_w1,mixture:normal_gap_seconds_log_w1 | False | False |

supports_relative_added_effect는 두 부모 모두 통과해야 한다. 생성4회는 독립학습4회가 아니다.
전체 지표·범위·부모별 값·변경 비용은 CSV에 보존한다. 최신 생성기 전체 직접 비교는 이번 실행 범위 밖이며 아직 완료하지 않았다.
