# 위치·개인 속도 시간 출력 결과

시간 출력4회 적합, 기존2부모×4draw×4팔=32개 자유 생성. native ARGN 재학습0회.
개발 평가이며 test 미사용. W1은 낮을수록 좋다. 고객 균등은 고객별 평균 W1과 다른 marginal 지표다.

| arm | normal_gap_seconds_log_w1 | onset_personal_gap_w1 | fraud_gap_seconds_log_w1 | normal_stay_personal_gap_w1 | onset_gap_w1 | fraud_ratio_log_w1 | normal_amount_log_w1 | class_0_merchant_tv | median_unique_merchants_per_customer |
|---|---|---|---|---|---|---|---|---|---|
| clock_joint | 0.13729 | 0.11678 | 0.09978 | 0.01228 | 0.51399 | 0.10459 | 0.06181 | 0.06776 | 538.25 |
| history_only | 0.09301 | 0.14334 | 0.13914 | 0.05399 | 0.23207 | 0.10953 | 0.0615 | 0.06756 | 538.25 |
| clock_prefix | 0.08397 | 0.09717 | 0.09609 | 0.00677 | 0.32057 | 0.11037 | 0.06157 | 0.06737 | 538.375 |
| event_two | 0.10413 | 0.10434 | 0.0941 | 0.0099 | 0.42639 | 0.1117 | 0.06155 | 0.06762 | 538.375 |
| customer_two | 0.13509 | 0.10142 | 0.10111 | 0.00611 | 0.30157 | 0.11145 | 0.06152 | 0.06761 | 538.125 |
| position_six | 0.0878 | 0.10106 | 0.11127 | 0.00614 | 0.26915 | 0.10944 | 0.061 | 0.06751 | 537.375 |
| personal_six | 0.05284 | 0.10021 | 0.10327 | 0.00766 | 0.2905 | 0.11424 | 0.06189 | 0.06772 | 537.75 |

## 동일 위치 지원집합의 정상 gap W1

| start | end | clock_joint | history_only | clock_prefix | event_two | customer_two | position_six | personal_six |
|---|---|---|---|---|---|---|---|---|
| 6 | 50 | 0.84338 | 0.42738 | 0.23605 | 0.22826 | 0.26025 | 0.30633 | 0.33103 |
| 50 | 200 | 0.61709 | 0.32668 | 0.1912 | 0.33713 | 0.15936 | 0.14242 | 0.05858 |
| 200 | 500 | 0.46731 | 0.28549 | 0.29639 | 0.47558 | 0.2299 | 0.19752 | 0.06914 |
| 500 | 1024 | 0.0669 | 0.07195 | 0.12863 | 0.08064 | 0.17985 | 0.20479 | 0.09698 |
| 1024 | -1 | 0.17521 | 0.09474 | 0.33332 | 0.15919 | 0.39847 | 0.08527 | 0.0792 |
| 1024 | 2048 | 0.16288 | 0.08875 | 0.32141 | 0.14673 | 0.38663 | 0.08379 | 0.07338 |
| 2048 | -1 | 0.46379 | 0.24065 | 0.60394 | 0.45278 | 0.65734 | 0.21662 | 0.23959 |

## 사전 등록한 보수적 교체 판단

| arm | fit_seed | normal_pass | improved_draws | late_pass | customer_pass | costs | passes |
|---|---|---|---|---|---|---|---|
| event_two | 20260930 | False | 0 | True | False | position_50_200,position_200_500 | False |
| customer_two | 20260930 | False | 0 | False | True | position_500_1024 | False |
| position_six | 20260930 | False | 1 | True | True | position_6_50,position_500_1024,fraud_gap_seconds_log_w1 | False |
| personal_six | 20260930 | True | 4 | True | True | position_6_50 | False |
| event_two | 20261001 | False | 0 | True | False | position_50_200,position_200_500 | False |
| customer_two | 20261001 | False | 0 | False | True | position_500_1024 | False |
| position_six | 20261001 | False | 1 | True | True | position_6_50,position_500_1024,fraud_gap_seconds_log_w1 | False |
| personal_six | 20261001 | True | 4 | True | True | position_6_50,fraud_ratio_log_w1 | False |

기준 통과 팔: []. 활성 모델은 최종 검토 전까지 clock_prefix. push 없음.
위치6계수와 위치3×속도2계수는 같은 크기의 서로 다른 basis이며 완전 nested 제거실험이 아니다.
이번 결과만으로 최신 생성기 대비 우위나 신규 기여가 확정되지 않는다.
