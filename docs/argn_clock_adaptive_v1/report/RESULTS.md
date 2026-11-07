# 위치·개인 속도 시간 출력 결과

시간 출력4회 적합, 기존2부모×4draw×4팔=32개 자유 생성. native ARGN 재학습0회.
개발 평가이며 test 미사용. W1은 낮을수록 좋다. 고객 균등은 고객별 평균 W1과 다른 marginal 지표다.

| arm | normal_gap_seconds_log_w1 | onset_personal_gap_w1 | fraud_gap_seconds_log_w1 | normal_stay_personal_gap_w1 | onset_gap_w1 | fraud_ratio_log_w1 | normal_amount_log_w1 | class_0_merchant_tv | median_unique_merchants_per_customer |
|---|---|---|---|---|---|---|---|---|---|
| clock_joint | 0.13729 | 0.11678 | 0.09978 | 0.01228 | 0.51399 | 0.10459 | 0.06181 | 0.06776 | 538.25 |
| history_only | 0.09301 | 0.14334 | 0.13914 | 0.05399 | 0.23207 | 0.10953 | 0.0615 | 0.06756 | 538.25 |
| clock_prefix | 0.08397 | 0.09717 | 0.09609 | 0.00677 | 0.32057 | 0.11037 | 0.06157 | 0.06737 | 538.375 |
| event_two | 0.11788 | 0.10161 | 0.09618 | 0.01007 | 0.43256 | 0.11259 | 0.06234 | 0.06771 | 537.875 |
| customer_two | 0.13053 | 0.10196 | 0.10141 | 0.00612 | 0.30158 | 0.10642 | 0.06148 | 0.06736 | 538.375 |
| position_six | 0.18852 | 0.10346 | 0.10135 | 0.00817 | 0.30887 | 0.10961 | 0.06148 | 0.06747 | 539.0 |
| personal_six | 0.17002 | 0.10151 | 0.10011 | 0.00646 | 0.30535 | 0.11429 | 0.06103 | 0.06728 | 537.875 |

## 동일 위치 지원집합의 정상 gap W1

| start | end | clock_joint | history_only | clock_prefix | event_two | customer_two | position_six | personal_six |
|---|---|---|---|---|---|---|---|---|
| 6 | 50 | 0.84338 | 0.42738 | 0.23605 | 0.22476 | 0.25092 | 0.31436 | 0.28487 |
| 50 | 200 | 0.61709 | 0.32668 | 0.1912 | 0.34719 | 0.15875 | 0.11138 | 0.10649 |
| 200 | 500 | 0.46731 | 0.28549 | 0.29639 | 0.48957 | 0.23577 | 0.20761 | 0.29446 |
| 500 | 1024 | 0.0669 | 0.07195 | 0.12863 | 0.09221 | 0.17509 | 0.05087 | 0.15804 |
| 1024 | -1 | 0.17521 | 0.09474 | 0.33332 | 0.14749 | 0.38975 | 0.413 | 0.50151 |
| 1024 | 2048 | 0.16288 | 0.08875 | 0.32141 | 0.13528 | 0.37778 | 0.35365 | 0.48634 |
| 2048 | -1 | 0.46379 | 0.24065 | 0.60394 | 0.43598 | 0.64906 | 1.68248 | 0.83887 |

## 사전 등록한 보수적 교체 판단

| arm | fit_seed | normal_pass | improved_draws | late_pass | customer_pass | costs | passes |
|---|---|---|---|---|---|---|---|
| event_two | 20260930 | False | 0 | True | False | position_50_200,position_200_500 | False |
| customer_two | 20260930 | False | 0 | False | True | position_500_1024 | False |
| position_six | 20260930 | False | 0 | False | True | position_6_50 | False |
| personal_six | 20260930 | False | 0 | False | True | position_6_50,position_500_1024 | False |
| event_two | 20261001 | False | 0 | True | False | position_50_200,position_200_500 | False |
| customer_two | 20261001 | False | 0 | False | True | position_500_1024 | False |
| position_six | 20261001 | False | 0 | False | True | position_6_50 | False |
| personal_six | 20261001 | False | 0 | False | True | position_6_50 | False |

기준 통과 팔: []. 활성 모델은 최종 검토 전까지 clock_prefix. push 없음.
위치6계수와 위치3×속도2계수는 같은 크기의 서로 다른 basis이며 완전 nested 제거실험이 아니다.
이번 결과만으로 최신 생성기 대비 우위나 신규 기여가 확정되지 않는다.
