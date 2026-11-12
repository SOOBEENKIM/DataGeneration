# 同一 모델에서 손실만 세분화한 paired 결과

coarse/refined 각각2기존 부모×4draw 평균. W1은 낮을수록 좋음.
같은 팔의 전후만 손실 단독 대조다. 행/고객 질량 및 구조/예산은 그대로다.

| arm | loss | normal_gap_seconds_log_w1 | late_normal_gap_w1 | tail2048_gap_w1 | equal_customer_normal_gap_w1 | onset_personal_gap_w1 | fraud_ratio_log_w1 |
|---|---|---|---|---|---|---|---|
| customer_two | coarse | 0.130526 | 0.389746 | 0.649061 | 0.21751 | 0.101962 | 0.106422 |
| customer_two | refined | 0.135092 | 0.398468 | 0.657344 | 0.213439 | 0.101418 | 0.11145 |
| event_two | coarse | 0.117875 | 0.147493 | 0.435984 | 0.444506 | 0.101607 | 0.112588 |
| event_two | refined | 0.104132 | 0.159193 | 0.452783 | 0.432006 | 0.104337 | 0.111697 |
| personal_six | coarse | 0.170024 | 0.501507 | 0.838873 | 0.198378 | 0.10151 | 0.114289 |
| personal_six | refined | 0.052838 | 0.079196 | 0.239587 | 0.220775 | 0.100208 | 0.114237 |
| position_six | coarse | 0.188522 | 0.412996 | 1.682478 | 0.199207 | 0.103463 | 0.109608 |
| position_six | refined | 0.0878 | 0.085269 | 0.216615 | 0.176328 | 0.10106 | 0.109435 |

원래 clock_prefix 보존 판단은 screen.csv와 MODEL_DECISION.json에 별도로 기록한다.
학습 목적값과 자유 생성 품질을 구분한다. tail2048의 고객 지원이 작으므로 신중히 해석한다.
