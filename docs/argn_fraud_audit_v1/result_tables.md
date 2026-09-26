# 라벨 포함 공식 ARGN: 전체 결과표

2026-09-27. 두 학습 seed × 두 생성 seed. TV는 낮을수록 좋다. 모든 수치는 재사용된 development validation 결과다. 범위는 신뢰구간이 아니다.

## 실제 학습량

| seed | last_epoch | selected_epoch | updates | seconds | selected_check_loss | budget_reached |
| --- | --- | --- | --- | --- | --- | --- |
| 20260927 | 35 | 30 | 280 | 555.525 | 1.0378 | False |
| 20260928 | 32 | 27 | 256 | 512.954 | 1.0273 | False |

## 생성 수와 라벨

| run | real_events | generated_events | real_fraud_rate | generated_fraud_rate | generated_frauds | invalid_generated_label_rate | invalid_amount_rate | invalid_gap_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seed_20260927/generated_20260929 | 177997 | 165897 | 0.00643831 | 0.00346601 | 575 | 0 | 0 | 0 |
| seed_20260927/generated_20260930 | 177997 | 182364 | 0.00643831 | 0.00364655 | 665 | 0 | 0 | 0 |
| seed_20260928/generated_20260929 | 177997 | 158274 | 0.00643831 | 0.0139631 | 2210 | 0 | 0 | 0 |
| seed_20260928/generated_20260930 | 177997 | 174299 | 0.00643831 | 0.00939764 | 1638 | 0 | 0 | 0 |

## 관계 TV

| run | merchant_category_tv | category_amount_tv | gap_previous_category_category_tv | history_amount_tv | class_0_gap_previous_category_category_tv | class_1_gap_previous_category_category_tv | class_1_category_amount_tv | class_1_history_amount_tv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seed_20260927/generated_20260929 | 0.866664 | 0.218963 | 0.126743 | 0.0538713 | 0.125796 | 0.583349 | 0.638061 | 0.409771 |
| seed_20260927/generated_20260930 | 0.867167 | 0.220344 | 0.127915 | 0.061446 | 0.127133 | 0.562303 | 0.641549 | 0.379115 |
| seed_20260928/generated_20260929 | 0.83152 | 0.212223 | 0.15604 | 0.0659566 | 0.157122 | 0.448788 | 0.449572 | 0.348553 |
| seed_20260928/generated_20260930 | 0.830211 | 0.21141 | 0.156031 | 0.0587151 | 0.156731 | 0.467997 | 0.464486 | 0.334587 |

## 실제 고객을 반으로 나눈 real–real 참고 범위

12회 분할. 비교 표본 크기와 고객 구성이 생성 비교와 다르므로 통계적 기준선/유의성 검정으로 사용하지 않는다.

| summary | merchant_category_tv | category_amount_tv | gap_previous_category_category_tv | history_amount_tv | class_0_gap_previous_category_category_tv | class_1_gap_previous_category_category_tv | class_1_category_amount_tv | class_1_history_amount_tv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| min | 0.0509299 | 0.0271094 | 0.0561446 | 0.0186241 | 0.0559046 | 0.408743 | 0.0658766 | 0.0933636 |
| median | 0.0541242 | 0.0561027 | 0.0692453 | 0.0311816 | 0.069268 | 0.447741 | 0.0947747 | 0.129776 |
| max | 0.0577418 | 0.109335 | 0.0739533 | 0.0597047 | 0.0741515 | 0.4637 | 0.117584 | 0.204462 |

## 첫 거래의 fit에 없던 가맹점–유형 조합

| run | real_events | generated_events | real_pair_absent_from_fit | generated_pair_absent_from_fit |
| --- | --- | --- | --- | --- |
| seed_20260927/generated_20260929 | 147 | 147 | 0 | 0.891156 |
| seed_20260927/generated_20260930 | 147 | 147 | 0 | 0.877551 |
| seed_20260928/generated_20260929 | 147 | 147 | 0 | 0.829932 |
| seed_20260928/generated_20260930 | 147 | 147 | 0 | 0.809524 |

## 실제 이력에서의 거래 유형 예측

각 고객의 최초512개 거래까지, 총69,086개. 현재 merchant embedding만 교란하며 이전 이력은 그대로 둔다. 조회표 진단에는 같은 prefix와 전체177,997개 결과를 모두 기록한다.

| run | intervention | events | nll | accuracy |
| --- | --- | --- | --- | --- |
| seed_20260927 | teacher | 69086 | 2.11958 | 0.329459 |
| seed_20260927 | shuffle_current_merchant | 69086 | 2.60305 | 0.137235 |
| seed_20260927 | zero_history_after_first | 69086 | 2.24414 | 0.278667 |
| seed_20260928 | teacher | 69086 | 1.89974 | 0.391049 |
| seed_20260928 | shuffle_current_merchant | 69086 | 2.73293 | 0.131676 |
| seed_20260928 | zero_history_after_first | 69086 | 2.02701 | 0.338549 |

## 0초 초과 5초 이하 간격

0건의 조건은 사기율 NA다. 실제 표본45개·사기1개인 조건의 일치/불일치만으로 우월성을 결론내리지 않는다.

| source | events | frauds | invalid_labels | fraud_rate |
| --- | --- | --- | --- | --- |
| original_validation | 45 | 1 | 0 | 0.0222222 |
| seed_20260927/generated_20260929 | 42 | 0 | 0 | 0 |
| seed_20260927/generated_20260930 | 26 | 0 | 0 | 0 |
| seed_20260928/generated_20260929 | 36 | 1 | 0 | 0.0277778 |
| seed_20260928/generated_20260930 | 25 | 0 | 0 | 0 |

상세 조건별 빈도·사기율은 conditional_risk.csv, 위치별 고객/거래 수는 position_curves.csv, 정상·사기별 실제 이력 예측은 probe_seed_*.csv에 남긴다.
