# CoF-SeqGen SAF matched-copy-v5 실험 결과

작성일: 2026-09-14  
상태: **dependency-routing 사전등록 gate FAIL / held-out test 미접근**

## 한 줄 결론

모든 비교 모델의 copy/new decoder와 파라미터 수를 동일하게 맞춰 v4의 confound는 제거했음. Current-gap route는 κ=1의 repeat curve와 gap-repeat MI를 복원했지만, 전체 mark transition 분포는 두 branch 모두 5/5 seed에서 악화시켜 최종 gate 실패함.

## 이번에 바꾼 내용

- C0/U0/O0/U1/O1 모두 동일한 `copy head + new-mark head` 사용함.
- U0/O0와 U1/O1의 copy head 입력 차원과 파라미터 수를 동일하게 맞춤.
- U0/O0에는 실제 current gap 대신 0 context를 입력함.
- U1/O1에만 실제 current-gap embedding을 입력함.
- 따라서 U0→U1과 O0→O1 비교에서 달라지는 것은 실제 current gap 사용 여부뿐임.
- 새로운 seed `20260919–20260923`으로 κ=0/κ=1 각각 학습 25개와 validation report 5개를 완료함.

## 검증 상태

- CPU 독립 2회 실행 history 완전 일치함.
- CPU 모델 state tensor bitwise 일치함.
- CPU train loss `10.9048 → 7.1444`로 감소함.
- CPU 생성·validation smoke test 통과함.
- SAF 전체 회귀 테스트 `62 passed, 1 skipped`임.
- 모든 학습·validation·집계에서 held-out test를 사용하지 않음.

## 사전등록 성공 기준

Unordered(`U0 → U1`)와 ordered(`O0 → O1`) 각각에 대해 아래를 모두 요구함.

1. κ=1에서 세 dependency error의 평균을 모두 낮출 것
2. 각 개선이 5개 seed 중 최소 4개에서 반복될 것
3. 각 개선폭이 κ=0보다 κ=1에서 더 클 것
4. 두 branch가 모두 통과할 것

Primary endpoint:

- `transition_conditioned_mark_tv`
- `short_gap_repeat_curve_l1`
- `gap_repeat_mi_error`

## 핵심 결과

양수는 current-gap route가 error를 낮췄다는 의미임.

| branch | endpoint | κ=1 평균 개선 | κ=1 양수 seed | κ=1−κ=0 interaction | interaction 양수 seed | 판정 |
|---|---|---:|---:|---:|---:|---|
| Unordered | transition-conditioned mark TV | **−0.001951** | **0/5** | **−0.002333** | **0/5** | **실패** |
| Unordered | short-gap repeat curve L1 | +0.001063 | 4/5 | +0.001513 | 4/5 | 통과 |
| Unordered | gap-repeat MI error | +0.00009691 | 5/5 | +0.00011237 | 5/5 | 통과 |
| Ordered | transition-conditioned mark TV | **−0.000966** | **0/5** | **−0.001190** | **0/5** | **실패** |
| Ordered | short-gap repeat curve L1 | +0.001535 | 4/5 | +0.000868 | 4/5 | 통과 |
| Ordered | gap-repeat MI error | +0.00009111 | 5/5 | +0.00008251 | 5/5 | 통과 |

12개 판정 항목 중 repeat curve·MI 관련 8개는 통과했고, transition TV 관련 4개는 모두 실패함.

## 어떻게 실패했는가

Current-gap route가 dependency를 전혀 학습하지 못한 것은 아님.

- Unordered와 ordered 모두 κ=1에서 gap-repeat MI error를 5/5 seed에서 낮춤.
- Repeat curve도 두 branch 모두 4/5 seed에서 개선함.
- κ=1−κ=0 interaction도 repeat curve와 MI에서는 사전등록 기준을 통과함.

그러나 더 완전한 분포를 비교하는 transition TV는 일관되게 악화됨.

- Unordered κ=1 routing gain: `−0.001951`, CI `[−0.003226, −0.000676]`, 0/5 seed 개선
- Ordered κ=1 routing gain: `−0.000966`, CI `[−0.001369, −0.000562]`, 0/5 seed 개선
- Unordered interaction: `−0.002333`, CI `[−0.003535, −0.001132]`, 0/5 seed 개선
- Ordered interaction: `−0.001190`, CI `[−0.001769, −0.000611]`, 0/5 seed 개선

따라서 seed 잡음에 의한 경계 실패가 아니라, aggregate repeat 관계를 맞추는 과정에서 mark별 transition 분포를 체계적으로 훼손한 실패임.

## 왜 실패했는가

확인된 구조적 한계는 copy logit이 current gap과 entity/history의 **상호작용을 표현하지 못한다는 점**임.

현재 v5 copy logit은 다음의 단순 선형 합임.

`copy_logit = w_h · hidden + w_g · gap_embedding + b`

따라서 gap을 바꿨을 때의 logit 변화 `w_g · Δgap_embedding`은 entity label이나 history와 무관하게 모든 entity에 동일함.

반면 controlled DGP의 κ=1 dependency는 다음 구조임.

- `entity_label=1`: current gap state가 repeat probability를 결정함.
- `entity_label=0`: 별도 receiver state가 repeat probability를 결정하며 current gap은 직접 driver가 아님.

실제 shared generation plan에서 `entity_label=1`은 약 4.8%, label 0은 약 95.2%임. 현재 additive route는 소수 label 1의 gap dependency를 학습하면서 동일한 gap 효과를 label 0에도 적용할 수밖에 없음. 이 구조는 repeat-by-gap 평균과 MI는 개선하지만 전체 mark transition 분포와 mark marginal을 왜곡한 관측 결과와 일치함.

추가 관측도 같은 방향임.

- κ=1 routed 모델은 두 branch 모두 validation loss가 control보다 소폭 높음.
- κ=1의 `gap_conditioned_mark_tv`와 `mark_sparse_tv`도 대체로 routed 모델에서 악화됨.
- transition TV 악화가 두 branch의 5/5 seed에서 동일하게 반복됨.

## 다음 실험

다음 버전은 decoder와 파라미터를 계속 동일하게 유지하면서, **current gap × history/static interaction만 추가하는 단일-factor 실험**이어야 함.

1. Base copy logit과 gap-dependent delta를 분리함.
2. `copy_logit = base(hidden) + gate(hidden) × delta(hidden, gap)` 형태의 gated interaction 사용함.
3. No-route control에는 동일한 모듈과 0 gap context를 사용해 파라미터 수를 계속 맞춤.
4. 학습 전 intervention audit을 추가함.
   - 동일 history에서 gap만 변경함.
   - κ=1 label 1에서는 copy probability가 변해야 함.
   - κ=1 label 0과 κ=0에서는 변화가 작아야 함.
5. 기존 세 primary endpoint와 gate는 변경하지 않음.
6. 별도 진단 지표로 `p(mark_t | mark_{t-1}, gap, entity_label)`을 추가해 label별 왜곡을 확인함.
7. 새로운 seed로 κ=0/κ=1 5-seed 재실험함.

현재 결과를 사후적으로 PASS 처리하지 않으며 실제 데이터·baseline 비교는 시작하지 않음.

## 산출물

- 5-seed 집계: `artifacts/cof_seqgen_saf/development/dgp_matched_copy_v5_five_seed_mechanism_analysis.json`
- 사전등록 판정: `artifacts/cof_seqgen_saf/development/dgp_matched_copy_v5_routing_decision.json`
- κ=0/κ=1 학습·validation: `artifacts/cof_seqgen_saf/development/gpu_dgp_k*_matched_copy_v5*`
- 모델 계약: `configs/benchmark_v2/cof_seqgen_saf_model.yaml`

## 최종 상태

**MATCHED_COPY_V5_RECOVERS_AGGREGATE_REPEAT_DEPENDENCY_BUT_HARMS_FULL_TRANSITION_DISTRIBUTION**

Held-out test는 계속 봉인함.
