# CoF-SeqGen SAF copy/new-v4 실험 결과

작성일: 2026-09-14  
상태: **dependency-routing 사전등록 gate FAIL / held-out test 미접근**

## 한 줄 결론

copy/new-v4는 κ=1에서 dependency error 3개를 두 branch 모두 5/5 seed에서 개선했으나, κ=0에서도 같은 개선이 일부 발생해 **현재 결과만으로 gap routing 고유 효과를 입증하지 못함**.

## 어디까지 확인됐는가

- `p(mark_t | mark_{t-1}, gap_bin)`을 직접 비교하는 `transition_conditioned_mark_tv` 구현 완료함.
- 같은 gap bin 안에서 mark transition을 깨뜨리는 train-only audit에서 새 지표가 변화를 탐지함.
- controlled κ=0/κ=1 및 전체 canonical 8개 데이터셋의 train-only metric audit 모두 통과함.
- 67-way mark 분류를 `이전 mark 복사 확률 + 새 mark 분포`로 분리한 copy/new-v4 구현함.
- CPU 결정론·loss 감소·생성 gate 통과함.
- 새 seed `20260914–20260918`로 κ=0/κ=1 각각 25개 학습과 5개 validation report를 완료함.
- 모든 학습·validation·집계에서 held-out test를 사용하지 않음.

## 성공 기준

Unordered(`U0 → U1`)와 ordered(`O0 → O1`) branch가 모두 아래 조건을 만족해야 PASS로 정의함.

1. κ=1에서 routed 모델이 세 dependency error를 평균적으로 낮출 것
2. 각 개선이 최소 4/5 seed에서 반복될 것
3. 각 개선폭이 κ=0보다 κ=1에서 더 클 것
4. 두 branch의 세 지표가 전부 통과할 것

Primary endpoint:

- `transition_conditioned_mark_tv`
- `short_gap_repeat_curve_l1`
- `gap_repeat_mi_error`

## 핵심 결과

양수는 routed 모델의 error 감소를 의미함.

| branch | endpoint | κ=1 평균 개선 | κ=1 양수 seed | κ=1−κ=0 interaction | interaction 양수 seed | 판정 |
|---|---|---:|---:|---:|---:|---|
| Unordered | transition-conditioned mark TV | +0.001814 | 5/5 | **−0.000021** | 4/5 | **실패** |
| Unordered | short-gap repeat curve L1 | +0.004538 | 5/5 | +0.010433 | 5/5 | 통과 |
| Unordered | gap-repeat MI error | +0.00015253 | 5/5 | +0.00017152 | 5/5 | 통과 |
| Ordered | transition-conditioned mark TV | +0.001992 | 5/5 | +0.002328 | 5/5 | 통과 |
| Ordered | short-gap repeat curve L1 | +0.003327 | 5/5 | **+0.001116** | **3/5** | **실패** |
| Ordered | gap-repeat MI error | +0.00008812 | 5/5 | +0.00009695 | 5/5 | 통과 |

총 12개 판정 항목 중 10개 통과, 2개 실패함.

## 어떻게 실패했는가

### 1. Unordered transition TV interaction

- κ=1 routing gain: `+0.001814`, 5/5 seed 개선함.
- κ=0 routing gain: `+0.001835`, 5/5 seed 개선함.
- 따라서 κ=1−κ=0 평균은 `−0.000021`, 95% CI `[−0.001219, 0.001177]`임.
- 4/5 seed에서는 interaction이 양수였지만 seed `20260916`의 interaction `−0.001742`가 나머지 작은 양수 효과를 상쇄함.

즉 copy/new-v4가 transition 분포를 개선한 것은 맞지만, 그 개선이 κ=1의 gap dependency 때문에 생겼다고 구분할 수 없었음.

### 2. Ordered repeat-curve interaction

- κ=1 routing gain: `+0.003327`, 5/5 seed 개선함.
- κ=0 routing gain: `+0.002211`, 5/5 seed 개선함.
- κ=1−κ=0 평균은 `+0.001116`으로 방향은 맞지만, 양수 seed가 3/5뿐임.
- seed `20260917`, `20260918`에서 각각 `−0.000117`, `−0.000288`로 반대 방향이 나옴.
- 95% CI도 `[−0.000540, 0.002772]`로 0을 포함함.

즉 평균 개선은 있으나 seed에 걸쳐 안정적인 κ-specific 효과로 재현되지 않았음.

## 왜 실패했는가

가장 큰 원인은 **비교군 구조가 맞지 않아 copy/new 효과와 gap routing 효과가 섞인 것**임.

- `U0/O0`: 기존 67-way categorical mark decoder
- `U1/O1`: copy/new decoder + current-gap route

따라서 `U0→U1`, `O0→O1` 비교에서는 다음 두 변화가 동시에 발생함.

1. 이전 mark 복사 여부를 별도 head로 분리한 구조 변화
2. 현재 gap을 copy 확률에 입력한 routing 변화

κ=0에서도 기본적인 mark 반복은 존재하므로 copy/new 구조 자체가 transition·repeat 지표를 개선할 수 있음. 실제 κ=0 개선이 이를 직접 보여줌. 현재 ablation으로는 이 구조 효과를 제거하고 순수 gap route 효과만 추정할 수 없음.

추가로 탈락한 두 interaction은 seed 변동에 비해 효과가 작음. 특히 unordered transition interaction 평균은 사실상 0이며, ordered repeat interaction도 두 seed에서 방향이 뒤집힘.

## 다음 실험

interactive-v5에서는 **decoder를 완전히 동일하게 맞춘 뒤 gap 입력만 ablation**해야 함.

1. U0/O0와 U1/O1 모두 동일한 `copy head + new-mark head` 사용함.
2. U0/O0의 copy head는 `history + static`만 입력받고, U1/O1만 `history + static + current gap`을 입력받게 함.
3. 파라미터 수까지 맞추기 위해 control에는 0 또는 학습된 null-gap context를 넣고, routed 모델에만 실제 current-gap context를 넣음.
4. 동일 history에서 current gap만 바꾸는 intervention audit 추가함.
   - κ=1 routed 모델의 copy probability는 gap에 따라 변해야 함.
   - κ=0 routed 모델과 no-route control은 변화가 없어야 함.
5. 사전등록 기준은 변경하지 않고 새로운 5개 seed로 κ=0/κ=1 재실험함.
6. 통과한 경우에만 실제 데이터·sequential baseline 비교로 넘어감.

현재 결과를 사후적으로 PASS 처리하지 않으며 실제 데이터/baseline 실험은 시작하지 않음.

## 산출물

- 5-seed 집계: `artifacts/cof_seqgen_saf/development/dgp_copy_new_v4_five_seed_mechanism_analysis.json`
- 사전등록 판정: `artifacts/cof_seqgen_saf/development/dgp_copy_new_v4_routing_decision.json`
- controlled metric audit: `artifacts/cof_seqgen_saf/metric_validity_audit_train_only_copy_new_v4_controlled.json`
- 전체 8개 데이터셋 metric audit: `artifacts/cof_seqgen_saf/metric_validity_audit_train_only_copy_new_v4.json`
- κ=0/κ=1 학습·validation: `artifacts/cof_seqgen_saf/development/gpu_dgp_k*_copy_new_v4*`

## 최종 상태

**COPY_NEW_V4_RECOVERS_KAPPA1_DEPENDENCY_BUT_MATCHED_ROUTING_EFFECT_NOT_PROVEN**

Held-out test는 계속 봉인함.
