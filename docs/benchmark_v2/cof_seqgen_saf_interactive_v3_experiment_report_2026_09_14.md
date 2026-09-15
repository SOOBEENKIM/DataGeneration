# CoF-SeqGen SAF interactive-v3 실험 결과

작성일: 2026-09-14  
상태: **사전등록 dependency-routing gate FAIL / held-out test 미접근**

## 결론

interactive-v3의 κ=0/κ=1 비교를 5개 seed로 완료함.

- Unordered route(`SAF-U0 → SAF-U1`)는 사전등록 기준 전부 통과함.
- Ordered route(`SAF-O0 → SAF-O1`)는 repeat 기반 지표 2개를 5/5 seed에서 개선했으나, `gap_conditioned_mark_tv`를 4/5 seed에서 악화시켜 전체 gate 실패함.
- 따라서 현 단계에서 실제 데이터 및 baseline 비교로 넘어가지 않음.
- 다만 실패 진단 결과, 탈락 지표가 DGP의 실제 dependency인 `gap → 이전 mark 반복 여부`를 직접 측정하지 못함이 확인됨.

## 실행 범위

- 데이터: controlled joint semi-Markov DGP의 κ=0, κ=1
- 후보: SAF-C0, SAF-U0, SAF-O0, SAF-U1, SAF-O1
- seed: 20260826–20260830, 각 cell 5개
- 학습: κ별 25 jobs, 총 50 jobs
- validation: κ별 5 reports, report당 5개 SAF 후보 비교
- CPU gate: 동일 seed 2회 history 및 model state 완전 일치, train loss `11.0185 → 7.1988`
- held-out test: 접근하지 않음(`test_accessed=false`)

## 사전등록 성공 기준

U0→U1과 O0→O1 각각에 대해 아래를 모두 요구함.

1. κ=1에서 세 dependency error를 모두 평균적으로 감소시킬 것
2. 각 개선 방향이 5개 seed 중 최소 4개에서 반복될 것
3. 각 개선폭이 κ=0보다 κ=1에서 더 클 것
4. 두 route branch가 모두 통과할 것

Primary endpoint:

- `gap_conditioned_mark_tv`
- `short_gap_repeat_curve_l1`
- `gap_repeat_mi_error`

## 핵심 결과

양수는 routed 후보가 error를 낮췄다는 의미임.

| 비교 | endpoint | κ=1 평균 개선 | 양수 seed | κ=1−κ=0 interaction | interaction 양수 seed | 판정 |
|---|---|---:|---:|---:|---:|---|
| U0→U1 | gap-conditioned mark TV | +0.001680 | 5/5 | +0.002601 | 5/5 | 통과 |
| U0→U1 | short-gap repeat curve L1 | +0.003897 | 5/5 | +0.004068 | 4/5 | 통과 |
| U0→U1 | gap-repeat MI error | +0.00014992 | 5/5 | +0.00014237 | 5/5 | 통과 |
| O0→O1 | gap-conditioned mark TV | **−0.000870** | **1/5** | **−0.001608** | **1/5** | **실패** |
| O0→O1 | short-gap repeat curve L1 | +0.002978 | 5/5 | +0.003277 | 5/5 | 통과 |
| O0→O1 | gap-repeat MI error | +0.00006436 | 5/5 | +0.00006433 | 5/5 | 통과 |

Ordered branch의 실패 endpoint 95% CI:

- κ=1 routing gain: `−0.000870`, CI `[−0.002331, 0.000590]`
- κ interaction: `−0.001608`, CI `[−0.003572, 0.000357]`

## 어떻게 실패했는가

FiLM route 자체가 완전히 작동하지 않은 실패는 아님.

- O1은 핵심 transition 지표인 repeat curve와 gap-repeat MI를 κ=1의 5/5 seed에서 일관되게 개선함.
- 그러나 절대 mark 분포를 gap별로 비교하는 TV는 O0보다 나빠졌고, 이 때문에 엄격한 전체 gate에서 탈락함.
- O1의 validation receiver NLL도 O0보다 평균 `0.00272` 높아, route가 전체 multiclass mark likelihood를 안정적으로 개선했다고 말할 수 없음.

## 왜 실패했는가

확인된 주원인은 **가설과 한 endpoint의 불일치**임.

DGP에서 κ가 바꾸는 것은 절대 mark 종류가 아니라 다음 확률임.

`p(mark_t = mark_{t-1} | gap state, static label)`

반면 실패한 `gap_conditioned_mark_tv`는 이전 mark를 조건에 넣지 않고 다음만 측정함.

`p(mark_t | gap bin)`

새 mark가 거의 균등하게 뽑히는 DGP에서는 repeat 확률이 크게 변해도 절대 mark 주변분포는 거의 균등하게 유지됨. 따라서 이 TV는 복원해야 할 transition dependency에 둔감함.

Train-only 진단에서 현재 mark를 같은 gap bin 안에서만 섞어 `p(mark_t | gap bin)`을 정확히 보존했음. 그 결과:

- `gap_conditioned_mark_tv = 0.0`
- `mark_sparse_tv = 0.0`
- `short_gap_repeat_curve_l1 = 0.30085`
- `gap_repeat_mi_error = 0.00015015`(κ=1)

즉 transition dependency를 크게 파괴해도 탈락 endpoint는 전혀 반응하지 않음이 직접 확인됨.

추가 구조적 한계도 남아 있음. 현재 FiLM은 67-way mark softmax에 gap을 주입함. 실제 DGP는 “이전 mark 복사 vs 새 mark 생성” 구조이므로, 작은 dependency 신호를 전체 mark 분류 loss가 희석할 수 있음. routed 후보의 receiver NLL이 κ=0/1 모두에서 control보다 조금 나빴다는 결과가 이 한계와 일치함.

## 다음 실험

1. `p(mark_t | mark_{t-1}, gap_bin, static)`을 직접 비교하는 transition-conditional TV를 구현함.
2. 같은-gap-bin mark permutation이 새 지표에서는 크게 탐지되는지 train-only metric audit을 먼저 통과시킴.
3. SAF interactive-v4를 `repeat/copy head + new-mark head`로 분해함.
   - `p(repeat_t | history, gap_t, static)`을 직접 학습
   - repeat이면 이전 mark 복사
   - new이면 별도 mark 분포에서 생성
4. 성공 기준을 다시 사전등록하고 **새 seed**로 κ=0/κ=1 5-seed 실험을 수행함. 현재 결과로 기준을 사후 변경해 PASS 처리하지 않음.
5. 수정된 gate를 통과한 뒤에만 실제 데이터와 sequential baseline 비교로 넘어감.

## 산출물

- 집계: `artifacts/cof_seqgen_saf/development/dgp_interactive_v3_five_seed_mechanism_analysis.json`
- 사전등록 판정: `artifacts/cof_seqgen_saf/development/dgp_interactive_v3_routing_decision.json`
- κ=0 metric 진단: `artifacts/cof_seqgen_saf/development/gap_mark_metric_diagnosis_kappa_0_00.json`
- κ=1 metric 진단: `artifacts/cof_seqgen_saf/development/gap_mark_metric_diagnosis_kappa_1_00.json`
- κ=0/1 training 및 validation: `artifacts/cof_seqgen_saf/development/gpu_dgp_k*_interactive_v3*`

## 최종 상태

**INTERACTIVE_V3_PARTIAL_MECHANISM_RECOVERY_BUT_PREREGISTERED_GATE_FAILED**

실제 데이터/baseline 단계는 시작하지 않음. Held-out test는 계속 봉인함.
