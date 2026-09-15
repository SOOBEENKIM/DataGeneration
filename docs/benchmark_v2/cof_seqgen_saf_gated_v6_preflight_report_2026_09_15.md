# CoF-SeqGen SAF v6 종합 재검토 및 preflight 결과

작성일: 2026-09-15  
상태: **static codec 수정 완료 / v6 intervention preflight FAIL / 5-seed 미실행 / held-out test 미접근**

## 결론

제안했던 `base(history) + gate(history) × delta(current gap)` 구조를 구현하고 CPU gate와 별도 pilot까지 실행했음. 그러나 본 5-seed 전에 고정한 intervention 기준에서 두 branch 모두 label·κ 선택성을 보이지 않아 FAIL함. 따라서 새로운 5-seed와 실제 데이터·baseline 실험은 시작하지 않았음.

재검토 과정에서 더 근본적인 기존 tensorizer 버그도 발견해 수정함. Controlled DGP의 정수형 `entity_label`이 기존 v2–v5 학습에서 전부 `<UNK>`로 변환되고 있었음. 이전 결과는 label-blind 모델의 개발 결과로는 남지만, 의도한 static-aware dependency routing의 증거로 사용할 수 없음.

## 연구 계보 재정리

### 기존 CoF 계열

- Chronological dependency와 coherence의 방향성 장점은 관찰됐음.
- 그러나 기존 실험에서 marginal fidelity와 combined 성능의 일관된 우월성은 확인되지 않았음.
- 이 한계로부터 gap support mismatch와 dependency route를 분리하는 SAF 가설을 세웠음.

### SAF support alignment

`SAF-C0`의 continuous hurdle-lognormal gap과 `SAF-U0`의 train-support-aligned discrete gap을 비교했음.

- v3 κ=0/1 gap KS 개선: `+0.0831 / +0.0795`, 모두 5/5 seed
- v4 κ=0/1 gap KS 개선: `+0.0846 / +0.0812`, 모두 5/5 seed
- v5 κ=0/1 gap KS 개선: `+0.0801 / +0.0774`, 모두 5/5 seed
- scaled W1 개선도 모든 버전·κ에서 약 `+0.198–0.203`, 모두 5/5 seed

따라서 support mismatch가 gap fidelity를 훼손한다는 개발 증거는 크고 반복적임. 다만 당시 static label codec 버그가 모든 후보에 공통으로 존재했으므로, 최종 논문 claim 전에는 수정된 tensorizer로 C0→U0를 다시 확인해야 함.

### Ordered decoding

Ordered hazard는 일부 dependency 지표에서 unordered decoder보다 개선됐지만 모든 gap marginal과 모든 버전에서 지배적이지 않았음. 독립 핵심 contribution으로 확정할 근거는 부족함.

### Dependency routing v3–v5

- v3 FiLM: repeat curve와 MI는 개선했지만 기존 absolute conditional TV에서 ordered branch가 실패함.
- metric audit: 기존 TV가 `p(mark_t | gap)`만 보아 실제 repeat transition에 둔감함을 확인함.
- v4 copy/new: 새 transition TV를 도입하고 κ=1 세 지표를 모두 개선했지만 decoder 변경과 gap route 변경이 섞여 있었음.
- v5 matched copy/new: decoder와 파라미터를 맞추고 gap 입력만 비교했음. Repeat curve와 MI 8개 판정은 통과했지만 transition TV 네 판정은 두 branch 모두 0/5 seed로 실패함.

당시에는 additive gap 효과가 label/history와 상호작용하지 못한다고 해석했음. 이번 재검토에서 static label 자체가 전부 `<UNK>`였음이 확인됐으므로, v2–v5는 애초에 의도한 label별 routing을 학습할 입력을 받지 못했음.

## 새로 발견한 static codec 결함

### 원인

`CategoryCodec.fit`에서는 pandas Series의 정수 값이 Python `int`로 key에 저장됐지만, `transform_split`의 row lookup 값은 NumPy `int64`였음. 기존 `_typed_key`가 실제 값과 함께 wrapper type까지 구분하여 다음을 서로 다른 범주로 처리했음.

- fit key: `{"type":"int","value":"0"}`
- transform key: `{"type":"int64","value":"0"}`

그 결과 controlled DGP validation 6,846개 entity가 모두 reserved `<UNK>` code 1로 변환됐음.

### 수정 및 회귀검증

- NumPy scalar를 Python scalar로 정규화한 뒤 typed key를 생성하도록 수정함.
- tensorizer state를 `cof-seqgen-saf-tensorizer-state-v2`로 올림.
- 수정 후 validation static code가 label 0=`3`, label 1=`4`로 분리됨.
- 실제 validation count: label 0 `6,515`, label 1 `331`.
- 정수형 static context 회귀 테스트를 추가함.
- SAF 전체 테스트: `64 passed, 1 skipped`.

기존 결과는 삭제하지 않고 다음 invalidation artifact로 보존함.

`artifacts/cof_seqgen_saf/development/STATIC_CONTEXT_CODEC_INVALIDATION_2026_09_15.json`

## v6 구현

모든 copy/new 후보에 동일한 세 모듈을 적용함.

`copy_logit = base(hidden) + sigmoid(gate(hidden)) × delta(current_gap)`

- U1/O1: 실제 current-gap context 사용
- C0/U0/O0: 같은 모듈에 zero gap context 사용
- U0/U1 및 O0/O1 state-dict key, tensor shape, 파라미터 수 동일
- delta head를 0으로 초기화하여 routed 모델도 no-gap 상태에서 시작함
- history에는 수정된 static categorical embedding이 포함됨

## CPU gate

- 독립 2회 training history 완전 일치
- model tensor state bitwise 일치
- best epoch: `9`
- best validation loss: `8.1644458668`
- train loss: `10.8977083 → min 7.2469007`
- 32-entity 생성·validation smoke 통과
- 모든 report `test_accessed=false`

## 별도 pilot 및 intervention audit

최종 5-seed와 겹치지 않는 seed `20260929`로 κ=0/κ=1의 U1/O1 네 모델을 학습함. 동일한 validation history에서 31개 train-fitted gap support 값을 바꾸며 latent copy probability의 범위를 측정함.

사전에 고정한 기준:

1. κ=1 label 1 평균 copy-probability range ≥ `0.05`
2. κ=1 label 0 및 κ=0 두 label의 평균 range ≤ `0.05`
3. κ=1 label 1 민감도가 가장 큰 비원인 cell의 최소 2배
4. 동일 파라미터 zero-gap control 최대 range ≤ `1e-8`

결과:

| branch | κ=1 label 1 range | 최대 비원인 range | 선택성 비율 | control max | 판정 |
|---|---:|---:|---:|---:|---|
| Unordered | 0.019390 | 0.018831 | 1.030 | 0.0 | FAIL |
| Ordered | 0.013434 | 0.013594 | 0.988 | 0.0 | FAIL |

Control 불변성과 비원인 절대 상한은 통과했지만, material sensitivity와 label·κ selectivity는 두 branch 모두 실패함.

## 실패 원인

static 입력 누락은 수정됐고, 학습된 h0의 label 간 hidden distance도 약 `2.34–3.03`으로 static 정보가 모델 내부에 존재함을 확인함. 그러나 κ=1의 평균 gate activation은 label별로 거의 같았음.

- Unordered: label 0 `0.6656`, label 1 `0.6737`
- Ordered: label 0 `0.7263`, label 1 `0.7197`

즉 현재 scalar gate는 전체 5% 미만인 label 1의 조건부 gap 신호를 선택적으로 켜지 못함. U branch는 큰 gate와 작은 delta, O branch는 작은 gate와 큰 delta를 학습해 multiplicative factorization의 scale 비식별성도 나타남. 구조적으로 control은 정확하지만, 표준 평균 likelihood 아래에서 희소 subgroup interaction을 안정적으로 학습하는 경로는 확보하지 못했음.

## 현재 주장 가능한 contribution

### 강하게 지지되지만 수정 후 확인이 필요한 항목

- **Train-support-aligned gap generation:** continuous support mismatch를 제거하면 gap KS와 W1이 반복적으로 크게 개선됨.
- **Support-aligned autoregressive factorization:** gap → mark → value 순서와 teacher-forced/generative route가 구현·검증됨.

### 아직 주장할 수 없는 항목

- current gap을 사용한 static/history-aware dependency routing 성공
- ordered hazard의 일관된 우월성
- 실제 데이터에서 sequential baseline 대비 fidelity·utility·privacy 전체 우월성
- SOTA 또는 held-out test 성능

## 다음 연구 판단

현재 v6의 최종 5-seed를 실행하면 안 됨. Preflight 실패를 무시하고 실행하거나 기준을 낮추면 validation-driven tuning이 됨.

과학적으로 방어 가능한 선택은 두 가지임.

1. **보수적 경로:** dependency routing을 현재 contribution에서 제외하고, 수정된 tensorizer로 C0→U0 support-alignment를 새 seed에서 재확인한 뒤 이를 핵심 contribution으로 확정함.
2. **새 exploratory family:** vector/bilinear interaction 또는 subgroup-balanced objective를 별도 가설로 사전등록하고, 현재 validation 결과와 분리된 개발 protocol에서 다시 시작함. 이는 v6의 단순 후속 retry가 아니라 새 연구 family여야 함.

현재 작업은 1번과 2번 중 어느 것도 자동 승인하지 않음. Held-out test와 실제 데이터·baseline 단계는 계속 봉인함.

## 산출물

- static codec invalidation: `artifacts/cof_seqgen_saf/development/STATIC_CONTEXT_CODEC_INVALIDATION_2026_09_15.json`
- intervention audit: `artifacts/cof_seqgen_saf/development/gated_v6_intervention_audit.json`
- corrected pilot: `artifacts/cof_seqgen_saf/development/gpu_dgp_k*_gated_v6_staticfix_pilot`
- CPU gate: `artifacts/cof_seqgen_saf/development/cpu_*gated_v6_staticfix*`
- model status: `configs/benchmark_v2/cof_seqgen_saf_model.yaml`

## 최종 상태

**STATIC_CODEC_REPAIRED_BUT_GATED_V6_FAILED_INTERVENTION_PREFLIGHT; FIVE_SEED_AND_REAL_DATA_NOT_STARTED**

Held-out test는 접근하지 않았음.
