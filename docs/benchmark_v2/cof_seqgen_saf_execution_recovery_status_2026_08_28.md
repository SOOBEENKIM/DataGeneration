# CoFSeqGen-SAF 실행 복구 및 현재 상태 감사 보고서

- 감사 시각: 2026-08-28 14:45 KST
- 저장소: `cof-seqgen-0707-2119-Version3-complete`
- 목적: 연결 및 이전 작업 중단 이후, 데이터 수집·구현·실험·결과·중단 지점을 파일과 산출물로 재구성한다.
- 이 문서는 **현재 실행 상태의 단일 복구 기준 문서**이다. 사전등록 문서나 과거 단계 보고서를 수정하거나 소급 해석하지 않는다.

## 0. 실제로 감사한 연결 폴더와 근거

이 감사는 첨부 문서만 읽고 작성한 추정이 아니다. 사용자가 연결한 다음 실제 코드 저장소를 직접 읽었다.

```text
<LOCAL_WORKSPACE>/cof-seqgen-0707-2119-Version3-complete/
└── cof-seqgen-0707-2119-Version3-complete/   ← 실제 Git/code repository
```

현재 branch와 tracked parent는 다음과 같다.

- branch: `benchmark-v2-redesign`
- HEAD: `7abe12b68780673666deef951dc7e1741f74f6af`
- HEAD subject: `docs: preregister CoF-ZDH-v3 D1`

다음 evidence를 서로 대조했다.

- 기존 1,063줄 종합 기록 `research_implementation_execution_overview_2026_08_26.md`
- SAF 사전등록 `preregistered_cof_seqgen_saf_source_protocol.md`
- 데이터 수집 보고서와 canonical 디렉터리·manifest
- tensorizer, model, training, validation, metric, baseline wrapper 실제 Python source
- κ=1 matrix config의 예정 candidate/seed 목록
- 개별 `training_report.json`, best/latest checkpoint, log, matrix manifest
- 개별 `validation_comparison.json`, validation log와 manifest
- 파일 수정 시각, artifact SHA-256, 현재 process table

중요하게, `research_implementation_execution_overview_2026_08_26.md`는 SAF 이전까지의 연구 계보와 CoF-ZDH-v3 D1 source-only 상태를 정리한 역사 snapshot이다. 그 뒤 새 family인 CoFSeqGen-SAF가 untracked worktree에서 구현·실행되었으므로 그 overview 하나만으로 현재 κ=1 상태를 알 수 없다. 이 문서가 그 이후를 이어 붙인다.

## 1. 최종 판정

**데이터 수집과 canonical materialization은 모두 완료되었다. 그 이후의 tensorizer, 모델, 학습·검증 runner, metric, baseline wrapper 구현도 상당 부분 완료되었고, `saf-static-gru-state-v2`에 대해서는 Controlled DGP κ=1의 CPU gate와 5-seed GPU 개발 검증까지 완료되었다. 그러나 현재 소스의 최종 모델 버전인 `saf-interactive-gap-route-v3`는 소스와 단위·통합 테스트만 존재하며, 전용 config·CPU gate·GPU checkpoint·validation 결과는 하나도 없다.**

따라서 현재 상태를 “데이터 수집만 끝남”이라고 표현하는 것도 틀리고, “최종 CoFSeqGen-SAF 모델 실험까지 끝남”이라고 표현하는 것도 틀리다. 정확한 표현은 다음과 같다.

> **데이터 단계 완료 → 공통 구현 완료 → static-v2 κ=1 개발 실험 완료 → static-v2 결과에서 interaction 표현력 한계 발견 → interactive-v3 소스 구현·테스트 완료 → interactive-v3 실행 config 생성 직전에 중단**

| 영역 | 판정 | 근거/제한 |
|---|---|---|
| 원천 데이터 수집 | 완료 | 선언된 모든 source가 acquisition manifest와 함께 존재 |
| canonical 변환·split | 완료 | 8개 canonical view, 모든 gate 0 failure |
| metric validity audit | 완료 | 8개 view에서 등록 sensitivity check 6/6; H5 margin 해석 문제는 별도 |
| tensorizer·auxiliary heads | 완료 | train-only codec/support, static/event field, decode 및 auxiliary head 구현 |
| checkpoint/training runner | 완료 | deterministic checkpoint/resume, matrix runner 구현 |
| validation runner | 완료 | train/validation 전용 생성·fidelity·utility·privacy 평가 구현 |
| 외부 baseline wrapper | 부분 완료 | CPAR 실행 완료, REaLTabFormer wrapper/smoke 존재, TabularARGN 격리 필요, TabDiT 실행 불가 |
| static-v2 CPU gate | 완료·통과 | v2에만 유효; v3 gate가 아님 |
| static-v2 κ=1 GPU 학습 | 완료 | 5 candidates × 5 seeds = 25 checkpoints |
| static-v2 κ=1 validation | 완료 | 5 seed reports, 동일 shared generation plan |
| static-v2 κ=0 GPU 학습 | 미실행 | 유효한 v2 κ=0 결과 없음 |
| interactive-v3 구현 | 소스 완료 | FiLM history×gap route 구현, 테스트 통과 |
| interactive-v3 CPU/GPU 실험 | 미시작 | 전용 config와 artifact가 없음 |
| real-data 모델 실험 | 미시작 | AMLSim/Sparkov/Berka/H&M/Citi Bike는 데이터만 준비됨 |
| held-out test | 봉인 유지 | 모든 확인된 report의 `test_accessed=false` |
| 논문 superiority/SOTA 결론 | 불가 | v3 미학습, κ=0 미실행, 주요 baseline 미완료 |

### 1.1 기존 연구에서 새 contribution이 나온 이유

기존 종합 overview가 기록한 핵심 연구 결과는 다음과 같다.

1. frozen non-v3 CoF는 AMLSim과 Sparkov에서 sequence coherence 방향의 장점을 반복해서 보였다.
2. 그러나 marginal fidelity와 combined score에서는 empirical IID/CTGAN 등보다 전반적으로 나빴다.
3. receiver/gap fidelity를 고치려던 joint-discrete CoF v3, continuous-gap CCMTPP C1, hurdle-RQS HCMTTPP H1은 각각 support/calibration/numerical gate를 통과하지 못했다.
4. CoF-ZDH-v3 D1은 bounded discrete hazard를 설계했지만 `SPECIFIED_NOT_IMPLEMENTED` 상태였다.

여기서 새 SAF family가 세운 논문 방향은 “또 하나의 decoder를 검증”하는 데 그치지 않고 다음 mechanism을 분해해 증명하는 것이다.

> **CoF 계열의 chronological dependency modeling이 가진 coherence 장점을 보존하되, 실제 train gap support에 맞춘 decoder와 current-gap→mark dependency route를 결합하면 temporal-relational fidelity를 개선하고 기존의 marginal fidelity degradation을 줄일 수 있는가?**

`Support-Aligned Autoregressive Factorization`의 고정 factorization은 다음과 같다.

```text
p(events | static)
  = product_t
      p(gap_t | strict_past, static)
      p(mark_t | strict_past, static, gap_t)
      p(value_t | strict_past, static, gap_t, mark_t)
```

계획된 contribution과 검증 질문은 다음처럼 분리되어 있다.

| Contribution / hypothesis | 비교 | 밝히려는 것 |
|---|---|---|
| SAF-H1 support alignment | C0 vs U0 | continuous support mismatch가 gap distribution error를 만드는가 |
| SAF-H2 ordered decoding | U0 vs O0, U1 vs O1 | 동일 support에서 순서를 이용한 hazard가 unordered categorical보다 유리한가 |
| SAF-H3 dependency routing | U0 vs U1, O0 vs O1 | current gap을 mark 생성에 명시적으로 넣어 joint dependency를 회복하는가 |
| SAF-H4 complete model | 최종 SAF vs primary sequential baselines | temporal-relational fidelity 우위를 보이는가 |
| SAF-H5 constraints | marginal/utility/privacy endpoints | H4 개선이 marginal fidelity·utility·privacy 희생으로 생긴 것이 아닌가 |

Controlled κ=0/κ=1은 단순 추가 데이터가 아니라 H3 mechanism identification의 negative/positive control pair다. AMLSim/Sparkov는 복잡한 금융 sequence, Berka/H&M은 real irregular sequential tabular data, Citi Bike는 cross-domain/Seq2Synth-aligned robustness 역할이다.

Primary sequential baseline 계획은 TabularARGN, TabDiT, CPAR, REaLTabFormer, empirical sequence sampler다. CTGAN/TVAE/GaussianCopula는 flattened marginal control이며 TabPFN은 generator baseline이 아니라 downstream utility evaluator 후보일 뿐이다.

static-v2의 additive route가 H3를 안정적으로 지지하지 못했기 때문에 현재 source는 U1/O1에 history×current-gap FiLM interaction을 넣는 `saf-interactive-gap-route-v3`로 발전했다. 따라서 최종 모델 contribution 후보는 단순한 “gap embedding concat”이 아니라 **support-aligned gap decoder + interaction-capable dependency routing + 공통 autoregressive factorization**의 결합이다.

## 2. 연결 중단이 실제로 남긴 상태

### 2.1 κ=1 실행의 정확한 시간축

저장소에는 Codex 토큰이 소진된 정확한 시각 자체는 기록되어 있지 않다. 그러나 file mtime, log terminal line, manifest 생성 순서를 이용하면 실제 실행 상태는 다음처럼 복구된다.

| 시각 (KST) | 실행/변경 | 판정 |
|---|---|---|
| 2026-08-27 00:10:29–00:10:31 | 초기 static-first-only κ=1 replication validation seed 20260827–29 | 결과 생성됐으나 이후 모델 결함으로 무효화 |
| 00:10:42 | 같은 초기 validation seed 20260830 | source version이 바뀌어 checkpoint-version mismatch로 중단 |
| 00:11:09 | `STATIC_FIRST_ONLY_PILOT_INVALIDATION_2026_08_27.json` | 초기 κ=0/κ=1 계열을 paper analysis에서 제외 |
| 00:12:02–00:16:00 | static-v2 CPU overfit, validation, repeat determinism gate | 전부 PASS |
| 00:28:49 | static-v2 κ=1 training matrix manifest | 5 candidates × 5 seeds = 25/25 완료 |
| 00:32:54–00:32:57 | static-v2 κ=1 validation seed 20260826–28 | 3/5 완료 |
| 00:34:43 | `models/cof_seqgen_saf.py` 수정 | 현재 FiLM `saf-interactive-gap-route-v3` source가 저장됨 |
| 00:36:52–00:36:53 | 이미 v2 code를 메모리에 load한 validation seed 20260829–30 | 나머지 2/5 완료, matrix manifest 생성 |
| 01:03:36 | 별도 CPAR 20-epoch κ=1 development process | fit, sample, external validation report 완료 |

여기서 매우 중요한 구분은 다음과 같다.

- 00:10의 seed 20260830 version mismatch는 **무효화된 초기 static-first-only 실행**의 실패다.
- 이후 새 output root `gpu_dgp_k1_static_v2_validation`에서 수행한 static-v2 validation은 seed 20260826–30 모두 성공했다.
- 00:34에 Python source가 v3로 바뀌었지만, 이미 시작되어 v2 module을 메모리에 load한 seed 20260829/30 child process는 v2 결과를 정상 완료했다.
- 따라서 current source file이 v3라는 사실과 stored static-v2 report 5개가 모두 유효하다는 사실은 모순이 아니다.

### 2.2 예정 job 대 실제 artifact 전수 대조

`cof_seqgen_saf_gpu_dgp_k1_static_v2_matrix.yaml`과 실제 산출물을 candidate×seed 단위로 비교하고 best checkpoint SHA까지 다시 계산했다.

```text
K1_STATIC_V2_TRAIN_EXPECTED          25
K1_STATIC_V2_TRAIN_MANIFEST_RECORDS  25
K1_STATIC_V2_TRAIN_FULL_ARTIFACT_OK  25
K1_STATIC_V2_TRAIN_FAILURES           0

K1_STATIC_V2_VALIDATION_EXPECTED       5
K1_STATIC_V2_VALIDATION_MANIFEST_RECORDS 5
K1_STATIC_V2_VALIDATION_FULL_REPORT_OK 5
K1_STATIC_V2_VALIDATION_FAILURES       0
```

각 training cell에서 `training_report.json`, `checkpoint_best.pt`, `checkpoint_latest.pt`, `tensorizer_state.json` 존재, implementation version, test seal, manifest checkpoint SHA를 모두 확인했다. 각 validation report에는 5 SAF candidates와 empirical control이 모두 있고, 5 seeds가 같은 shared generation plan hash를 사용한다.

반대로 다음은 실제로 없다.

```text
K0_STATIC_V2_TRAIN_ROOT_EXISTS       false
K0_STATIC_V2_VALIDATION_ROOT_EXISTS  false
STATIC_V2_PAIRED_ANALYSIS_EXISTS     false
INTERACTIVE_V3_CONFIG_COUNT          0
INTERACTIVE_V3_ARTIFACT_COUNT        0
```

따라서 사용자가 기억한 “κ=1을 계속 실행하다 토큰이 끝남”은 artifact 관점에서 다음처럼 판정된다.

> **κ=1 static-v2 학습·validation은 전부 끝났다. 토큰/작업 중단으로 남은 것은 κ=1의 미완성 cell이 아니라, 그 결과를 반영한 interactive-v3 전용 config·CPU gate·새 κ=1 실행이다.**

### 2.3 실제 중단 지점

static-v2 결과를 조사한 뒤, 단순 additive gap route로는 “이전 mark가 무엇이든 현재 gap이 짧을 때 그 mark를 반복”하는 **history × current-gap interaction**을 충분히 표현하기 어렵다는 판단으로 모델 소스를 v3 FiLM route로 변경했다.

현재 확인되는 상태는 다음과 같다.

- `models/cof_seqgen_saf.py`의 버전은 `saf-interactive-gap-route-v3`이다.
- `gap_to_mark_film`과 `history_by_gap_film_interaction`이 실제 소스에 존재한다.
- 그러나 이름에 `interactive_v3`가 들어간 config와 artifact는 0개이다.
- 마지막 대규모 config 추가 patch는 저장소에 반영되지 않았다.
- 따라서 v3는 단 한 epoch도 학습되지 않았다.

**재개할 때 기존 static-v2 checkpoint를 v3로 resume하면 안 된다.** 구현 버전 gate가 이를 거부하도록 되어 있으며, 설령 강제로 우회해도 서로 다른 구조의 결과가 섞여 실험 계보가 훼손된다.

### 2.4 현재 실행 프로세스와 GPU 확인 한계

- 프로세스 테이블에서 CoFSeqGen-SAF training, validation, external baseline Python 작업은 발견되지 않았다.
- 현재 sandbox에서 `nvidia-smi`는 NVIDIA driver와 통신하지 못했다.
- 그러므로 “연구 프로세스가 현재 실행 중이지 않다”는 것은 확인했지만, 다른 사용자까지 포함한 워크스테이션 GPU 전체 idle 여부는 이 감사에서 확정하지 않는다.
- 기존 model config의 `GPU_DEVELOPMENT_RUNNING` 표기는 사실과 달라 `V3_SOURCE_IMPLEMENTED_AND_TESTED_NOT_TRAINED_TEST_SEALED`로 정정했다.

## 3. 데이터 수집 및 materialization 상태

### 3.1 완료 범위

`docs/benchmark_v2/cof_seqgen_saf_data_collection_report_2026_08_26.md`의 acquisition/materialization 판정은 현재 파일 상태와 일치한다.

| Canonical dataset/view | Entities | Events | Train / validation / test entities | Failed gates |
|---|---:|---:|---:|---:|
| Controlled joint semi-Markov κ=0 | 45,645 | 1,095,902 | 31,951 / 6,846 / 6,848 | 0 |
| Controlled joint semi-Markov κ=1 | 45,645 | 1,095,902 | 31,951 / 6,846 / 6,848 | 0 |
| AMLSim | 9,999 | 1,323,234 | 6,999 / 1,499 / 1,501 | 0 |
| Sparkov | 983 | 1,296,675 | 688 / 147 / 148 | 0 |
| Berka full | 4,500 | 1,056,320 | 3,150 / 675 / 675 | 0 |
| Berka deterministic nested 900 | 900 | 211,249 | 630 / 135 / 135 | 0 |
| H&M deterministic 10,000 | 10,000 | 236,028 | 7,000 / 1,500 / 1,500 | 0 |
| Citi Bike 2016-02 | 7,483 | 560,874 | 5,238 / 1,122 / 1,123 | 0 |

물리적으로 확인된 canonical 디렉터리는 다음 8개이다.

- `data/cof_seqgen_saf/canonical/amlsim`
- `data/cof_seqgen_saf/canonical/sparkov`
- `data/cof_seqgen_saf/canonical/berka`
- `data/cof_seqgen_saf/canonical/berka_nested_900`
- `data/cof_seqgen_saf/canonical/hm`
- `data/cof_seqgen_saf/canonical/citi_bike`
- `data/cof_seqgen_saf/canonical/controlled_coupling_joint_semimarkov_v2b_kappa_0_00`
- `data/cof_seqgen_saf/canonical/controlled_coupling_joint_semimarkov_v2b_kappa_1_00`

통과한 공통 gate는 entity leakage, timestamp parsing, nonnegative gap, entity별 정확히 하나의 missing first gap, train-only vocabulary, sequence report, source manifest, oracle identity, mark context이다.

### 3.2 고정된 재현성 hash

| Artifact | SHA-256 |
|---|---|
| AMLSim canonical manifest | `253ff73ebf06cb80617007b780b6909d6a4688abf4c0ae89b76e4e6cf791df7c` |
| Sparkov canonical manifest | `5ec8d547e72823b90a989a9bb596b2aa96be2b50673518a85b17adf7bd731cc0` |
| Berka full canonical manifest | `76a568f0db09b5fa5b951bb33e0d04bd20870b0df0e2d4b7e26851fe74244237` |
| Berka nested-900 canonical manifest | `ab2a4964a250d6bd38166dc9d93be0a7fba35233f6e09068bbb14ad666783253` |
| H&M canonical manifest | `add788132b5bb487299045b37c4081c148397f14fd40d9bb0f9f3a4d2eda3cb2` |
| Citi Bike canonical manifest | `5b7dca8c83b52c644001f7a6388116c469e438f7a1d1968cee8a17037c467b93` |
| Controlled κ=0 canonical manifest | `d10d58ff0e070ea818f6487dd16d64b6f18cc16a8df21b5a20778c6ce6484ec6` |
| Controlled κ=1 canonical manifest | `247cc11a636295794b55363072924e5e8aba8007c60e5d61b367a9f8ed0ff43a` |
| Global materialization summary | `91998b4e383b7b5c85fa598402b9e9c0c8a0610f16adc11e0a3bf9fa81d4a11e` |

### 3.3 “수집 완료”가 의미하지 않는 것

- H&M 10,000명은 결정론적 sample이지만 Seq2Synth가 사용한 비공개 customer ID/seed와 동일하다고 주장할 수 없다.
- Citi Bike는 schema-aligned이며 Seq2Synth 보고 행·trajectory 수와 identity-matched하지 않는다.
- Berka nested-900은 deterministic comparison view이며 TabDiT의 비공개 split과 동일하지 않다.
- 데이터 완료는 baseline 학습이나 모델 성능 검증 완료를 의미하지 않는다.

## 4. 현재 구현된 코드

### 4.1 데이터 계약·adapter·tensorizer

- `data/cof_seqgen_saf_contract.py`: canonical schema, split 및 integrity contract
- `data/cof_seqgen_saf_adapters.py`: dataset-specific adapter/materialization 지원
- `data/cof_seqgen_saf_tensorizer.py`: train-only category/numeric codec, support fit, static/event roles, PAD/UNK/missing 구분, encode/decode, sealed split loader
- `scripts/acquire_cof_seqgen_saf.py`: source acquisition
- `scripts/materialize_cof_seqgen_saf.py`: canonical materialization
- `scripts/verify_cof_seqgen_saf_data.py`: source와 canonical artifact 독립 검증

### 4.2 모델

- `models/cof_seqgen_saf.py`
- 후보: SAF-C0, SAF-U0, SAF-O0, SAF-U1, SAF-O1
- gap decoder: hurdle-lognormal, unordered categorical, ordered hazard
- strict-past shifted GRU
- static context를 모든 GRU layer initial state로 전달
- current gap을 mark head에 FiLM interaction으로 routing하는 v3
- current gap/current mark를 value head에 routing
- first gap NLL masking, support-safe sampling, numerical guard
- auxiliary event heads 구현

현재 factorization은 다음과 같다.

1. `gap_t ~ p(gap_t | strict_past, static)`
2. `mark_t ~ p(mark_t | strict_past, static, current_gap_t)`
3. `value_t ~ p(value_t | strict_past, static, current_gap_t, current_mark_t)`

### 4.3 학습·검증·분석 runner

- `experiments/cof_seqgen_saf_training.py`: deterministic training, checkpoint latest/best, resume 및 implementation-version 검사
- `scripts/run_cof_seqgen_saf_training.py`: 단일 run CLI
- `scripts/run_cof_seqgen_saf_training_matrix.py`: candidate×seed matrix 실행
- `experiments/cof_seqgen_saf_validation.py`: validation generation과 공통 metric 평가
- `scripts/validate_cof_seqgen_saf.py`: 단일 validation CLI
- `scripts/run_cof_seqgen_saf_validation_matrix.py`: seed matrix validation
- `scripts/analyze_cof_seqgen_saf_dgp_replication.py`: registered contrast 분석
- `benchmarks/cof_seqgen_saf_metrics.py`: temporal, relational, marginal, utility, privacy 지표

### 4.4 baseline integration

- `generators/cof_seqgen_saf_baselines.py`: local/flattened control
- `generators/cof_seqgen_saf_external_baselines.py`: external baseline adapters
- `experiments/cof_seqgen_saf_external_validation.py`: 공통 canonical projection과 평가
- `scripts/validate_cof_seqgen_saf_external.py`: external baseline CLI
- `configs/benchmark_v2/cof_seqgen_saf_baseline_lock.yaml`: 버전·commit·역할 lock

### 4.5 source를 직접 확인한 구현 경계

단순히 파일명이 존재하는지만 확인한 것이 아니라 다음 실행 계약이 실제 코드에 들어 있는지 확인했다.

| 코드 경계 | 실제 구현 상태 |
|---|---|
| split sealing | `load_canonical_dataset(..., allowed_splits=("train", "validation"))`가 Parquet row body에서 test를 제외하고 full split hash만 보존 |
| train-only representation | receiver/aux/static vocabulary, numeric normalization, gap support가 train entity event로만 fit |
| strict-past history | shifted GRU output을 사용해 position t가 현재 event가 아니라 t 이전 event만 보도록 구현 |
| first event semantics | 첫 gap은 NaN이며 gap NLL에서 제외하지만 첫 mark/value는 이후 history에 유지 |
| static route bug fix | static projection이 first output뿐 아니라 모든 GRU layer initial state에 들어감 |
| support-aligned decoding | U/O decoder가 fitted train gap support state를 sample하고 representative로 decode |
| interaction route | v3 U1/O1은 gap embedding으로 history hidden의 FiLM scale/shift를 생성 |
| teacher forcing | 학습 시 observed gap→mark, observed gap/mark→value; 생성 시 generated values만 순차 routing |
| auxiliary heads | auxiliary categoricals를 먼저 생성하고 그것을 auxiliary numeric head에 routing |
| sequence length fairness | 모든 generator가 train-only shared external length plan을 사용; learned termination은 미구현 |
| deterministic runner | Python/NumPy/Torch/CUDA seed, deterministic algorithms, atomic latest/best checkpoint, loader RNG state 저장 |
| unsafe resume prevention | checkpoint schema와 `model_implementation_version`이 현재 source와 다르면 fail-closed |
| validation isolation | checkpoint seal/version 검사 후 train+validation만 load, validation 비교 뒤 `test_accessed=false` 기록 |

현재 코드 기준으로 canonical tensorization부터 train/checkpoint/sample/validation metric까지 연결되는 path는 구현되어 있다. 미구현인 핵심은 learned sequence termination이며, 연구 진행상 더 직접적인 미완료 항목은 v3 전용 실행 config와 v3 CPU/GPU 결과다.

## 5. 모델 계보와 결과 유효성

### 5.1 static-first-only 초기 pilot: 논문 분석에서 제외

초기 `CausalHistoryEncoder`는 static context를 첫 event output에만 더하고 GRU initial state에는 넣지 않았다. 따라서 event 1 이후 direct static route가 사라졌다.

무효화 marker:

- `artifacts/cof_seqgen_saf/development/STATIC_FIRST_ONLY_PILOT_INVALIDATION_2026_08_27.json`
- SHA-256: `fc83c733e565edec3721363804b686a55d10d2f92ecc788ccd5c93082a3d8277`
- decision: `PILOT_ONLY_EXCLUDED_FROM_PAPER_ANALYSIS`

다음 artifact root들은 구현 pipeline 진단 외 논문 성능 분석에 사용하면 안 된다.

- `gpu_dgp_k0`
- `gpu_dgp_k0_replication`
- `gpu_dgp_k0_replication_validation`
- `gpu_dgp_k1`
- `gpu_dgp_k1_replication`
- `gpu_dgp_k1_replication_validation`

### 5.2 saf-static-gru-state-v2: 유효한 개발 실험

v2는 static context를 GRU initial state에 전달해 위 결함을 수정했다. 현재까지 실제 학습·검증된 가장 최신 구현은 v2이다.

#### CPU gate

- artifact: `artifacts/cof_seqgen_saf/development/cpu_static_v2_gate.json`
- SHA-256: `b5f373f4867ca82fa992be9aaf03b7f609e8e6c4b5dacc0250e6182132958dfd`
- initial train loss: 10.9579977
- final train loss: 7.4057059
- relative decrease: 32.42%
- best validation loss: 8.6604428 at epoch 9
- 2회 반복의 best epoch, history, model state, RNG state가 exact match
- model state content SHA-256: `678972852dc58ea2ec19c1bb8cd278776c7669933d98841d9ffd0de02bf40b36`
- CPU end-to-end validation report SHA-256: `3e309af9f679f4833edb613e9860c82977f39d6422753b6434f036e5de675f03`
- `test_accessed=false`

#### Controlled κ=1 GPU training

- root: `artifacts/cof_seqgen_saf/development/gpu_dgp_k1_static_v2`
- matrix manifest SHA-256: `391b75f01d74b1ba7d53c507f696288077155d175703c4c4a280432587c5cead`
- candidates: C0/U0/O0/U1/O1
- seed IDs: 20260826–20260830; 이 값들은 seed identifier이며 실행 날짜를 뜻하지 않는다.
- 총 25/25 학습 record 완료
- 모든 record의 `model_implementation_version=saf-static-gru-state-v2`
- 모든 record의 `test_accessed=false`

#### Controlled κ=1 validation

- root: `artifacts/cof_seqgen_saf/development/gpu_dgp_k1_static_v2_validation`
- matrix manifest SHA-256: `f3dbdde5bd62226b37188f265e4b8532ceb33d4067f6f217c1a6a9777e435cf4`
- 총 5/5 seed report 완료
- shared generation plan SHA-256: `505e831ad8e5df4a6efe746eb1b2c77fb7efe78c167443d3aa55f2fdca8c7d74`
- 모든 report의 `test_accessed=false`

### 5.3 saf-interactive-gap-route-v3: 현재 소스, 아직 결과 없음

v2의 U1/O1 additive route 결과가 H3를 안정적으로 지지하지 않아 history representation에 current-gap FiLM scale/shift를 적용하는 interaction route로 변경했다.

현재 v3 상태:

- 모델 소스: 있음
- 단위/통합 테스트: 통과
- 전용 config: 없음
- CPU overfit/determinism gate: 없음
- GPU checkpoint: 없음
- generated validation sample: 없음
- κ=0/κ=1 contrast: 없음
- real-data 결과: 없음

그러므로 아래 v2 결과는 v3의 성능 결과가 아니다. v3 설계 동기를 제공하는 개발 증거로만 사용한다.

## 6. static-v2 κ=1 개발 결과

아래 값은 5 seeds의 validation mean ± sample SD이다. 작을수록 좋다.

| Candidate | gap-conditioned mark TV | short-gap repeat L1 | gap-repeat MI error | gap KS | gap scaled W1 |
|---|---:|---:|---:|---:|---:|
| SAF-C0 | 0.009779 ± 0.000688 | 0.008094 ± 0.000840 | 0.0001851 ± 0.0000080 | 0.099732 ± 0.001174 | 0.253375 ± 0.001011 |
| SAF-U0 | 0.013639 ± 0.001621 | 0.008004 ± 0.001377 | 0.0001965 ± 0.0000103 | 0.020213 ± 0.000253 | 0.050664 ± 0.000067 |
| SAF-O0 | 0.010249 ± 0.000782 | 0.007100 ± 0.000744 | 0.0001688 ± 0.0000155 | 0.020610 ± 0.000648 | 0.050821 ± 0.000148 |
| SAF-U1 additive | 0.011159 ± 0.001023 | 0.006874 ± 0.001350 | 0.0001834 ± 0.0000143 | 0.021173 ± 0.000380 | 0.050803 ± 0.000284 |
| SAF-O1 additive | 0.011261 ± 0.000346 | 0.008043 ± 0.000865 | 0.0001972 ± 0.0000049 | 0.020529 ± 0.000500 | 0.050815 ± 0.000120 |
| Empirical trajectory control, 1 run | 0.010990 | 0.002630 | 0.0000669 | 0.006404 | 0.008898 |

### 6.1 등록 가설별 해석

차이는 `control error - treatment error`로 계산했으므로 양수면 treatment가 개선된 것이다.

#### H1: support alignment

C0 → U0:

- gap KS 개선 평균: +0.079518, 5/5 seeds 양수
- gap scaled W1 개선 평균: +0.202711, 5/5 seeds 양수
- zero-rate error: 0.000074 → 0

**판정: κ=1 development에서 강한 지지.** 다만 κ=0과 v3 replication 이전이므로 최종 논문 결론은 아니다.

#### H2: ordered hazard

U0 → O0:

- gap-conditioned mark TV 개선: +0.003390, 5/5
- short-gap repeat L1 개선: +0.000904, 4/5
- gap-repeat MI error 개선: +0.0000277, 4/5
- gap KS와 W1은 소폭 악화

**판정: 부분 지지.** ordered decoder가 dependency metric에는 유리했지만 모든 gap marginal metric을 지배하지 않았다.

#### H3: explicit gap→mark routing

U0 → U1 additive:

- conditioned TV 개선 +0.002481, 4/5
- repeat L1 개선 +0.001130, 4/5
- MI error 개선 +0.0000132, 5/5

O0 → O1 additive:

- conditioned TV -0.001011, 0/5
- repeat L1 -0.000943, 1/5
- MI error -0.0000284, 0/5

**판정: 혼합/실패.** 특히 ordered branch에서 additive route가 일관되게 나빠 H3를 주장할 수 없다. 이 결과가 v3 interaction route를 도입한 직접적인 이유다.

#### H4: 최신 sequential baseline 대비 우위

**판정: 아직 검증 불가.** CPAR 한 번의 개발 실행 외 tuned multi-seed primary baseline 결과가 없고 v3도 학습되지 않았다.

#### H5: marginal fidelity·utility·privacy 비열등

모든 SAF 후보에서 `all_calibrated_marginal_structural_endpoints_pass`는 0/5였다. 그러나 empirical train-trajectory resampler도 같은 전체 margin을 통과하지 못했다. 현재 q95 margin은 DGP의 finite train-vs-validation variation보다 지나치게 좁거나 endpoint 구성과 맞지 않을 가능성이 있다.

**판정: 성공도 실패도 아닌 비결정적 상태.** candidate output을 본 뒤 DGP margin을 소급 조정하면 안 된다. 다음 단계에서는:

- 기존 DGP margin 결과는 그대로 보존하고 calibration limitation으로 보고한다.
- real data 실행 전에는 validation output을 보지 않고 train-only resampling design과 endpoint 정의를 재감사한다.
- utility와 privacy는 개별 paired metric으로 함께 보고하되, 전체 H5 pass 하나로 과도하게 축약하지 않는다.

참고로 v2의 trajectory exact-copy rate는 모든 후보에서 0이고 transition n-gram exposure mean은 약 0.825–0.827이다. empirical control은 각각 1.0과 1.0이므로 copy detector는 의도대로 반응한다. TSTR ratio는 gap 약 0.962–0.999, repeat 약 1.000–1.001 범위이나 superiority 근거로 해석하지 않는다.

## 7. 외부 baseline 상태

### 7.1 dependency lock

Main environment:

- Python 3.10
- PyTorch 2.1.2+cu121
- NumPy 1.26.4
- pandas 2.3.3
- SDV 1.38.0
- REaLTabFormer 0.2.4
- transformers 4.46.3
- lock 당시 `pip check` clean

Baseline별 판정:

| Baseline | 상태 | 현재 연구상 역할 |
|---|---|---|
| CPAR | wrapper 및 full DGP κ=1 dev run 완료 | primary sequential dev comparator |
| REaLTabFormer | wrapper, compatibility shim, CPU smoke 존재 | primary relational baseline; full run 미실행 |
| TabularARGN | source commit pin 완료 | Python ≥3.11, torch 2.11 격리 환경 필요 |
| TabDiT | commit pin 완료 | upstream generator/train/sample code 부재, reported-only |
| Empirical sequence sampler | 실행 완료 | memorization negative/control baseline |
| CTGAN/TVAE/GaussianCopula | wrapper/control 경로 구현 | flattened marginal control, primary temporal ranking 아님 |

TabPFN은 generator ranking baseline이 아니다.

### 7.2 CPAR 개발 실행 결과

- artifact: `artifacts/cof_seqgen_saf/development/external_dgp_k1/cpar_epochs20/external_validation.json`
- report SHA-256: `34bfce99903d155a4d6350e6162401e4cc964d6174a13d64d13ae8b2cddb9b93`
- fit: 31,951 train entities, 766,935 events
- fit time: 3,292.71 s
- sample time: 560.04 s
- model SHA-256: `1ff1d57e66ac1524e5fe7764d3c2d624bb2ba2dd62f42019c7a3787394da345a`
- generated SHA-256: `9520ac1f1d71ef624fed8dc8a325131efbbf5dc69f3949f95e57ff2c3c94d263`
- shared generation plan: static-v2와 동일
- conditioned mark TV: 0.031790
- short-gap repeat L1: 0.301943
- gap KS: 0.350005
- gap scaled W1: 0.369290
- trajectory copy: 0
- transition n-gram exposure: 0.804708
- `claim_status=DEVELOPMENT_VALIDATION_ONLY`
- `test_accessed=false`

이 단일 20-epoch 결과에서 CPAR가 v2 SAF보다 temporal metric이 나쁘지만, tuning과 multi-seed가 없는 개발 run이므로 “SAF가 최신 baseline보다 우수”라는 결론은 허용되지 않는다.

## 8. 현재 테스트 상태

2026-08-28 감사 시점에 현 v3 소스 대상으로 SAF test suite를 재실행했다.

```text
56 passed, 1 skipped, 5 warnings in 4.70s
```

- skip: explicit external dependency smoke only
- warning: SDV metadata deprecation/order warning
- failure: 0

이 결과는 구현 계약과 CPU smoke가 깨지지 않았다는 뜻이지 v3가 학습되거나 연구 가설이 성립했다는 뜻은 아니다.

## 9. 기존 MD 문서의 정확한 읽는 법

### `research_implementation_execution_overview_2026_08_26.md`

- legacy CoF부터 v2.5, v2.8, joint-discrete v3, CCMTPP C1, HCMTTPP H1, CoF-ZDH-v3 D1까지의 이전 연구 계보와 결과를 상세히 기록한다.
- 새 SAF contribution의 출발점인 “CoF의 coherence 장점은 보였지만 marginal fidelity/combined 우월성은 없었다”는 근거를 여기서 가져온다.
- 이 문서의 최종 상태는 CoF-ZDH-v3 D1 `SPECIFIED_NOT_IMPLEMENTED`이며, 그 뒤 생성된 `cof_seqgen_saf` source/config/artifact는 포함하지 않는다.
- 따라서 현재 SAF κ=1 진행률을 이 문서의 “D1 미구현” 문장으로 판단하면 안 된다.

### `cof_seqgen_saf_data_collection_report_2026_08_26.md`

- acquisition과 materialization 결과는 현재도 유효하다.
- “baseline/model/training을 실행하지 않았다”는 문장은 **2026-08-26 데이터 보고서 작성 시점**의 경계 설명이다.
- 그 뒤 v2 학습과 baseline 개발 실행이 수행됐으므로 현재 전체 상태 설명으로 읽으면 안 된다.

### `cof_seqgen_saf_preexecution_implementation_report_2026_08_26.md`

- metric audit 및 당시 설계 기록으로 유효하다.
- “tensorizer, auxiliary heads, runner, wrapper가 남았다”는 remaining boundary는 이후 모두 구현되었으므로 현재 상태로는 오래됐다.
- 당시 43 tests도 현재 56 pass/1 skip으로 대체되었다.

### `preregistered_cof_seqgen_saf_source_protocol.md`

- 가설, split 봉인, 비교 규칙의 사전등록 근거다.
- 실행 현황 문서가 아니므로 완료 여부를 여기서 추론하면 안 된다.

### 이 문서

- 2026-08-28 현재의 구현·실행·중단·복구 상태를 설명한다.
- 위 역사 문서의 당시 사실과 사전등록 내용은 보존하고, 현재 runtime status만 갱신한다.

## 10. 남은 blocker와 정확한 재개점

### Blocker A: v3 전용 config 부재

먼저 아래와 같은 v3 전용 config를 새 output root와 함께 만들어야 한다. 현재 이 파일들은 존재하지 않는다.

- `cof_seqgen_saf_cpu_overfit_interactive_v3.yaml`
- `cof_seqgen_saf_cpu_validation_interactive_v3.yaml`
- `cof_seqgen_saf_gpu_dgp_k1_interactive_v3_matrix.yaml`
- `cof_seqgen_saf_gpu_dgp_k1_interactive_v3_validation_matrix.yaml`
- `cof_seqgen_saf_gpu_dgp_k0_interactive_v3_matrix.yaml`
- `cof_seqgen_saf_gpu_dgp_k0_interactive_v3_validation_matrix.yaml`
- `cof_seqgen_saf_dgp_interactive_v3_analysis.yaml`

기존 `static_v2` config/output root를 덮어쓰면 안 된다.

### Blocker B: v3 CPU gate 미통과

v3에 대해 다음을 새로 확인해야 한다.

- tiny E2E train → checkpoint → reload → validation generation
- two-run bitwise determinism
- overfit loss decrease
- FiLM parameters의 nonzero gradient
- routed/unrouted candidate separation
- exact support preservation
- `test_accessed=false`

### Blocker C: mechanism identification의 짝이 없음

현재 유효한 v2 결과는 κ=1뿐이다. κ=0에 보이는 기존 결과는 static-first-only invalidation 대상이다. v3로 κ=0과 κ=1을 같은 seed, candidate, generation plan 규칙으로 모두 새로 실행해야 interaction이 실제 coupling mechanism을 회복하는지 말할 수 있다.

### Blocker D: H4 baseline matrix 미완료

CPAR 단일 dev run만 있다. REaLTabFormer full run, TabularARGN isolated runtime, 공정한 tuning budget, multi-seed 또는 등록된 stochastic replication 규칙이 필요하다. TabDiT는 실행 코드가 없는 한 reported-only를 유지한다.

### Blocker E: real-data 실험 미시작

AMLSim, Sparkov, Berka, H&M, Citi Bike는 model-ready canonical data지만 어떤 v3 candidate나 primary baseline도 이 데이터에 fit되지 않았다.

### Blocker F: 연구 코드가 아직 version control에 고정되지 않음

현재 `git status`에서 SAF 구현·config·test·문서의 다수가 untracked(`??`)이다. 따라서 artifact hash는 있어도 소스 snapshot의 commit ID가 없다. GPU 재실행 전 다음을 해야 한다.

특히 static-v2 checkpoint/report 안에는 `saf-static-gru-state-v2` version과 model state가 기록되어 있지만, worktree의 `models/cof_seqgen_saf.py`는 이미 v3로 바뀌었다. static-v2 당시 Python source가 별도 commit으로 보존되지 않았기 때문에 현재 코드로 v2 checkpoint를 재생성/재검증할 수 없다. 저장된 v2 validation 결과 자체는 report/checkpoint/manifest SHA로 완결되어 있지만, 논문 재현성 측면에서는 이 source snapshot 부재를 반드시 기록해야 한다.

- unrelated 사용자 변경을 건드리지 않는다.
- SAF 관련 파일만 명시적으로 review한다.
- 데이터/대형 artifact를 commit하지 않는다.
- 실행에 사용한 source/config를 commit 또는 immutable source snapshot hash로 고정한다.

## 11. 권장 재개 순서

### Step 1. v3 experiment family materialization

v2 config를 복제하되 다음을 반드시 변경한다.

- `model_implementation_version: saf-interactive-gap-route-v3`
- 새로운 output root `*_interactive_v3`
- v2 checkpoint resume 금지
- C0/U0/O0의 route-off 구조는 대조군으로 유지
- U1/O1만 FiLM interaction route 활성화
- seed와 shared length-plan 규칙은 v2와 동일하게 유지

### Step 2. v3 CPU gate

전용 config 생성 후:

```bash
<COFSEQ_PYTHON> -m scripts.run_cof_seqgen_saf_training \
  --config configs/benchmark_v2/cof_seqgen_saf_cpu_overfit_interactive_v3.yaml
```

CPU validation과 repeat determinism gate가 모두 통과할 때만 GPU로 넘어간다.

### Step 3. v3 Controlled κ=1

```bash
<COFSEQ_PYTHON> -m scripts.run_cof_seqgen_saf_training_matrix \
  --config configs/benchmark_v2/cof_seqgen_saf_gpu_dgp_k1_interactive_v3_matrix.yaml

<COFSEQ_PYTHON> -m scripts.run_cof_seqgen_saf_validation_matrix \
  --config configs/benchmark_v2/cof_seqgen_saf_gpu_dgp_k1_interactive_v3_validation_matrix.yaml
```

먼저 O0→O1 및 U0→U1에서 interaction route가 v2의 H3 failure를 고치는지 확인한다. validation development 결과이므로 test는 계속 봉인한다.

### Step 4. v3 Controlled κ=0 paired mechanism run

동일 seed/candidate budget으로 κ=0을 실행한다. κ=0에서는 불필요한 gap route가 dependency metric을 인위적으로 개선하지 않는지 확인한다.

### Step 5. 사전등록 contrast 분석

```bash
<COFSEQ_PYTHON> -m scripts.analyze_cof_seqgen_saf_dgp_replication \
  --config configs/benchmark_v2/cof_seqgen_saf_dgp_interactive_v3_analysis.yaml
```

최소 판정 항목:

- H1: C0 vs U0
- H2: U0 vs O0, U1 vs O1
- H3: U0 vs U1, O0 vs O1
- κ interaction: H3 effect가 κ=1에서 κ=0보다 큰가
- seed-paired effect와 uncertainty
- marginal/utility/privacy endpoint의 개별 결과

### Step 6. baseline 및 real-data development

v3 mechanism gate를 통과한 뒤에만 계산량이 큰 순서로 진행한다.

1. CPAR registered/tuned budget과 replication
2. REaLTabFormer full run
3. TabularARGN isolated environment 구축 후 full run
4. AMLSim, Sparkov
5. Berka, H&M, Citi Bike/Seq2Synth-aligned views
6. frozen validation 기준으로 architecture와 hyperparameter 선택
7. 모든 선택을 고정한 뒤 마지막에 held-out test 1회

## 12. 다음 작업자가 하지 말아야 할 것

- `gpu_dgp_k0`, `gpu_dgp_k1`, `*_replication` 초기 결과를 paper result에 합치지 않는다.
- static-v2 checkpoint를 current v3 source에 강제로 load하지 않는다.
- static-v2 output directory를 v3가 덮어쓰지 않는다.
- CPAR 한 번의 결과로 SOTA를 주장하지 않는다.
- H5 margin을 이미 본 candidate 결과에 맞춰 소급 조정하지 않는다.
- Seq2Synth와 H&M/Citi Bike identity-matched라고 표현하지 않는다.
- validation을 test라고 부르지 않는다.
- held-out test를 model/debug/tuning 과정에서 열지 않는다.

## 13. 현재 단계의 paper-claim 가능 범위

현재 증거로 허용되는 문장:

> Controlled κ=1 development validation에서 train-support-aligned discrete gap decoder는 continuous hurdle-lognormal 대비 gap KS와 scaled W1을 5/5 seeds에서 개선했다. Ordered hazard는 일부 temporal-relational dependency metric에서 unordered decoder보다 개선됐지만, additive gap-to-mark routing은 ordered branch에서 H3를 회복하지 못했다. 이 failure analysis를 바탕으로 history×current-gap FiLM interaction을 갖는 v3를 구현했으며, v3의 CPU/GPU 실험은 아직 수행되지 않았다.

현재 증거로 허용되지 않는 문장:

- CoFSeqGen-SAF가 최신 모델보다 우수하다.
- explicit gap routing이 dependency를 복원했다.
- marginal fidelity·utility·privacy 비열등이 입증됐다.
- real sequential dataset에서 generalization이 확인됐다.
- 최종 논문 모델 실험이 완료됐다.

## 14. 복구 완료의 정의

다음 조건을 모두 충족해야 “중단된 지점에서 완전히 복구되어 다음 연구 단계로 넘어갔다”고 판정한다.

- [ ] v3 전용 config와 output root 생성
- [ ] v3 CPU E2E/overfit/determinism gate 통과
- [ ] v3 κ=1 5-candidate × registered seeds 학습·validation 완료
- [ ] v3 κ=0 paired 학습·validation 완료
- [ ] v3 H1/H2/H3 및 κ interaction aggregate 생성
- [ ] test access 0 재검증
- [ ] source/config immutable snapshot 고정
- [ ] H3 성공/실패에 따른 다음 model decision 기록
- [ ] primary baseline 실행 예산과 real-data 실행 순서 고정

현재 체크 가능한 완료 항목은 데이터 수집·canonical materialization, 공통 구현, metric audit, static-v2 κ=1 개발 실험, CPAR 개발 pipeline뿐이다. **실제 재개점은 데이터 다운로드가 아니라 v3 전용 experiment config와 CPU gate이다.**
