# CoF-SeqGen 연구 구현·실행·데이터·결과 종합 기록

작성 기준일: 2026-08-26 (Asia/Seoul)  
저장소: `<REPO_ROOT>`  
branch: `benchmark-v2-redesign`  
문서 작성 시 HEAD: `7abe12b68780673666deef951dc7e1741f74f6af` (`docs: preregister CoF-ZDH-v3 D1`)

## 1. 가장 먼저 알아야 할 결론

맞다. 지금까지 benchmark v2 재설계, v2.5 full experiment, v2.6–v2.8 validation candidate selection, AMLSim/Sparkov 외부 검증, CoF-SeqGen v3, CCMTPP, HCMTTPP H1, 그리고 현재의 CoF-ZDH-v3 D1 설계는 모두 위 저장소에서 구현·실행·보존되어 있다.

다만 “가장 최근 모델”은 다음 세 의미를 구분해야 정확하다.

| 구분 | 가장 최근 대상 | 상태 |
|---|---|---|
| 실제로 끝까지 학습·validation sample 생성·평가된 최신 모델 | `CoF-HCMTTPP-v2 H1`, Sparkov `attempt_003` | `COMPLETE`였으나 scientific gate `FAIL`; family 영구 종료 |
| 일부 데이터에서 학습 중 실패한 동일 최신 모델 | `CoF-HCMTTPP-v2 H1`, AMLSim `attempt_003` | step 14,000까지 finite, 14,001–14,099 사이 최초 non-finite; `FAILED` |
| 현재 HEAD의 가장 최신 연구 후보 | `CoF-ZDH-v3 D1` | 문서/config로만 사전등록됨; 모델·runner 미구현, 실행 0회 |

따라서 현재 연구 상태를 한 줄로 요약하면 다음과 같다.

> 기존 non-v3 CoF는 외부 validation/internal test에서 **coherence 강점만** 재현했고 전체 fidelity/combined 우월성은 지지되지 않았다. 이를 고치기 위한 v3 joint discrete, CCMTPP C1, HCMTTPP H1은 각각 사전등록 gate에서 종료되었으며, 현재는 H1의 수치 실패 구조를 제거한 `CoF-ZDH-v3 D1`을 **source-only 설계한 상태**이다.

이 문서는 저장된 config, source, terminal marker, evaluation/metrics, aggregate, forensic report, Git history를 읽어 작성했다. 기존 runtime artifact, checkpoint, frozen data 및 사용자 변경은 수정하지 않았다.

## 2. 현재 저장소 상태와 provenance

### 2.1 Git 상태

- branch: `benchmark-v2-redesign`
- HEAD: `7abe12b68780673666deef951dc7e1741f74f6af`
- legacy tag: `legacy-kappa-v1` → `b72cd99207cc60e1b555139361d25bf2aed4878d`
- v2.4 gate-pass tag: `benchmark-v2.4-gate-pass` → `35ec654061f99e8e5ccd54381d40329546ed2eb7`

문서 작성 시 worktree에는 이 연구 정리와 무관한 사용자 소유 변경이 이미 있었다.

```text
 D images/tabdiff_demo.gif
 D images/tabdiff_demo.mp4
 D images/tabdiff_flowchart.jpg
?? (
?? -maxdepth
?? -print
?? -type
?? docs/benchmark_v2/forensic_cof_ccmtpp_v1_c1_gate.md
```

이 파일들은 이 문서 작성에서 수정·stage·삭제하지 않았다.

### 2.2 연구 계보

```text
legacy CoF-SeqGen / TabDiff fork
  └─ benchmark v2.0–v2.4: DGP·gate·AUROC calibration 확립
      └─ v2.5: 13 generators × 5 seeds controlled full experiment
          ├─ v2.6: train/validation-only candidate selection
          ├─ v2.7: single-factor evaluation-only selection
          ├─ v2.8: CTGAN/TVAE 통과, CoF receiver guard 실패
          └─ CoF-SeqGen v3: joint discrete redesign → architecture limitation

external protocol v1: AMLSim + Sparkov frozen sequences
  ├─ empirical IID / CTGAN / TVAE / frozen non-v3 CoF validation
  ├─ confirmatory internal-test 8 jobs
  └─ 결론: CoF의 coherence 우위만 descriptive하게 지지

CoF model-level redesign lineage
  ├─ cof_ccmtpp_v1 C1: causal MTPP + continuous gap → Y0 gap gate 실패
  ├─ cof_hcmttpp_v2 H1: hurdle + RQS + conditional tail
  │   ├─ attempt_001: RQS inverse 구현 오류
  │   ├─ attempt_002: spawn ownership hand-off 오류
  │   └─ attempt_003: AMLSim numerical failure, Sparkov scientific gate failure
  └─ cof_zdh_v3 D1: bounded discrete-hazard gap decoder
      └─ 현재는 specification/config만 존재; 구현·실행 안 됨
```

### 2.3 주요 commit 흐름

| 단계 | 대표 commit | 의미 |
|---|---|---|
| legacy snapshot | `0066ad3` | benchmark redesign 전 source snapshot |
| benchmark v2.4 | `35ec654` | CPU gate와 smoke 완료 보고 |
| v2.5 preregistration | `4dfe83c` | 13 generator, 5-seed full protocol 고정 |
| v2.5 full runner | `225083a`, `4386754`, `99a445f` | frozen orchestration, watchdog, continuation scheduler |
| v2.5 evaluator correction | `58f0dcc` | support diagnostic을 validity hard guard에서 분리 |
| v2.5 forensic | `5502f84` | learned row-marginal failure 분석 |
| v2.6 selection | `6e59ef2` → `f02004a` | train/validation-only finite candidate selection |
| v2.7/v2.8 | `7c3a5a0` → `3b4770f` | evaluation-only single-factor 후보와 aggregate |
| CoF v3 | `11569ba` → `91acaf9` | joint discrete redesign·runner·실행 forensic |
| external data | `0564252` → `29cbee2` | feasibility audit와 frozen materializer |
| external validation | `fceaa8e` → `9458ada` | validation, internal-test, aggregate/forensic |
| CCMTPP C1 | `610e48c` → `8957c50` | causal marked TPP 구현·실행·gap forensic |
| HCMTTPP H1 | `4913097` → `a72cf11` | hurdle-RQS 모델, 세 execution attempt, 종료 forensic |
| ZDH D1 | `7abe12b` | 다음 family의 source-only 설계 고정 |

## 3. 저장소 전체 구조

### 3.1 디렉터리 역할

```text
cof-seqgen-0707-2119-Version3-complete/
├── benchmarks/        controlled DGP, temporal coupling, binning, gate/audit
├── configs/           benchmark·model·runner·external protocol YAML
├── data/              raw-derived data, frozen NPZ, transform/manifests, adapters
├── eval/              fidelity/coherence/association/guard/selection/statistics
├── experiments/       runner, authorization, watchdog, artifact orchestration
├── generators/        baseline adapters, SamplingPlan, model execution backends
├── models/            CoF/non-v3, v3, CCMTPP, HCMTTPP model code
├── scripts/           CLI entrypoints, preparation, calibration, aggregate, forensic
├── tests/             unit/contract/regression/integration tests
├── artifacts/         append-only runtime/checkpoints/samples/evaluation/logs
├── docs/benchmark_v2/ preregistration, contracts, corrections, forensic conclusions
├── logs/              초기 legacy/kappa/sweep 실행 로그
└── results/           초기 AMLSim/Sparkov/legacy result CSV
```

`REPO_MAP.md`는 초기 TabDiff fork를 설명하는 오래된 지도다. 현재 연구의 authoritative map은 실제 `configs/benchmark_v2`, `models`, `generators`, `experiments`, `eval`, `scripts`, `artifacts`, `docs/benchmark_v2`를 함께 봐야 한다.

### 3.2 현재 파일 규모

`du -sh` 기준 대략적인 보존 규모다.

| 경로 | 크기 | 파일 수 | 내용 |
|---|---:|---:|---|
| `artifacts/` 전체 | 약 253 GB | 8,933개 이상 | checkpoint, sample, evaluation, logs, manifests |
| `artifacts/external_validation_v1/` | 약 231 GB | 1,147 | AMLSim/Sparkov 4-model validation runtime |
| `artifacts/benchmark_v2_5/` | 약 21 GB | 5,064 | 65-cell controlled full experiment와 attempts |
| `artifacts/benchmark_v2_6/` | 약 470 MB | 233 | candidate trajectories/evaluations/aggregate |
| `artifacts/cof_ccmtpp_v1/` | 약 397 MB | 127 | AMLSim/Sparkov C1 execution |
| `artifacts/benchmark_v3/` | 약 336 MB | 134 | direct/factorized joint candidates |
| `artifacts/cof_hcmttpp_v2/` | 약 284 MB | 135 | H1 attempts 001–003 |
| `artifacts/benchmark_v2_4/` | 약 24 MB | 1,680 | 200-seed AUROC calibration/gate evidence |
| `artifacts/external_confirmatory_internal_test_v1/` | 약 9.4 MB | 84 | 8 confirmatory internal-test jobs |
| `data/` 전체 | 약 358 MB | 223개 이상 | controlled/external frozen arrays와 legacy data |

Runtime이 큰 이유는 source만이 아니라 각 attempt의 checkpoint, generated `sample.npz`, `progress.jsonl`, stdout/stderr, evaluation/diagnostic, checksum/index를 append-only로 보존했기 때문이다.

### 3.3 Frozen non-v3 CoF-SeqGen 핵심 구현

후속 redesign의 기준점이 된 기존 CoF는 `models/cof_seqgen.py`의 `CoFSeqGen`, `models/seq_denoiser.py`의 `SeqDenoiser`, `generators/cof_seqgen_adapter.py`의 `CoFSeqGenAdapter`로 이어진다.

`SeqDenoiser`는 row가 아니라 길이 축을 가진 full-sequence Transformer encoder다.

```text
amount noisy projection
+ gap-bin embedding
+ receiver embedding
+ position embedding
+ diffusion-time embedding
+ sequence-level Y embedding
        ↓
full self-attention Transformer encoder
        ↓
amount head + gap-bin head + receiver head + Y head
```

수치 amount에는 VP cosine Gaussian diffusion을 적용하고, gap/receiver에는 absorbing MASK corruption을 적용한다. discrete mask probability는 최대 0.7로 고정되어 완전 마스킹 불안정을 막는다. Y는 entity/sequence 수준 label을 모든 valid position에 broadcast하며 classifier-free guidance용 null class를 둔다.

기본 loss는 다음 구조다.

```text
L = L_diff + lambda * L_coh

L_diff = amount MSE
       + gap-bin cross entropy
       + receiver cross entropy
       + label BCE
```

`L_coh`를 사용할 때 `models/coherence_teacher.py`의 frozen teacher와 `models/soft_g.py`의 differentiable behavior summary를 통해 amount, timing, receiver head로 gradient가 흐르게 설계되었다. `soft_g`는 velocity, gap, receiver repeat/distinct behavior, amount sum을 causal window에서 요약한다. v2.5 frozen full config에서는 `coherence_lambda=0.0`이므로 controlled comparison의 CoF는 구조 손실 없이 동일 계약으로 평가되었다.

Sampling은 `models/sampler.py`의 DDIM-style 경로와 `generators/sampling_plan.py`의 frozen label/length plan을 사용한다. Adapter가 model fit, checkpoint/resume, fixed step sampling, `SyntheticBatch` 조립을 담당한다.

이 non-v3 경로는 external validation에서도 source commit `99a445f6dc893a8c2240d950de4f92877cc07f8a`와 config hash를 frozen baseline으로 사용했다. 후속 v3/CCMTPP/H1/D1은 이 결과를 덮어쓰지 않고 별도 family/artifact root를 사용한다.

### 3.4 Legacy logs/results

Benchmark redesign 이전의 exploratory 실행도 삭제하지 않고 남아 있다.

- `logs/`: kappa curve, guidance sweep, CTGAN/TVAE baseline, AMLSim/Sparkov teacher/sweep 실행 로그
- `results/`: AMLSim/Sparkov sweep·sequence-teacher CSV, coherence bounds, fidelity/ablation tables
- `figs/`: kappa curves와 LaTeX result tables
- `data/kappa_{0.00,0.30,0.50,0.70,1.00}/`: legacy kappa data bundles

이 legacy 결과는 `legacy-kappa-v1` provenance로 보존하지만, v2.4 이후의 preregistered gate/full/external 결론과 섞어서 재해석하지 않는다. 이 문서의 정량 결론은 주로 `artifacts/benchmark_v2_4` 이후의 append-only evidence를 사용한다.

## 4. 데이터: 무엇이 있고 어떻게 만들어졌는가

## 4.1 공통 sequence representation

주요 frozen sequence NPZ는 다음 배열 계약을 공유한다.

| key | 의미 |
|---|---|
| `x_num` | amount 수치 채널, shape `[N, L, 1]`, `float32` |
| `dt_bin` | gap/Δt discrete bin, shape `[N, L]`, `int64` |
| `x_cat` | receiver categorical code, shape `[N, L, 1]`, `int64` |
| `valid_mask` | valid event mask, shape `[N, L]`, `bool` |
| `y_entity` | sequence-level binary Y |
| `lengths` | 실제 sequence 길이 |
| `entity_ids` | entity-disjoint split audit용 ID; 모델 입력으로 사용하지 않음 |

PAD와 invalid position은 `valid_mask`로 제외한다. 외부 데이터의 receiver vocabulary는 train split에서만 만들어지고 code `0=PAD`, `1=UNK` 계약을 사용한다.

### 4.2 Controlled benchmark v2.5 frozen data

경로:

```text
data/benchmark_v2_5/frozen/joint_semimarkov_v2b/kappa_1.00/
├── train.npz
├── validation.npz
├── test.npz
├── shared_sampling_plan.npz
├── meta.json
└── data_manifest.json
```

| split | sequence 수 | shape |
|---|---:|---|
| train | 31,951 | `x_num (31951,32,1)`, `dt_bin/x_cat/mask (31951,32)` |
| validation | 7,989 | 길이 32 padded sequences |
| test | 7,989 | 길이 32 padded sequences |

주 시나리오는 `joint_semimarkov_v2b`, `kappa=1.0`이다. primary endpoint는 continuous association-recovery error:

```text
abs(delta_joint(real) - delta_joint(synthetic))
```

작을수록 좋다. 4-bin/8-bin support는 diagnostic이며 primary decision을 바꾸지 않는다.

DGP는 `benchmarks/temporal_coupling_v2.py`와 `benchmarks/semi_markov.py`에 있다. Label과 sequence length random stream을 scenario/kappa stream과 분리해 비교 가능성을 유지한다. `joint_semimarkov_v2b`에서는 gap burst/normal 상태와 receiver repeat driver의 alignment를 kappa로 조절한다. Gap은 state별 exponential scale, receiver는 이전 receiver 반복 여부, amount는 별도 normal emission으로 생성한다. Train에서만 raw gap binning을 fit하고 validation/test에 적용한다. DGP와 CPU gate는 GPU를 사용하지 않는 경로로 구현되었다.

Benchmark 기본 규모/구성은 train 31,951, test 7,989, length 16–32, fraud rate 0.05, gap bins 16, receiver categories 64다. `kappa=0`은 gap/receiver alignment가 없는 기준, `kappa=1`의 fraud class는 alignment가 최대인 구조로 해석한다.

### 4.3 AMLSim frozen external data

경로:

```text
data/external_sequence_protocol_v1/frozen/amlsim/attempt_001/
```

프로토콜:

- entity: `SENDER_ACCOUNT_ID`
- 정렬: `TIMESTAMP`, `TX_ID`
- receiver: 수취 계좌
- sequence: 32-event non-overlapping window
- 마지막 window 길이 16–31은 유지, 16 미만은 제외
- Y=1 iff window 안에 laundering/fraud transaction이 하나 이상 존재
- sender 단위 group-stratified 70/15/15 split

| split | entities | sequences | valid rows | Y=1 sequence prevalence | UNK rows |
|---|---:|---:|---:|---:|---:|
| train | 6,657 | 30,349 | 914,756 | 0.0356519 | 0 |
| validation | 1,426 | 6,542 | 197,634 | 0.0353103 | 3,658 |
| internal test | 1,429 | 6,360 | 191,244 | 0.0363208 | 4,061 |

Receiver vocabulary cardinality는 9,656이며 train vocabulary 9,654개 + PAD/UNK다. high-cardinality receiver 때문에 초기 empirical IID receiver-TV 계산의 dense `N×K` 경로가 7,200초 wall cap을 소진했고, 이후 정의는 유지한 채 sparse/vectorized path로 교정했다.

### 4.4 Sparkov frozen external data

경로:

```text
data/external_sequence_protocol_v1/frozen/sparkov/attempt_001/
```

프로토콜:

- entity: `cc_num`
- 정렬: transaction datetime, stable original row order
- receiver: `merchant`
- fraudTrain 안에서 card 단위 group-stratified 70/15/15 split
- 동일 32-event window/Y 규칙
- 공개 `fraudTest`는 materialization, validation, selection에서 차단; 별도 temporal robustness authorization 없이는 접근 금지

| split | entities | sequences | valid rows | Y=1 sequence prevalence | UNK rows |
|---|---:|---:|---:|---:|---:|
| train | 636 | 28,147 | 898,168 | 0.0216719 | 0 |
| validation | 135 | 6,121 | 195,250 | 0.0204215 | 0 |
| internal test | 138 | 6,245 | 199,159 | 0.0212970 | 0 |

Receiver vocabulary cardinality는 695이며 train receiver 693개 + PAD/UNK다.

### 4.5 Frozen bundle 안의 provenance 파일

두 외부 dataset의 `attempt_001`에는 다음이 있다.

```text
train.npz
validation.npz
internal_test.npz
train_transform_state.json
entity_split_manifest.json
window_manifest.json
leakage_audit.json
raw_schema_manifest.json
provenance_manifest.json
summary.json
```

`train_transform_state.json`은 amount transform, gap edges, receiver vocabulary를 train-only로 고정한다. `provenance_manifest.json`은 raw/source/config/split/transform hash를 묶는다. Materialization terminal evidence는 다음 경로다.

```text
artifacts/external_sequence_protocol_v1/materialization/amlsim/attempt_001/
artifacts/external_sequence_protocol_v1/materialization/sparkov/attempt_001/
```

## 5. 구현의 공통 실행 구조

실제 runner 계열은 대체로 다음 call graph를 따른다.

```text
scripts/run_*.py
  └─ authorization / plan / dry-run validation
      └─ experiments/*_runner.py
          ├─ append-only ownership + manifest
          ├─ runner-owned spawned child + wall-cap/watchdog
          ├─ progress/checkpoint/interrupt propagation
          └─ generators/*_backend.py
              ├─ frozen train/validation + SamplingPlan load
              ├─ models/*.py build/fit/sample
              └─ eval/*.py metrics/guards
                  └─ artifact index + checksum + terminal-last marker
```

### 5.1 Controlled v2.5

- CLI/orchestration: `scripts/run_full_experiment_v2_5.py`
- model registry: `generators/full_registry_v2_5.py`
- artifact writer: `experiments/full_artifact_store_v2_5.py`
- evaluator: `eval/full_evaluation_v2_5.py`
- guards: `eval/model_guards_v2_5.py`
- aggregate/statistics: `eval/full_statistics_v2_5.py`
- frozen config: `configs/benchmark_v2/full_v2_5.yaml`

### 5.2 External validation/internal test

- data adapter: `data/external_sequence_adapter.py`
- validation CLI: `scripts/run_external_validation_v1.py`
- validation runner: `experiments/external_validation_runner_v1.py`
- metrics/protocol: `eval/external_validation_metrics_v1.py`, `eval/external_validation_protocol_v1.py`
- confirmatory CLI: `scripts/run_external_confirmatory_internal_test_v1.py`
- confirmatory runner: `experiments/external_confirmatory_internal_test_runner_v1.py`

### 5.3 CCMTPP C1

- model: `models/cof_ccmtpp_v1.py`
- backend: `generators/cof_ccmtpp_v1_execution_backend.py`
- runner: `experiments/cof_ccmtpp_v1_execution_runner.py`
- CLI: `scripts/run_cof_ccmtpp_v1.py`
- config: `configs/benchmark_v2/cof_ccmtpp_v1_{source_only,execution_runner}.yaml`

### 5.4 HCMTTPP H1

- model: `models/cof_hcmttpp_v2.py`
- backend: `generators/cof_hcmttpp_v2_execution_backend.py`
- runner: `experiments/cof_hcmttpp_v2_execution_runner.py`
- readiness boundary: `experiments/cof_hcmttpp_v2_execution_readiness.py`
- CLI: `scripts/run_cof_hcmttpp_v2.py`
- config: `configs/benchmark_v2/cof_hcmttpp_v2_{source_only,execution_runner}.yaml`

### 5.5 Artifact terminal-last 계약

일반적인 candidate attempt 구조는 다음과 같다.

```text
attempt_00N/
├── manifest.json                 immutable provenance
├── OWNERSHIP.json / RUNNING.json
├── progress.jsonl
├── checkpoints/
│   ├── step_*.pt
│   └── latest
├── sample.npz
├── metrics.json
├── evaluation.json
├── diagnostics.json
├── runtime.json
├── artifact_index.json
├── checksum_manifest.json
└── COMPLETE.json | INVALID.json | FAILED.json
```

Terminal marker는 마지막에 기록한다. `COMPLETE`는 process가 끝났다는 뜻이며 scientific gate `PASS`와 같지 않다. 예를 들어 CCMTPP C1과 Sparkov H1 attempt_003은 runtime `COMPLETE`지만 gate는 `FAIL`이다.

### 5.6 테스트와 fail-closed 경계

각 연구 단계는 source-only test에서 실제 학습을 호출하지 않고 plan/contract를 먼저 검증한 뒤, 별도 authorization을 통해 runtime을 열었다. 주요 regression 영역은 다음과 같다.

- DGP, binning, receiver calibration, fanout semantics: `tests/test_benchmark_v2_dgp.py`, `tests/test_fanout_semantics_v2_3.py`
- v2.5 artifact/resume/watchdog/statistics: `tests/test_full_*_v2_5.py`, `tests/test_learned_resume_v2_5.py`
- v2.6–v2.8 candidate/aggregate: `tests/test_candidate_runner_v2_6.py`, `tests/test_evaluation_*_v2_{7,8}.py`
- external split/UNK/leakage/materialization: `tests/test_external_sequence_protocol_v1.py`, `tests/test_external_frozen_materialization_v1.py`
- external validation/memory/performance: `tests/test_external_validation_*_v1.py`
- CCMTPP causal mask, pointer/hierarchy, runner watchdog: `tests/test_cof_ccmtpp_v1_*.py`
- H1 RQS/tail/ownership/readiness: `tests/test_cof_hcmttpp_v2_*.py`

특히 internal test와 Sparkov public fraudTest는 단순히 “안 읽는 관행”이 아니라 path resolution 전 차단을 test하는 계약으로 구현했다. Authorization mismatch, source/config/data/SamplingPlan hash mismatch, existing ownership, non-finite state, wall cap은 fail-closed다.

## 6. Controlled benchmark v2.4–v2.5

### 6.1 v2.4 CPU gate와 smoke

v2.4는 100-seed 초기 결과를 재분석한 뒤 global familywise statistic으로 교체·사전등록했다.

```text
T(s) = max over 2 scenarios × 2 kappa × 2 classifiers
       abs(test_AUROC - 0.5)
```

Calibration 200 clean seeds의 maximum을 threshold로 고정하고, 독립 validation 200 seeds의 exact upper bound와 injected leakage power를 검증했다. CPU gate가 통과한 상태는 tag `benchmark-v2.4-gate-pass`로 보존되었다. 핵심 evidence:

```text
artifacts/benchmark_v2_4/gates/gate_report.json
artifacts/benchmark_v2_4/artifact_index.json
artifacts/benchmark_v2_4/auroc_calibration/
docs/benchmark_v2/preregistered_analysis_v2_4*.md
docs/benchmark_v2/auroc_gate_calibration_v2_4.md
```

### 6.2 v2.5 full experiment 설계

고정 범위:

- scenario: `joint_semimarkov_v2b`
- kappa: `1.0`
- seed: 1–5
- 13 generators × 5 seeds = 65 cells
- learned generator cap: generator/seed당 2 GPU-hours
- CTGAN/TVAE는 Y=0/Y=1 separate-class model로 총 budget을 반분
- full-sequence block은 oracle/reference이지 learned competitor가 아님

13 generators:

1. empirical IID
2. block bootstrap 2
3. block bootstrap 4
4. block bootstrap 8
5. full-sequence reference
6. independent Markov
7. joint Markov
8. HMM
9. HSMM
10. CTGAN separate-class
11. TVAE separate-class
12. neural sequence baseline
13. CoF-SeqGen

### 6.3 v2.5 결과

최종 상태:

```text
artifacts/benchmark_v2_5/full/FINAL_COMPLETE.json
artifacts/benchmark_v2_5/full/finalization_attempt_001/full_experiment_report.md
```

`FINAL_COMPLETE`까지 도달했지만 analysis status는 `INVALID`, C2는 `NOT_EVALUABLE`이다. 이유는 C2 primary comparison에 필요한 learned generators와 empirical IID 모두가 5개 valid seeds를 갖지 못했기 때문이다. INVALID seed를 NaN으로 제외해 평균을 만들지 않았다.

Valid generator의 mean association-recovery error:

| generator | status | mean error | std |
|---|---|---:|---:|
| full-sequence reference | VALID | 0.00271536 | 0.00146454 |
| HMM | VALID | 0.00444163 | 0.00228445 |
| joint Markov | VALID | 0.00757022 | 0.00225189 |
| block 8 | VALID | 0.00813807 | 0.00273986 |
| block 4 | VALID | 0.01488537 | 0.00105559 |
| block 2 | VALID | 0.03360024 | 0.00130205 |
| independent Markov | VALID | 0.07116286 | 0.00138333 |
| empirical IID | VALID after evaluator correction/continuation | 0.07195501 | 0.00070574 |

Invalid generator families:

- HSMM: row-marginal hard guard 실패
- CTGAN: row-marginal hard guard 실패
- TVAE: row-marginal hard guard 실패
- neural sequence: row-marginal hard guard 실패/전체 valid seeds 부족
- CoF-SeqGen: row-marginal hard guard 실패

4-bin/8-bin support failure는 correction 이후 diagnostic으로만 남겼고 primary invalidity 원인이 아니다. Forensic은 evaluator defect를 refute하고 stored generated distribution이 실제 threshold를 벗어난 것으로 판정했다.

주요 결과 파일:

```text
artifacts/benchmark_v2_5/full/finalization_attempt_001/
├── analysis.json
├── aggregate_statistics.csv
├── raw_results.csv
├── support_diagnostics.csv
├── effect_sizes_and_holm.csv
├── per_seed_runtime.csv
├── full_experiment_report.md
├── artifact_index.json
└── checksum_manifest_report.json
```

실행/continuation 로그:

```text
artifacts/benchmark_v2_5/full/runner_console.log
artifacts/benchmark_v2_5/full/runner_console_attempt_002.log
artifacts/benchmark_v2_5/full/runner_console_attempt_003.log
artifacts/benchmark_v2_5/full/run_state.jsonl
artifacts/benchmark_v2_5/full/continuation_attempt_003/run_state.jsonl
```

## 7. v2.6–v2.8 train/validation-only candidate selection

### 7.1 공통 원칙

- test split을 candidate fit/selection에 사용하지 않음
- five row-marginal guards 모두 PASS한 후보만 eligible
- amount KS, gap KS, amount effect, gap effect, receiver frequency를 기록
- 결과를 보고 threshold를 변경하지 않음
- append-only candidate/trajectory/worker/aggregate artifacts

### 7.2 v2.6

초기 16 evaluation candidates 및 single-factor 9 candidates를 실행했다. CTGAN/TVAE/CoF 모두 `NO_PASSING_CANDIDATE`; neural은 secondary `NOT_EVALUABLE`이었다.

```text
artifacts/benchmark_v2_6/selection/aggregate_attempt_001/
artifacts/benchmark_v2_6/selection_single_factor/aggregate_attempt_001/
```

### 7.3 v2.7

결과:

| model | selection | candidate | five guard 값 |
|---|---|---|---|
| CTGAN | NO_PASSING_CANDIDATE | — | amount 개선 후 gap/receiver 실패 지속 |
| TVAE | SELECTED | `tvae_v27_c01_amount_inverse_decoder` | amount KS 0.004900, gap KS 0.005769, amount effect 0.029938, gap effect 0.001375, receiver 0.006794 |
| CoF | NO_PASSING_CANDIDATE | — | amount 개선 후 gap/receiver 문제가 남음 |

```text
artifacts/benchmark_v2_7/candidate_selection/aggregate_attempt_001/
docs/benchmark_v2/forensic_no_passing_candidate_v2_7.md
```

### 7.4 v2.8

| model | candidate | 결과 |
|---|---|---|
| CTGAN | `ctgan_v28_c01_joint_gap_receiver_decoder` | all-five PASS, SELECTED |
| TVAE | frozen `tvae_v27_c01_amount_inverse_decoder` | all-five PASS reference |
| CoF | `cof_v28_c01_gap_distribution_sampler` | receiver frequency 0.0211266 > 0.02로 FAIL |

CTGAN selected metric:

```text
amount effect 0.0139476
amount KS     0.00321845
gap effect    0.00669708
gap KS        0.00189562
receiver      0.00357626
```

CoF v2.8 metric:

```text
amount effect 0.00529733 PASS
amount KS     0.00333091 PASS
gap effect    0.00676709 PASS
gap KS        0.00190082 PASS
receiver      0.02112656 FAIL (threshold 0.02)
```

따라서 `primary_c2_selection_ready=false`; fresh test는 열리지 않았다.

```text
artifacts/benchmark_v2_8/candidate_selection/aggregate_attempt_001/
docs/benchmark_v2/evaluation_aggregate_v2_8.md
```

## 8. CoF-SeqGen v3 joint discrete redesign

### 8.1 설계

v2.8 CoF가 amount/gap은 회복했으나 receiver guard를 근소하게 실패해 두 joint discrete 후보를 고정했다.

- `cof_v3_c01_direct_joint`: `z = gap_bin * K_receiver + receiver_code`, 하나의 joint embedding/head
- `cof_v3_c02_factorized_joint`: `p(gap|h,Y) p(receiver|gap,h,Y)`
- amount residual path, backbone, Y/L conditioning, valid mask는 고정
- post-hoc calibration 금지

Source:

```text
models/cof_seqgen_v3.py
generators/cof_seqgen_v3_candidate.py
experiments/cof_seqgen_v3_execution_runner.py
scripts/run_cof_seqgen_v3_validation.py
```

### 8.2 실행과 결과

`attempt_001`은 child finalization/deadlock 문제로 FAILED 보존되었다. correction 후 `attempt_002`에서 두 후보 모두 sample/evaluation까지 갔지만 INVALID였다.

| candidate | amount KS | amount effect | gap KS | gap effect | receiver | 결론 |
|---|---:|---:|---:|---:|---:|---|
| direct joint | 0.002371 PASS | 0.025977 PASS | 0.088096 FAIL | 0.092396 FAIL | 0.055186 FAIL | INVALID |
| factorized joint | 0.004755 PASS | 0.023221 PASS | 0.082064 FAIL | 0.473613 FAIL | 0.302605 FAIL | INVALID |

Forensic 판정:

- evaluator/mask/label mapping defect: REFUTED
- joint codec/support/decoder implementation defect: REFUTED
- current objective/discrete generation architecture limitation: SUPPORTED

```text
artifacts/benchmark_v3/candidate_selection/
docs/benchmark_v2/forensic_v3_attempt_002_failure.md
docs/benchmark_v2/forensic_v3_attempt_002_evidence.{json,csv}
```

## 9. 외부 AMLSim/Sparkov validation

### 9.1 비교 모델

각 dataset에서 동일한 네 모델을 비교했다.

1. empirical IID
2. CTGAN separate-class
3. TVAE separate-class
4. frozen non-v3 CoF-SeqGen

Frozen CoF baseline provenance:

- source commit: `99a445f6dc893a8c2240d950de4f92877cc07f8a`
- controlled config SHA-256: `81bc16416f6dad3c23eaf1990c3cb76e5449e74f1ddc634f42d3a897cd3c5bb3`

외부 dataset에서는 controlled benchmark의 five-guard 숫자 threshold를 복사하지 않았다. train-only bootstrap reference로 fidelity/coherence/combined score를 고정했다. 점수는 작을수록 좋다.

### 9.2 Validation 결과

#### AMLSim validation

| rank | model | fidelity | coherence | combined |
|---:|---|---:|---:|---:|
| 1 | empirical IID | 1.8783 | 13.7394 | 7.8088 |
| 2 | TVAE | 6.8964 | 13.1489 | 10.0226 |
| 3 | frozen CoF | 12.3967 | **12.9044** | 12.6506 |
| 4 | CTGAN | 12.3592 | 13.7514 | 13.0553 |

CoF는 coherence만 가장 좋았고 fidelity/combined에서 baseline을 이기지 못했다.

#### Sparkov validation

| rank | model | fidelity | coherence | combined |
|---:|---|---:|---:|---:|
| 1 | empirical IID | 2.096461 | 5.019435 | 3.557948 |
| 2 | CTGAN | 10.486262 | 3.940879 | 7.213571 |
| 3 | frozen CoF | 13.063355 | **3.857913** | 8.460634 |
| 4 | TVAE | 48.694145 | 429.970598 | 239.332371 |

사전등록된 CoF-vs-TVAE continue rule은 Sparkov에서 통과했지만, CoF가 전체 최고라는 뜻은 아니다.

#### 두 dataset macro validation

| rank | model | macro fidelity | macro coherence | macro combined |
|---:|---|---:|---:|---:|
| 1 | empirical IID | 1.9874 | 9.3794 | 5.6834 |
| 2 | CTGAN | 11.4227 | 8.8462 | 10.1344 |
| 3 | frozen CoF | 12.7300 | **8.3812** | 10.5556 |
| 4 | TVAE | 27.7953 | 221.5598 | 124.6775 |

결론: frozen non-v3 CoF는 두 dataset에서 coherence 방향의 강점은 보였지만 fidelity와 combined score의 전체 우월성은 보이지 않았다.

Evidence:

```text
docs/benchmark_v2/amlsim_external_validation_aggregate_forensic_v1.{md,json,csv}
docs/benchmark_v2/sparkov_external_validation_aggregate_forensic_v1.{md,json,csv}
docs/benchmark_v2/cross_dataset_external_validation_aggregate_v1.{md,json,csv}
artifacts/external_validation_v1/{amlsim,sparkov}/<model>/attempt_*/
```

### 9.3 실행 중 발견·교정된 infrastructure/performance 문제

- AMLSim empirical IID `attempt_001`: receiver-TV dense `N×K` 계산으로 wall cap FAIL
  - `attempt_002`: metric 정의/seed는 유지하고 sparse/vectorized 계산으로 COMPLETE
- AMLSim TVAE `attempt_001`: CTGAN과 동시에 `DataTransformer.transform(n_jobs=-1)`을 실행해 global OOM/SIGKILL
  - 외부 CTGAN/TVAE transform을 config-bound `n_jobs=1`로 고정
  - TVAE `attempt_002` 단독 continuation COMPLETE
- Sparkov IID는 새 source provenance 통일을 위해 append-only `attempt_002`도 보존

이 교정들은 scientific metric/threshold/model architecture 변경이 아니라 실행 복잡도와 memory safety 수정이다.

## 10. Confirmatory internal-test

Validation에서 model/fit-state/threshold/conditioning을 고정한 뒤 AMLSim/Sparkov `internal_test.npz`에서 네 모델을 정확히 한 번씩, 총 8 jobs 평가했다. Sparkov public `fraudTest`는 계속 제외했다.

### 10.1 AMLSim internal test

| rank | model | fidelity | coherence | combined |
|---:|---|---:|---:|---:|
| 1 | empirical IID | 1.796531 | 15.307182 | 8.551856 |
| 2 | TVAE | 6.166695 | 14.707800 | 10.437247 |
| 3 | CTGAN | 12.103967 | 15.318657 | 13.711312 |
| 4 | frozen CoF | 13.771584 | **14.457383** | 14.114484 |

### 10.2 Sparkov internal test

| rank | model | fidelity | coherence | combined |
|---:|---|---:|---:|---:|
| 1 | empirical IID | 2.142842 | 4.559339 | 3.351091 |
| 2 | CTGAN | 11.133907 | 4.152255 | 7.643081 |
| 3 | frozen CoF | 12.681494 | **3.134546** | 7.908020 |
| 4 | TVAE | 48.588298 | 430.002688 | 239.295493 |

### 10.3 Macro internal-test

| rank | model | macro fidelity | macro coherence | macro combined |
|---:|---|---:|---:|---:|
| 1 | empirical IID | 1.969686 | 9.933261 | 5.951473 |
| 2 | CTGAN | 11.618937 | 9.735456 | 10.677197 |
| 3 | frozen CoF | 13.226539 | **8.795964** | 11.011252 |
| 4 | TVAE | 27.377496 | 222.355244 | 124.866370 |

최종 claim state:

```text
ONLY_DESCRIPTIVE_COHERENCE_ADVANTAGE_SUPPORTED
```

즉 “CoF가 IID·CTGAN·TVAE보다 전체적으로 우수하다”는 주장은 지원되지 않고, “sequence coherence에서는 descriptive advantage가 있다”만 지원된다.

```text
artifacts/external_confirmatory_internal_test_v1/{amlsim,sparkov}/<model>/attempt_001/
docs/benchmark_v2/external_confirmatory_internal_test_aggregate_forensic_v1.{md,json,csv}
```

## 11. cof_ccmtpp_v1 C1

### 11.1 모델 설계

`cof_ccmtpp_v1`은 기존 full-sequence diffusion과 분리한 conditional coherence-preserving marked temporal point process family다.

C1에서 실제 사용한 요소:

- causal Transformer event decoder
- shifted history, future attention 차단
- sequence Y/target length/valid mask conditioning
- continuous `log1p(gap)` mixture-logistic head
- flat receiver decoder
- 기존 amount encode/decode contract
- seed 4001, 20,000 updates, 7,200초 hard cap

C2 pointer-copy, C3 hierarchy, C4 Y-balanced likelihood는 C1 gate가 열어야 실행 가능하도록 설계되었다.

### 11.2 실행 결과

두 dataset 모두 20,000 updates와 hard validity를 완료했지만 C1 scientific gate는 실패했다.

| dataset | runtime | peak GPU memory | Y0 gap KS | C0 Y0 gap KS | Y1 gap KS | C0 Y1 gap KS | 결론 |
|---|---:|---:|---:|---:|---:|---:|---|
| AMLSim | 243.74s | 776.5 MB | 0.235709 | 0.227624 | 0.203175 | 0.466188 | Y0 비열화로 FAIL |
| Sparkov | 171.64s | 142.1 MB | 0.069522 | 0.042308 | 0.152207 | 0.300652 | Y0 비열화로 FAIL |

Coherence는 두 dataset/Y class에서 C0보다 개선되었다.

| dataset | C1 coherence Y0/Y1 | C0 coherence Y0/Y1 | full receiver TV |
|---|---|---|---:|
| AMLSim | 0.096367 / 0.046203 | 0.213972 / 0.108703 | 0.562530 |
| Sparkov | 0.0002969 / 0.0002589 | 0.0005020 / 0.0005177 | 0.062438 |

최종 chain state:

```text
STOP_CCMTPP_V1_CHAIN_C1_GATE_FAIL
```

C2/C3/C4는 실행·authorization이 금지되었다.

Forensic은 양 dataset 공통 Y0 악화의 주원인을 `positive-gap conditional calibration problem`으로 판정했다. evaluator bin mapping이나 train→validation shift가 주원인은 아니었다.

```text
artifacts/cof_ccmtpp_v1/external_validation/{amlsim,sparkov}/C1/seed_4001/attempt_001/
docs/benchmark_v2/forensic_cof_ccmtpp_v1_c1_gap_failure.{md,json,csv}
docs/benchmark_v2/cof_ccmtpp_v1_{architecture,artifact_contract,execution_runner}.md
```

## 12. cof_hcmttpp_v2 H1 — 가장 최근 실제 실행 모델

### 12.1 H1 설계

H1은 C1의 causal backbone, Y/length conditioning, amount path, flat receiver, optimizer/budget/SamplingPlan을 고정하고 gap decoder만 바꿨다.

```text
p(gap | h,Y)
  = P(gap=0 | h,Y) * delta_0
    + P(gap>0 | h,Y) * p_positive(gap | h,Y)
```

구성:

- exact-zero Bernoulli hurdle
- positive `log1p(gap)` 16-bin monotone rational-quadratic spline body
- train-only Y별 strict-exceedance tail threshold
- conditional tail gate
- conditional exponential tail scale `beta_base(Y) * exp(r_beta(h,Y))`
- C1 non-gap modules 고정

### 12.2 attempt chronology

| attempt | AMLSim | Sparkov | 의미 |
|---|---|---|---|
| `attempt_001` | FAILED | FAILED | 약 2.5초, `RQS inverse root is invalid`; valid endpoint의 float32 analytic inverse 구현 오류 |
| `attempt_002` | FAILED | FAILED | parent-created ownership와 spawned child attach identity mismatch; runner contract 오류 |
| readiness gate | CPU synthetic PASS | CPU synthetic PASS | parent→spawn→attach→model build→event_nll→backward→optimizer 1 step 검증 |
| `attempt_003` | FAILED | COMPLETE | 최초 실제 H1 scientific execution attempt |

Attempt 001/002는 architecture performance 결과가 아니라 deterministic implementation failure였고 append-only로 보존했다.

### 12.3 AMLSim attempt_003

- status: `FAILED`
- elapsed: 277.6716초
- failure: `InvalidH1GapStateError: H1 gap parameters are non-finite or invalid`
- 마지막 finite progress/checkpoint: step 14,000
- 최초 invalid 관찰 구간: step 14,001–14,099
- 정확히 어떤 내부 tensor/module이 최초 원인인지: 저장된 diagnostic 해상도가 부족해 `INCONCLUSIVE`
- evaluator/artifact/runner defect 가설: REFUTED
- 분류: H1 numerical-stability/architecture limitation

```text
artifacts/cof_hcmttpp_v2/external_validation/amlsim/H1/seed_4001/attempt_003/
├── progress.jsonl
├── checkpoints/
├── FAILED.json
├── artifact_index.json
└── checksum_manifest.json
```

### 12.4 Sparkov attempt_003

- status: runtime `COMPLETE`
- actual updates: 20,000
- elapsed: 340.61초
- peak GPU memory: 1.0417 GB
- hard validity: PASS
- scientific H1 gate: FAIL

| metric | H1 value | reference/criterion | result |
|---|---:|---:|---|
| Y0 overall gap KS | 0.199433 | C0 0.042308 이하 | FAIL |
| Y1 overall gap KS | 0.110331 | C0 0.300652 이하 | PASS |
| Y0 positive-only gap KS | 0.199433 | C1 0.069522보다 엄격히 작음 | FAIL |
| Y0 coherence | 0.0004750 | C1 0.0002969 이하 | FAIL |
| Y1 coherence | 0.0005177 | C1 0.0002589 이하 | FAIL |
| Y0 receiver TV | 0.055170 | C1 0.062882 이하 | PASS |
| Y1 receiver TV | 0.248495 | C1 0.242979 이하 | FAIL |
| full receiver TV | 0.054899 | C1 0.062438 이하 | PASS |

### 12.5 H1 최종 상태

```text
STOP_COF_HCMTTPP_V2_FAMILY_H1_GATE_FAIL
```

H1 attempt_004, H1 retry, internal test, Sparkov fraudTest는 허용되지 않는다.

```text
docs/benchmark_v2/forensic_cof_hcmttpp_v2_h1_attempt_003.md
docs/benchmark_v2/forensic_cof_hcmttpp_v2_h1_attempt_003.json
docs/benchmark_v2/forensic_cof_hcmttpp_v2_h1_attempt_003_steps.csv
artifacts/cof_hcmttpp_v2/external_validation/{amlsim,sparkov}/H1/seed_4001/attempt_00{1,2,3}/
```

## 13. CoF-ZDH-v3 D1 — 현재 HEAD의 최신 설계

D1은 H1 retry가 아니라 별도 family다. 아직 model class, runner, authorization, runtime artifact가 없다.

### 13.1 고정 설계

- family: `cof_zdh_v3`
- candidate: `D1` 하나만
- 변경 factor: gap decoder 하나만
- C1 causal backbone, Y/L, mask, amount, flat receiver, optimizer, seed, budget, SamplingPlan은 고정
- exact zero를 별도 category로 모델링
- positive `u=log1p(gap)`를 train-only Y별 32개 fixed interval의 conditional discrete hazard로 모델링
- bounded hazard logit: `12 * tanh(raw/12)`
- within-bin density: uniform in `u`
- tail: train-only fixed-scale shifted exponential
- RQS inverse/root finder 없음
- `exp(r_beta)` 없음
- clipping/fallback/redraw 금지

H1의 두 수치 메커니즘, 즉 RQS inverse와 unbounded learned exponential scale을 구조적으로 제거한다. 이는 안정성 설계 근거이지 성능 PASS 보장은 아니다.

### 13.2 향후 D1 gate

AMLSim과 Sparkov 모두에서 다음을 conjunctive하게 만족해야 한다.

1. hard validity PASS
2. numerical invalid count 0
3. C0 대비 Y0/Y1 gap KS 비열화 없음
4. C1 대비 Y0 positive-only gap KS 엄격 개선
5. C1 대비 Y0/Y1 coherence 비열화 없음
6. C1 대비 classwise/full receiver TV 비열화 없음

PASS해도 internal test 권한을 자동으로 열지 않는다. 하나라도 실패하면:

```text
STOP_COF_ZDH_V3_FAMILY_D1_GATE_FAIL
```

현재 source-only 파일:

```text
configs/benchmark_v2/cof_zdh_v3_d1_source_only.yaml
docs/benchmark_v2/cof_zdh_v3_d1_architecture_specification.md
docs/benchmark_v2/preregistered_cof_zdh_v3_d1.md
docs/benchmark_v2/cof_zdh_v3_d1_artifact_contract.md
docs/benchmark_v2/cof_zdh_v3_d1_red_green_test_plan.md
```

현재 config에 명시된 상태:

```text
SPECIFIED_NOT_IMPLEMENTED
model_path: null
execution_runner_path: null
authorization_path: null
runtime_root_created: false
```

## 14. 로그와 결과를 찾는 방법

### 14.1 특정 run이 성공/실패했는지

각 attempt의 terminal marker부터 본다.

```text
COMPLETE.json   실행과 required artifact 완료
INVALID.json    실행은 되었지만 hard scientific/data contract 위반
FAILED.json     code/infrastructure/wall-cap/child exception
WORKER_COMPLETE.json  model/dataset worker의 terminal 완료
AGGREGATE_COMPLETE.json  stored evidence aggregate 완료
FINAL_COMPLETE.json  top-level full experiment finalization 완료
STOPPED.json    global contract/infrastructure stop
```

### 14.2 실제 학습 진행

- `progress.jsonl`: step, loss, elapsed time, memory
- `checkpoints/step_*.pt`: checkpoint
- `checkpoints/latest`: 마지막 atomic checkpoint pointer
- `runtime.json`: actual updates, elapsed, peak memory, device
- `stdout.log`, `stderr.log`, runner console log: process output

### 14.3 모델 생성 결과와 평가

- `sample.npz`: synthetic sequences
- `metrics.json`: primary fidelity/coherence/association metrics
- `evaluation.json`: validity/access status
- `diagnostics.json`: classwise/channel/support/decode/NLL detail
- `gate_decision.json`: preregistered parent comparison과 PASS/FAIL
- `artifact_index.json`, `checksum_manifest.json`: 파일 목록과 hash integrity

### 14.4 과학적 결론의 우선순위

결론을 읽을 때 다음 순서를 지켜야 한다.

1. preregistration/config
2. terminal marker와 checksum/index
3. `evaluation.json`의 hard validity
4. `gate_decision.json` 또는 aggregate selection report
5. forensic Markdown/JSON/CSV

단순 `COMPLETE`나 낮은 일부 metric 하나만으로 다음 단계 authorization 또는 모델 우월성을 주장하면 안 된다.

## 15. 핵심 문서와 evidence index

### Controlled benchmark

- `docs/benchmark_v2/preregistered_analysis_v2_4.md`
- `docs/benchmark_v2/preregistered_full_experiment_v2_5.md`
- `docs/benchmark_v2/baseline_definitions_v2_5.md`
- `docs/benchmark_v2/full_experiment_artifact_contract_v2_5.md`
- `docs/benchmark_v2/forensic_row_marginal_failure_v2_5.md`
- `artifacts/benchmark_v2_5/full/finalization_attempt_001/`

### Candidate selection

- `docs/benchmark_v2/preregistered_model_selection_v2_6.md`
- `docs/benchmark_v2/forensic_selection_failure_v2_6.md`
- `docs/benchmark_v2/forensic_single_factor_failure_v2_6.md`
- `docs/benchmark_v2/forensic_no_passing_candidate_v2_7.md`
- `docs/benchmark_v2/evaluation_aggregate_v2_8.md`

### External validation

- `docs/benchmark_v2/preregistered_external_sequence_protocol_v1.md`
- `docs/benchmark_v2/external_frozen_materialization_v1.md`
- `docs/benchmark_v2/preregistered_external_validation_v1.md`
- `docs/benchmark_v2/cross_dataset_external_validation_aggregate_v1.md`
- `docs/benchmark_v2/external_confirmatory_internal_test_aggregate_forensic_v1.md`

### CoF redesigns

- `docs/benchmark_v2/cof_seqgen_v3_model_redesign_specification.md`
- `docs/benchmark_v2/forensic_v3_attempt_002_failure.md`
- `docs/benchmark_v2/cof_ccmtpp_v1_architecture.md`
- `docs/benchmark_v2/forensic_cof_ccmtpp_v1_c1_gap_failure.md`
- `docs/benchmark_v2/cof_hcmttpp_v2_architecture_specification.md`
- `docs/benchmark_v2/forensic_cof_hcmttpp_v2_h1_attempt_003.md`
- `docs/benchmark_v2/cof_zdh_v3_d1_architecture_specification.md`

## 16. 최종 연구 해석

### 16.1 확인된 것

- Controlled DGP/gate는 v2.4에서 통과·동결되었다.
- v2.5 full experiment는 65-cell terminal state와 finalization까지 끝났다.
- v2.5의 C2 learned comparison은 `NOT_EVALUABLE`; 임의 평균/threshold 완화는 하지 않았다.
- CTGAN/TVAE는 v2.7/v2.8 train/validation-only single-factor 수정으로 controlled five guards를 통과하는 후보를 찾았다.
- non-v3 CoF는 AMLSim/Sparkov validation과 internal test에서 반복적으로 coherence가 가장 좋거나 유리한 방향이었다.
- 그러나 CoF의 fidelity와 combined score는 empirical IID/CTGAN보다 전반적으로 나빴고, 전체 우월성은 지지되지 않았다.
- receiver/gap fidelity 병목을 고치려 한 CoF v3, CCMTPP C1, HCMTTPP H1은 각자의 사전등록 gate에서 중단되었다.

### 16.2 주장할 수 없는 것

- “CoF가 모든 baseline보다 우수하다.”
- “H1은 AMLSim에서도 정상 완료했다.”
- “Sparkov H1 COMPLETE이므로 H1이 성공했다.”
- “D1이 구현 또는 실행되었다.”
- “public Sparkov fraudTest를 평가했다.”
- “candidate 결과를 본 뒤 threshold를 넓혀 PASS시켰다.”

### 16.3 현재 정확한 종착점

```text
v2.5 controlled C2: NOT_EVALUABLE
v2.8 primary selection: NOT READY (CoF receiver guard failure)
CoF v3: STOPPED after architecture limitation evidence
CCMTPP-v1: STOP_CCMTPP_V1_CHAIN_C1_GATE_FAIL
HCMTTPP-v2: STOP_COF_HCMTTPP_V2_FAMILY_H1_GATE_FAIL
CoF-ZDH-v3 D1: SPECIFIED_NOT_IMPLEMENTED
external claim: ONLY_DESCRIPTIVE_COHERENCE_ADVANTAGE_SUPPORTED
```

## 17. 재현 또는 후속 작업 전에 확인할 체크리스트

1. `git rev-parse HEAD`와 실행 대상 config/source hash를 확인한다.
2. 기존 attempt의 terminal/index/checksum을 읽기 전용으로 검증한다.
3. 새 연구 family인지, frozen family continuation인지 구분한다.
4. train-only transform/SamplingPlan/threshold의 hash를 고정한다.
5. validation/test/fraudTest access boundary를 확인한다.
6. authorization이 dataset/candidate/seed/attempt를 정확히 한정하는지 검증한다.
7. append-only 새 attempt가 선택되는지 확인한다.
8. terminal marker와 checksum/index가 없으면 과학적 결과로 해석하지 않는다.
9. 부분 결과를 보고 threshold, seed, budget, objective를 바꾸지 않는다.
10. 현재 D1은 별도 구현 지시 전에는 실행할 수 없다.

---

이 문서는 “현재 저장소에서 무엇이 실제로 구현·실행되었고, 무엇이 실패·종료되었으며, 무엇이 아직 설계뿐인지”를 한 파일에서 추적하기 위한 snapshot이다. 세부 수치의 최종 근거는 각 절에 적은 append-only artifact와 forensic JSON/CSV다.
