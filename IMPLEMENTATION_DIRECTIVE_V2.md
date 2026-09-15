# 원격 Codex 구현 지시서: CoF-SeqGen Benchmark v2 재설계

> 대상 저장소: `cof-seqgen-0707-2119-Version3-complete`
>
> 실행 환경: 연구실 GPU 워크스테이션의 SSH/VS Code 환경
>
> 문서 목적: 현재 v1 연구의 핵심 식별 문제를 제거하고, **단일 행 주변분포는 라벨 간 동일하지만 순차 의존성만 다른 2단 benchmark suite**를 구현·검증한 뒤 공정한 baseline과 CoF-SeqGen을 다시 비교한다. `v2a`는 해석 가능한 1차 Markov calibration scenario, `v2b`는 공동 semi-Markov 결속을 갖는 primary model-challenge scenario다.
>
> 이 문서는 아이디어 메모가 아니라 **구현 명세서이자 acceptance contract**다. 임의로 연구 질문을 바꾸거나 gate를 생략하지 말 것.

---

## 0. 원격 Codex에게 줄 최상위 명령

아래 문장을 이 파일과 함께 원격 Codex에게 전달한다.

```text
저장소 루트에서 REMOTE_CODEX_CoF-SeqGen_Benchmark-v2_구현지시서.md를 처음부터 끝까지 읽고,
Phase 0~6을 순서대로 구현하라.

기존 데이터·결과·로그·모델 파일을 삭제하거나 덮어쓰지 말고, legacy v1 결과와 v2 결과를 섞지 마라.
benchmark gate가 PASS하기 전에는 CoF 전체 학습을 실행하지 마라.
각 phase의 테스트와 acceptance criteria를 통과한 증거를 docs/v2_implementation_report.md에 기록하라.

기본 실행 범위는 IMPLEMENT_AND_SMOKE이다:
- 모든 코드 구현
- unit/integration/smoke test 실행
- 작은 데이터로 end-to-end smoke run
- benchmark validation gate의 full-data CPU 실행

5-seed 전체 GPU 학습은 내가 "FULL_EXPERIMENT까지 실행"이라고 명시한 경우에만 실행하라.
그렇지 않으면 정확한 full-run 명령과 예상 산출물을 작성한 뒤 멈춰라.
```

### 실행 모드

- `IMPLEMENT_AND_SMOKE`:
  - 코드·config·test·문서 구현
  - CPU gate와 작은 smoke training까지 실행
  - full GPU 실험은 실행하지 않음
- `FULL_EXPERIMENT`:
  - 위 작업에 더해 모든 baseline과 CoF의 정식 multi-seed GPU 실험 실행
  - 통계·표·그림까지 생성

실행 모드가 명시되지 않으면 `IMPLEMENT_AND_SMOKE`로 간주한다.

---

## 1. 현재 연구에서 확정된 문제

### 1.1 중심 식별 문제

현재 v1 κ-data는 fraud-mode에서:

- gap이 각 행마다 독립 `Exp(0.5)`
- receiver가 각 행마다 항상 category 0

이고, 나머지는:

- gap이 각 행마다 독립 `Exp(2.0)`
- receiver가 category 1–63

이다. 따라서 `P(row | y=1) != P(row | y=0)`이며, 행동–라벨 결합을 순차 모델 없이도 행 단위 조건부 분포만으로 표현할 수 있다.

실제 확인된 반례:

- label-first conditional i.i.d. row bootstrap
- κ=0.7, 2-bin gap: 약 0.0123/0.0134/0.0128
- κ=1.0, 2-bin gap: 약 0.0015/0.0006/0.0004
- 기존 CoF 평균보다 낮고 κ=1에서는 C0보다도 낮음

따라서 v1 결과는 다음 명제를 입증하지 못한다.

> Row-independent generators are structurally incapable of preserving the target coupling.

### 1.2 padding 문제

- 실제 데이터 sequence가 left-pad되어 있음
- `models/soft_g.py`가 `valid_mask`를 받지 않음
- padding이 과거 거래로 집계됨
- `models/teacher.py`의 `compute_g_from_real` 및 통계도 padding을 포함
- `models/sampler.py`는 sampling forward에서 padding mask를 전달하지 않음

v2에서는 mask가 데이터에서 metric까지 끊기지 않는 하나의 계약이어야 한다.

### 1.3 연구 재현성 문제

- 프로젝트 코드·결과 상당수가 Git에서 untracked
- 주요 checkpoint와 synthetic sample 부재
- baseline seed가 모델 학습까지 완전히 고정되지 않음
- 결과를 생성하는 config·command·commit hash가 결과 폴더에 없음
- 일부 stale test 존재

---

## 2. 이번 수정의 연구 목표

### Primary research question

> 라벨 간 단일 행 조건부 주변분포가 동일하고 라벨 차이가 오직 entity 내부의 순차 의존성에만 존재할 때, row-independent generator와 sequence generator가 행동–라벨 coherence를 얼마나 보존하는가?

### Benchmark v2의 필수 성질

1. 모든 κ에서 `P(row | y=0) = P(row | y=1)`이 population level에서 동일해야 한다.
2. 라벨 차이는 lag dependence, persistence, transition, run structure에만 존재해야 한다.
3. κ는 fraud entity 비율이 아니라 **순차 의존성 강도**를 조절해야 한다.
4. label-conditional i.i.d. row bootstrap은 C1과 실질적으로 동등해야 한다.
5. true DGP oracle은 C0에 접근해야 한다.
6. 위 조건은 2/4/8-bin에서 확인되어야 한다.
7. benchmark gate는 CoF 결과를 보기 전에 고정·통과해야 한다.
8. 한 행에서는 label 신호가 없어야 하지만, 연속 pair 또는 더 긴 context에는 검출 가능한 positive-control 신호가 있어야 한다.
9. `v2a`와 `v2b`의 목적을 분리한다.
   - `v2a`: metric·gate·baseline이 예상대로 작동하는지 검증하는 calibration scenario
   - `v2b`: 저차 독립 Markov만으로는 충분하지 않은 joint semi-Markov model-challenge scenario

### v2에서 허용되는 주장

gate와 full experiment가 모두 성공한 경우에만:

> Under benchmarks with matched class-conditional row marginals and label-dependent temporal dependence, row-independent generators fail to recover the sequence-level coupling while sequence-aware generators can preserve part or all of it.

“모든 row-independent generator가 모든 행동–라벨 결합을 구조적으로 보존할 수 없다”와 같은 무제한 일반화는 금지한다.

`v2a`에서 correctly specified 또는 거의 correctly specified인 Markov baseline이 CoF와 동률이거나 우수한 것은 실패가 아니다. `v2a` 단독으로 CoF 우월성을 주장하지 않는다. CoF의 모델 기여는 `v2b`와 일반 sequence baseline 비교에서 판단한다.

---

## 3. 절대 지켜야 할 원칙

### 하지 말아야 할 것

- 기존 `data/kappa_*`, `results/`, `logs/`, `models/*.pt` 삭제 금지
- `git reset --hard`, `git clean -fdx`, 광범위한 checkout 금지
- 기존 v1 CSV를 v2 결과에 복사하거나 병합 금지
- benchmark gate 실패 상태에서 CoF 성능을 보고 DGP parameter를 조정하는 행위 금지
- CoF가 잘 나오도록 κ, `rho_max`, bin 수, primary metric을 사후 변경하는 행위 금지
- conditional i.i.d. bootstrap을 “학습된 생성기 성능”으로 표현 금지
- synthetic sample을 저장하지 않은 실험을 정식 결과로 채택 금지
- single-seed 결과로 유의성 주장 금지
- NN ratio 하나로 “no memorization” 결론 금지
- legacy `baseline_coherence_harness.py`를 v2 코드로 대규모 덮어쓰기 금지

### 반드시 할 것

- v1 보존
- source/config/test만 Git 추적하고 대형 data/checkpoint/result는 Git에서 제외
- 모든 결과에 resolved config, seed, Git SHA, environment 기록
- 모든 generator가 같은 label/length sampling plan 사용
- 모든 multi-seed 결과에서 model seed와 sampling seed를 분리·기록
- 모든 정식 합성 sample 저장
- CPU gate PASS 후 GPU experiment 진행
- failure도 숨기지 말고 report에 기록

---

## 4. 작업 단계와 중단 조건

```text
Phase 0  Preflight + v1 동결
   ↓
Phase 1  mask-aware sequence contract 수정
   ↓
Phase 2  benchmark v2 DGP 구현
   ↓
Phase 3  benchmark validity gate
   ├─ FAIL → 원인 보고 후 DGP/metric만 수정, 모델 학습 금지
   └─ PASS
        ↓
Phase 4  공통 generator interface + baseline adapters
        ↓
Phase 5  unified evaluation/artifact pipeline + smoke
        ↓
Phase 6  기존 CoF adapter 연결 + smoke
        ↓
Phase 7  FULL_EXPERIMENT일 때만 5+ seed GPU 실행
        ↓
Phase 8  실제 AMLSim/Sparkov 재계산
```

### 강제 중단 조건

다음 중 하나라도 발생하면 full model training으로 넘어가지 말고 `docs/v2_implementation_report.md`에 blocker를 기록한다.

- single-row leakage gate 실패
- conditional i.i.d. bootstrap이 C1보다 실질적으로 우수
- DGP oracle이 C0에 접근하지 못함
- mask padding invariance test 실패
- 생성 sample 저장 또는 manifest 기록 실패
- baseline과 CoF가 동일 label/length plan을 사용하지 않음

---

## 5. Phase 0 — Preflight와 v1 동결

### 5.1 저장소 루트 확인

현재 디렉터리 또는 하위의 저장소 중 다음 항목이 동시에 존재하는 곳을 `REPO_ROOT`로 선택한다.

```text
models/cof_seqgen.py
models/soft_g.py
scripts/generate_kappa_data.py
scripts/baseline_coherence_harness.py
tests/test_soft_g_grad.py
```

중첩된 동일 이름 폴더가 있으면 실제 파일이 있는 내부 폴더에서 작업한다. 절대 경로를 소스에 하드코딩하지 않는다.

### 5.2 읽기 전용 상태 점검

다음을 기록한다.

```bash
pwd
git status --short
git rev-parse HEAD
git branch --show-current
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
python -c "import numpy, sklearn; print(numpy.__version__, sklearn.__version__)"
```

결과를 `docs/audit/v2_preflight.md`에 기록한다.

### 5.3 기존 상태 보호

- dirty/untracked 파일을 임의로 제거하지 않는다.
- 기존 결과 위치와 파일 수를 `docs/audit/v1_asset_inventory.md`에 기록한다.
- 현재 Git history가 upstream TabDiff commit만 포함하고 CoF-SeqGen 프로젝트 source가 untracked이면 **먼저 추적 가능한 프로젝트 snapshot commit을 만든다. 태그부터 만들지 않는다.**
- `.gitignore`를 정리한 뒤 source/config/test/doc만 추적한다.
- `data/`, 기존 `results/`, `logs/`, `*.pt`, `*.npz`, 대형 모델 파일을 무조건 Git add하지 않는다.
- Git에서 제외하는 data/result/checkpoint에는 상대경로·크기·수정시각·가능하면 SHA256을 기록한 `docs/audit/v1_artifact_manifest.json`을 만든다. 이 manifest가 source snapshot과 기존 결과를 연결한다.
- source/config/test/doc를 검토한 뒤 다음 의미의 commit을 만든다.

```text
chore: snapshot CoF-SeqGen legacy v1 before benchmark redesign
```

- 해당 commit에 annotated tag를 만든다.

```text
legacy-kappa-v1
```

- 그 다음 새 branch `benchmark-v2-redesign`에서 작업한다.
- commit/tag/branch 생성 전 `git status --short`를 report에 남기며, 이미 사용자 branch 정책이 있거나 commit 권한이 없으면 임의 진행하지 말고 snapshot manifest까지 만든 뒤 blocker로 보고한다.
- 기존 자산을 이동하지 않는다. v2는 별도 경로를 사용한다.

### 5.4 v2 전용 경로

```text
data/benchmark_v2/
artifacts/benchmark_v2/
configs/benchmark_v2/
docs/benchmark_v2/
```

`artifacts/benchmark_v2/`는 Git에서 제외하되, 디렉터리 schema를 설명하는 README는 추적한다.

### Phase 0 acceptance criteria

- [ ] 기존 파일 삭제 없음
- [ ] preflight 문서 생성
- [ ] v1 asset inventory 생성
- [ ] 프로젝트 source snapshot commit 생성
- [ ] `legacy-kappa-v1` annotated tag가 snapshot commit을 가리킴
- [ ] Git 제외 artifact manifest 생성
- [ ] v2 전용 경로가 기존 결과와 분리됨
- [ ] Git status에서 의도하지 않은 대형 binary staging 없음

---

## 6. 목표 코드 구조

다음 구조를 만든다.

```text
benchmarks/
├── __init__.py
├── types.py
├── temporal_coupling_v2.py
├── semi_markov.py
├── binning.py
└── validation.py

generators/
├── __init__.py
├── base.py
├── sampling_plan.py
├── empirical_conditional_iid.py
├── empirical_conditional_block.py
├── conditional_ctgan.py
├── conditional_tvae.py
├── class_conditional_markov.py
├── joint_sequence_baseline.py
├── parametric_dgp_oracle.py
└── cof_seqgen_adapter.py

eval/
├── behavior_summaries_v2.py
├── coherence_v2.py
├── sequence_metrics.py
├── fidelity_v2.py
└── stats_v2.py

experiments/
├── __init__.py
├── artifact_store.py
├── seed.py
└── runner_v2.py

scripts/
├── generate_benchmark_v2.py
├── validate_benchmark_v2.py
├── run_v1_conditional_diagnostic.py
├── run_benchmark_v2.py
├── summarize_benchmark_v2.py
└── recompute_real_metrics_v2.py

configs/benchmark_v2/
├── main.yaml
├── smoke.yaml
├── baselines.yaml
└── cof.yaml

tests/
├── test_benchmark_v2_dgp.py
├── test_benchmark_v2_gate.py
├── test_soft_g_mask.py
├── test_generator_contract.py
├── test_sampler_padding.py
├── test_coherence_v2.py
└── test_artifact_store.py
```

### Module과 seam

#### Benchmark module interface

호출자가 알아야 하는 것은 다음뿐이어야 한다.

```python
bundle = generate_benchmark(config: BenchmarkConfig, seed: int) -> DatasetBundle
```

Markov state, stationary initialization, seed stream, emission 구현은 module 내부에 숨긴다.

#### Generator seam

실제로 여러 adapter가 필요하므로 명시적인 seam을 둔다.

```python
class SequenceGenerator(Protocol):
    name: str

    def fit(
        self,
        train: SequenceBatch,
        *,
        config: Mapping[str, Any],
        seed: int,
    ) -> None:
        ...

    def sample(
        self,
        plan: SamplingPlan,
        *,
        seed: int,
    ) -> SyntheticBatch:
        ...
```

모든 baseline과 CoF adapter는 이 interface를 만족해야 한다.

#### Evaluation interface

```python
reference = fit_coherence_reference(real_batch, config)
report = evaluate_generator(real_batch, synth_batch, reference, config)
```

모델별 평가 코드를 runner에 복제하지 않는다.

---

## 7. 공통 데이터 계약

`benchmarks/types.py`에 dataclass를 정의한다.

```python
@dataclass(frozen=True)
class SequenceBatch:
    x_num: np.ndarray       # float32 [N, L, d_num]
    dt_bin: np.ndarray      # int64   [N, L]
    x_cat: np.ndarray       # int64   [N, L, d_cat]
    valid_mask: np.ndarray  # bool    [N, L], True=valid
    y_entity: np.ndarray    # int64   [N], values {0,1}
    lengths: np.ndarray     # int64   [N]
    entity_ids: np.ndarray  # str/int [N]

    def y_position(self) -> np.ndarray:
        ...


@dataclass(frozen=True)
class SyntheticBatch:
    x_num: np.ndarray
    dt_bin: np.ndarray
    x_cat: np.ndarray
    valid_mask: np.ndarray
    y_entity: np.ndarray
    lengths: np.ndarray


@dataclass(frozen=True)
class DatasetBundle:
    train: SequenceBatch
    test: SequenceBatch
    metadata: Mapping[str, Any]
```

### 불변조건

각 dataclass 생성 시 검증한다.

- `valid_mask.sum(axis=1) == lengths`
- v2 canonical padding은 right padding
- 각 행에서 `valid_mask == [True...True, False...False]`
- padded `x_num`, `dt_bin`, `x_cat` 값은 0
- `y_entity.shape == (N,)`
- position label은 저장된 복제본보다 `y_entity`와 mask에서 파생하는 것을 우선
- 모든 array의 dtype과 shape 검증
- NaN/Inf 금지

현재 CoF가 position-level `y`를 요구하면 adapter 내부에서만 broadcast한다.

```python
y_position = y_entity[:, None] * valid_mask
```

---

## 8. Phase 1 — mask-aware sequence contract

## 8.1 `models/soft_g.py`

### 새 interface

`valid_mask`를 필수 keyword-only argument로 추가한다. 기본값 `None`을 두지 않는다. 모든 call site가 의식적으로 mask를 넘기게 한다.

```python
def soft_g(
    bin_probs: Tensor,
    cat_probs: Optional[Tensor],
    amt: Tensor,
    tau: Tensor,
    W: float,
    temp: float,
    *,
    valid_mask: Tensor,
) -> Tensor:
```

수정 전후에 다음 검색으로 모든 call site를 확인한다.

```bash
rg -n "soft_g\\(" .
rg -n "compute_g_from_real\\(" .
rg -n "compute_g_stats_from_data\\(" .
```

`models/`, `scripts/`, `tests/` 어디에도 `valid_mask` 없이 호출하는 경로가 남아 있으면 Phase 1 미완료다.

### 구현

```python
mask_f = valid_mask.to(bin_probs.dtype)
source_mask = mask_f[:, None, :]  # [B, 1, L], source position m
target_mask = mask_f[:, :, None]  # [B, L, 1], target position j
pair_mask = source_mask * target_mask

causal = torch.tril(
    torch.ones(L, L, device=bin_probs.device, dtype=bin_probs.dtype),
    diagonal=-1,
)

dt_hat = (bin_probs * tau_dev[None, None, :]).sum(-1)
dt_hat = dt_hat * mask_f
T = torch.cumsum(dt_hat, dim=1)
diff = T.unsqueeze(2) - T.unsqueeze(1)

w = torch.sigmoid((W - diff) / temp)
w = w * causal[None] * pair_mask

vel = w.sum(dim=2)
gap = dt_hat
...
g = torch.stack([vel, gap, rep, amt_sum], dim=-1)
g = g * mask_f[..., None]
```

필수 성질:

- padded source는 history에 포함되지 않음
- padded target의 모든 `g`는 0
- valid position의 gradient는 유지
- padded position의 값 변경이 valid `g`에 영향 없음
- left-pad와 right-pad에서 유효 sequence의 `g`가 같음

## 8.2 `models/teacher.py`

다음 함수에 `valid_mask`를 필수 전달한다.

- `compute_g_from_real`
- `compute_g_stats_from_data`
- `pretrain_teacher`

`compute_g_stats_from_data`는:

```python
g_all = torch.cat([g_b[mask_b] ...], dim=0)
```

처럼 valid position만 mean/std에 포함해야 한다.

`BehaviorTeacher.forward`도 mask를 받게 한다. v2 데이터는 right-pad이므로 `pack_padded_sequence`를 사용할 수 있다. 최소 요구사항:

- mask prefix-contiguous assert
- lengths를 이용해 padded position이 GRU state에 영향을 주지 않음
- output padded position은 0 또는 이후 loss에서 완전 제외

## 8.3 `models/cof_seqgen.py`

`soft_g` 호출을 다음처럼 수정한다.

```python
g_gen = soft_g(
    bin_probs,
    cat_probs,
    amt_hat,
    self.tau,
    self.W,
    self.temp,
    valid_mask=mask,
)
```

추가 수정:

- discrete corruption mask는 `valid_mask`와 AND
- padded token을 MASK token으로 바꾸지 않음
- 모든 loss는 valid position만 사용
- `use_mask = bool(mask.any().item())`처럼 명확하게 처리
- docstring의 label-loss weight를 실제 코드와 일치시킴

masked-only CE는 primary 변경으로 넣지 않는다. 다음 ablation용 config로만 노출한다.

```yaml
loss_positions: all_valid   # all_valid | corrupted_only
```

## 8.4 `models/seq_denoiser.py`

이미 `src_key_padding_mask`를 받으므로 유지하되 다음을 검증한다.

- dtype bool
- shape `[B,L]`
- `True=pad`
- output padded position을 downstream loss가 사용하지 않음

## 8.5 `models/sampler.py`

`ddim_sample`에 `valid_mask`를 필수 추가한다.

```python
def ddim_sample(
    ...,
    valid_mask: Tensor,
    ...
):
```

모든 denoiser forward에서:

```python
src_key_padding_mask=~valid_mask
```

를 전달한다.

`start_from_mask=True`일 때:

- valid position만 MASK token
- padded position은 0 유지
- 매 discrete feedback step 뒤 padded position을 0으로 복원
- numerical `x`와 `x_num_hat`도 padded position을 0으로 유지

```python
x = x * valid_mask[..., None]
dt_bin_cond = torch.where(valid_mask, dt_bin_cond, 0)
x_cat_cond = torch.where(valid_mask[..., None], x_cat_cond, 0)
```

### Phase 1 tests

`tests/test_soft_g_mask.py`:

1. `test_left_right_padding_invariance`
2. `test_padded_values_do_not_change_valid_g`
3. `test_padded_targets_are_zero`
4. `test_padding_source_excluded_from_velocity`
5. `test_padding_source_excluded_from_fanout`
6. `test_padding_source_excluded_from_amount_sum`
7. `test_valid_gradients_nonzero`
8. `test_padded_gradients_zero`
9. `test_all_valid_matches_legacy_behavior`

`tests/test_sampler_padding.py`:

1. denoiser가 모든 sampling step에서 padding mask를 받는지 spy/mock으로 검사
2. 합성 sample의 pad 값이 0인지 검사
3. length와 mask가 sampling plan과 정확히 일치하는지 검사

### Phase 1 acceptance criteria

```bash
pytest -q tests/test_soft_g_grad.py tests/test_soft_g_mask.py tests/test_sampler_padding.py
```

- [ ] 모든 테스트 PASS
- [ ] 기존 attached-gradient 테스트 유지
- [ ] legacy all-valid 입력의 수치가 변경되지 않음
- [ ] left/right padding valid `g` 일치

---

## 9. Phase 2 — benchmark v2 DGP

## 9.1 기본 config

`configs/benchmark_v2/main.yaml`:

```yaml
schema_version: benchmark_v2.0
name: temporal_coupling_suite_v2

scenarios:
  - markov_persistence_v2a
  - joint_semimarkov_v2b
primary_model_challenge_scenario: joint_semimarkov_v2b

data:
  n_train: 31951
  n_test: 7989
  sequence_length:
    min: 16
    max: 32
  fraud_rate: 0.05
  n_gap_bins: 16
  n_receiver_categories: 64
  padding_side: right
  window_width: 7.0
  soft_g_temperature: 1.0

coupling:
  kappas: [0.0, 0.3, 0.5, 0.7, 1.0]
  semantics_by_scenario:
    markov_persistence_v2a: normalized_persistence_contrast
    joint_semimarkov_v2b: normalized_cross_channel_synchronization

  gap_regime:
    burst_stationary_probability: 0.30
    normal_gap_scale: 2.0
    burst_gap_scale: 0.5
    rho_nonfraud: 0.0
    rho_fraud_max: 0.90

  receiver:
    rho_nonfraud: 0.0
    rho_fraud_max: 0.90
    stationary_distribution: uniform

  joint_semimarkov:
    burst_stationary_probability: 0.30
    duration:
      family: discrete_lognormal
      burst_target_mean: 6.0
      normal_target_mean: 14.0
      sigma: 0.60
      max_duration: 128
    receiver_repeat_probability:
      low: 0.05
      high: 0.90
    equilibrium_initialization: true
    shared_gap_receiver_state_for_fraud_at_kappa_1: true

  amount_log:
    mean: 5.0
    std: 1.0
    dependence: iid_label_independent

seeds:
  base: 42
  test_offset: 100000

binning:
  fit_split: train
  shared_across_kappa: true
  reference_kappa: 0.0

evaluation:
  primary_bins: 8
  confirmatory_bins: 4
  descriptive_bins: 2
  primary_channels_by_scenario:
    markov_persistence_v2a: [velocity, gap, fanout]
    joint_semimarkov_v2b: [joint_alignment]
  negative_control_channels_by_scenario:
    markov_persistence_v2a: [amount]
    joint_semimarkov_v2b: [velocity, gap, fanout, amount]
  negative_control_abs_dynamic_range_max: 0.005
  dynamic_range:
    hard_min_c1_minus_c0: 0.01
    target_c1_minus_c0: 0.02
```

`smoke.yaml`은 동일 구조에서:

```yaml
data:
  n_train: 512
  n_test: 256
```

로 축소한다.

## 9.2 v2a `markov_persistence`에서 κ의 의미

v1처럼 “fraud entity 중 fraud-mode 비율”로 정의하지 않는다.

```python
rho_gap(y=0) = rho_gap_nonfraud
rho_gap(y=1) = rho_gap_nonfraud + kappa * (rho_gap_fraud_max - rho_gap_nonfraud)

rho_cat(y=0) = rho_cat_nonfraud
rho_cat(y=1) = rho_cat_nonfraud + kappa * (rho_cat_fraud_max - rho_cat_nonfraud)
```

기본값에서는:

```text
y=0: rho=0
y=1: rho=0.9*κ
```

κ=0이면 두 라벨 모두 i.i.d.이고, κ=1이면 fraud만 강한 persistence를 가진다.

## 9.3 v2a gap regime Markov process

숨은 상태:

```text
z_t = 0: normal
z_t = 1: burst
```

모든 라벨과 κ에서 stationary burst probability `π=0.30`을 동일하게 유지한다.

주어진 `π`, `rho`에 대해:

```python
P(N -> B) = (1 - rho) * pi
P(N -> N) = 1 - (1 - rho) * pi
P(B -> N) = (1 - rho) * (1 - pi)
P(B -> B) = pi + (1 - pi) * rho
```

이 전이행렬의 stationary distribution은 정확히 `[1-π, π]`다.

초기 상태:

```python
z_0 ~ Bernoulli(pi)
```

초기 상태를 항상 normal로 두지 않는다. 그렇지 않으면 초기 위치에서 라벨/길이 관련 누출이 생길 수 있다.

Emission:

```python
gap_t | z_t=0 ~ Exponential(scale=2.0)
gap_t | z_t=1 ~ Exponential(scale=0.5)
```

첫 gap도 같은 emission에서 표본한다. v1의 `gaps_i[0]=2.0` 같은 예외를 두지 않는다.

결과:

- 모든 라벨에서 한 시점의 gap 주변분포는 동일한 mixture
- fraud에서는 burst/normal이 시간적으로 뭉침
- 라벨 차이는 transition/autocorrelation에만 존재

## 9.4 v2a receiver persistence

`K=64`, 초기 receiver:

```python
c_0 ~ Uniform({0, ..., K-1})
```

전이:

```python
if rng.random() < rho:
    c_t = c_{t-1}
else:
    c_t = rng.integers(0, K)
```

이는 다음 전이확률과 같다.

```text
P(c_t=j | c_{t-1}=i) = rho*1[j=i] + (1-rho)/K
```

모든 라벨에서 stationary marginal은 정확히 uniform이다. fraud에서만 run length와 repeat rate가 증가한다.

category 0을 특정 라벨 전용으로 사용하지 않는다.

## 9.5 amount

직접 log-amount를 표본한다.

```python
amount_log_t ~ Normal(5.0, 1.0)
```

- label, κ, hidden state와 독립
- negative control channel
- 양 class에서 같은 RNG mechanism

## 9.6 sequence length와 label

```python
y_entity ~ Bernoulli(0.05)
length ~ DiscreteUniform(16, 32)
```

length는 label과 독립이어야 한다.

모든 sequence는 right-pad한다.

## 9.7 seed stream

한 함수 안의 RNG 소비 순서가 바뀌어 전체 데이터가 조용히 바뀌지 않도록 `numpy.random.SeedSequence`로 stream을 분리한다.

예:

```python
root_ss = np.random.SeedSequence([base_seed, split_id])
label_ss, length_ss, gap_state_ss, gap_emit_ss, cat_ss, amount_ss = root_ss.spawn(6)
```

κ에 독립이어야 하는 label/length는 모든 κ에서 같은 seed stream을 사용한다.

κ에 따라 달라지는 transition sample은 κ id를 포함한 별도 stream을 사용한다.

metadata에 모든 child seed entropy/spawn key를 기록한다.

## 9.8 binning

- gap bin edge는 test를 사용하지 않고 train에서만 fit
- κ 간 비교를 위해 κ=0 train의 valid raw gaps로 한 번 fit
- 동일 bin edges와 `tau_k`를 모든 κ에 사용
- `binning.json`을 별도 저장
- train/test 모두 동일 binning 사용
- duplicate edge 처리와 effective bin 수 기록

## 9.9 저장 schema

```text
data/benchmark_v2/
├── benchmark_manifest.json
├── markov_persistence_v2a/
│   ├── binning.json
│   ├── kappa_0.00/
│   │   ├── train.npz
│   │   ├── test.npz
│   │   ├── meta.json
│   │   ├── train_audit_latent.npz
│   │   └── test_audit_latent.npz
│   └── ...
└── joint_semimarkov_v2b/
    ├── binning.json
    ├── kappa_0.00/
    └── ...
```

학습용 NPZ:

- `x_num`
- `dt_bin`
- `x_cat`
- `valid_mask`
- `y_entity`
- `lengths`
- `entity_ids`

감사용 latent NPZ:

- raw gap
- gap regime state
- receiver repeat indicator
- transition parameters

감사용 latent는 학습 loader가 읽지 못하도록 별도 파일로 둔다.

## 9.10 analytical assertions

`benchmarks/temporal_coupling_v2.py` 내부에서:

```python
stationary @ transition == stationary
transition.sum(axis=1) == 1
0 <= probabilities <= 1
```

를 tolerance와 함께 검사한다.

## 9.11 2단 benchmark suite

### Scenario v2a — `markov_persistence_v2a`

앞 절의 stationary 1차 Markov 설계를 그대로 사용한다.

목적:

- row-marginal equivalence 검증
- κ↔persistence 단조성 검증
- coherence metric과 gate calibration
- empirical i.i.d., learned Markov, DGP oracle의 예상 ordering 검증

주의:

- gap 관측값은 숨은 상태의 emission이므로 관측 `dt_bin` 자체가 엄밀히 1차 Markov일 필요는 없다.
- 그래도 low-order Markov/HMM에 매우 유리한 scenario다.
- learned Markov/HMM baseline이 CoF와 동률 또는 우수해도 이 scenario의 calibration 목적에는 부합한다.
- v2a만으로 CoF 우월성을 주장하지 않는다.

### Scenario v2b — `joint_semimarkov_v2b`

v2b는 primary model-challenge scenario다. 목표는 다음 세 조건을 동시에 만족하는 것이다.

1. 단일 행 marginal은 라벨 간 동일
2. 각 개별 채널의 marginal과 기본 시간구조도 가능한 한 동일
3. fraud label 신호는 gap burst와 receiver repeat가 **같은 장기 상태에 공동 결속되는가**에 존재

#### 두 개의 동일분포 semi-Markov state process

각 entity에 대해 독립적으로:

```text
z_gap[t]       ∈ {normal, burst}
z_receiver[t]  ∈ {normal, burst}
```

를 생성한다.

- 두 process는 동일한 stationary occupancy `π=0.30`
- 동일한 duration family와 parameter
- 서로 독립
- duration은 geometric이 아니라 discrete log-normal

상태는 normal/burst가 번갈아 나타나는 alternating renewal process다.

기본 target mean:

```text
E[D_burst]  = 6
E[D_normal] = 14
π_burst = 6 / (6 + 14) = 0.30
```

duration:

```python
D_s = clip(round(exp(Normal(mu_s, sigma=0.60))), 1, 128)
```

Monte Carlo histogram으로 PMF를 근사하지 않는다. `d=1..D_max`에 대해 log-normal CDF 구간:

```text
P(D=d) = P(d-0.5 <= exp(X) < d+0.5)
```

를 계산하고, 1 미만 tail은 `d=1`, `D_max` 초과 tail은 `d=D_max`에 포함해 finite discrete PMF를 만든다. `scipy.stats.lognorm.cdf`를 사용할 수 있고, dependency를 추가하지 않으려면 normal CDF를 `erf`로 구현한다.

truncation/rounding 뒤 PMF의 실제 mean이 target mean과 맞도록 `mu_s`를 bisection으로 보정한다. 보정된 `mu`, discrete PMF, 실제 mean, truncation mass를 metadata에 저장한다.

#### 정확한 equilibrium initialization

항상 새 duration의 시작점에서 sequence를 자르면 초기 transient가 생긴다. 이를 금지한다.

초기 상태:

```text
P(z_0=burst) = π
```

초기 상태의 전체 duration `D*`는 length-biased PMF에서 표본한다.

```text
P(D*=d | state=s) ∝ d · P(D=d | state=s)
```

그 duration 안의 age를:

```text
age ~ DiscreteUniform(0, D*-1)
residual = D* - age
```

로 표본한 뒤 남은 residual부터 sequence를 생성한다. 이후에는 원래 duration PMF를 사용해 상태를 교대한다.

이 구현은 잔여시간을 직접 다음 분포에서 표본하는 것과 동등해야 한다.

```text
P(R=r | state=s) = P(D_s >= r) / E[D_s],   r=1,2,...
```

증명상:

```text
P(D*=d) ∝ dP(D=d)
age | D*=d ~ Uniform{0,...,d-1}
R = d-age
```

이면 위 inspection-paradox residual distribution이 나온다.

구현은 다음 두 방식 중 하나만 source of truth로 선택한다.

1. length-biased total duration + uniform age
2. survival-function 기반 residual PMF 직접 표본

두 방식을 동시에 섞지 않는다. 선택한 방식과 PMF를 metadata에 기록하고, unit test에서는 다른 방식으로 계산한 theoretical residual PMF와 일치하는지 검증한다.

대안으로 burn-in 후 crop을 구현할 수 있지만, primary 구현은 위 equilibrium-residual 방식을 사용한다. analytical occupancy test를 반드시 둔다.

#### 위치별 정상성 검증

전체 평균 occupancy만 검사하면 초반 transient를 놓칠 수 있다. `validate_benchmark_v2.py`는 위치 `t=0,...,L-1`별로 다음을 계산한다.

```text
P(z_gap[t]=burst | y=0)
P(z_gap[t]=burst | y=1)
P(z_receiver[t]=burst | y=0)
P(z_receiver[t]=burst | y=1)
```

검사:

- 각 위치의 burst occupancy가 target π에 가까운가
- 같은 위치에서 y=0/y=1 차이가 practical tolerance 안인가
- early positions와 late positions 사이에 체계적 drift가 없는가
- position별 observed gap marginal과 receiver category marginal도 라벨 간 같은가

기본 config:

```yaml
position_stationarity:
  occupancy_target_tolerance: 0.03
  between_label_tolerance: 0.03
  simultaneous_ci: entity_bootstrap
```

finite-sample에서 한 위치가 우연히 tolerance를 넘을 수 있으므로 max difference만 보고 즉시 parameter를 바꾸지 않는다. simultaneous bootstrap band, drift trend, 인접 위치 pattern을 함께 보고한다. 체계적 early-position bias가 있으면 FAIL이다.

#### gap emission

모든 라벨에서 동일:

```python
gap_t | z_gap[t]=normal ~ Exp(2.0)
gap_t | z_gap[t]=burst  ~ Exp(0.5)
```

따라서 gap sequence의 marginal과 semi-Markov duration 구조 자체는 라벨 간 동일하다.

#### receiver repeat driver

receiver category는 항상 uniform stationary가 되도록:

```python
if rng.random() < q_t:
    receiver_t = receiver_{t-1}
else:
    receiver_t = rng.integers(0, K)
```

를 사용한다.

non-fraud:

```python
driver_t = z_receiver[t]
q_t = q_low + (q_high - q_low) * driver_t
```

fraud:

```python
driver_strength_t = (
    (1 - kappa) * z_receiver[t]
    + kappa * z_gap[t]
)
q_t = q_low + (q_high - q_low) * driver_strength_t
```

기본값:

```text
q_low=0.05
q_high=0.90
```

해석:

- κ=0: fraud/non-fraud 모두 receiver repeat driver가 gap state와 독립
- κ=1: fraud에서 receiver repeat가 gap burst와 같은 semi-Markov state에 결속
- 모든 κ에서 `E[driver_strength]=π`이므로 평균 repeat probability는 동일
- receiver category의 한 시점 marginal은 uniform
- gap의 한 시점 marginal도 동일
- label 신호는 `short gap ↔ receiver repeat/run`의 cross-channel temporal alignment에 존재

κ∈(0,1)에서 `driver_strength`가 연속값이 되는 것은 의도된 interpolation이다. metadata에 κ 의미를 `normalized_cross_channel_synchronization`으로 기록한다.

#### v2b가 단순 Markov보다 어려운 이유

- 상태 duration이 non-geometric이므로 단순 1차 observed transition만으로 duration tail을 정확히 표현하기 어려움
- gap과 receiver를 독립적으로 fit한 channel-wise Markov generator는 공동 상태 alignment를 잃음
- 한편 correctly specified joint HSMM은 강력한 baseline이므로, CoF가 모든 parametric oracle을 이겨야 한다고 주장하지 않음

#### v2b strong baselines

최소 다음을 분리한다.

1. `channel_independent_markov`: gap/receiver transition을 y별 독립 fit
2. `joint_observed_markov`: `(dt_bin, receiver_repeat)` joint transition을 y별 fit
3. 가능하면 `joint_hmm_or_hsmm`: hidden-state sequence baseline
4. neural sequence baseline

CoF 모델 기여는:

- row-independent baseline보다 우수한가
- channel-independent Markov보다 cross-channel coupling을 잘 보존하는가
- joint Markov/neural baseline과 비교해 어떤 trade-off를 보이는가

로 기술한다. correctly specified HSMM oracle을 반드시 이겨야만 기여가 있다는 식으로 판정하지 않는다.

## 9.12 scenario별 κ와 산출물 분리

data path:

```text
data/benchmark_v2/
├── markov_persistence_v2a/
│   └── kappa_*/
└── joint_semimarkov_v2b/
    └── kappa_*/
```

artifact path도 동일하게 scenario를 포함한다.

```text
artifacts/benchmark_v2/runs/<scenario>/kappa_*/<generator>/seed_*/
```

두 scenario 결과를 같은 CSV row로 합칠 때 `scenario` column을 필수로 넣는다.

### Phase 2 tests

`tests/test_benchmark_v2_dgp.py`:

1. `test_transition_rows_sum_to_one`
2. `test_stationary_distribution_exact`
3. `test_kappa_zero_same_transition_for_labels`
4. `test_kappa_controls_persistence_not_marginal`
5. `test_receiver_stationary_uniform`
6. `test_amount_label_independent_by_construction`
7. `test_length_label_independent_by_construction`
8. `test_initial_state_stationary`
9. `test_right_padding_contract`
10. `test_train_test_seed_spaces_disjoint`
11. `test_generation_deterministic_for_same_seed`
12. `test_generation_changes_for_different_seed`
13. `test_shared_binning_across_kappa`
14. `test_no_fraud_exclusive_category`
15. `test_semimarkov_duration_mean_calibration`
16. `test_semimarkov_equilibrium_occupancy`
17. `test_semimarkov_residual_matches_survival_formula`
18. `test_semimarkov_no_early_position_occupancy_drift`
19. `test_positionwise_occupancy_equal_across_labels`
20. `test_v2b_row_marginals_equal_across_labels`
21. `test_v2b_channel_univariate_dynamics_match_at_kappa_one`
22. `test_v2b_cross_channel_alignment_increases_with_kappa`
23. `test_scenario_paths_are_disjoint`

### Phase 2 acceptance criteria

```bash
pytest -q tests/test_benchmark_v2_dgp.py
python -m scripts.generate_benchmark_v2 --config configs/benchmark_v2/smoke.yaml
```

- [ ] 모든 analytical test PASS
- [ ] smoke data schema PASS
- [ ] κ=0에서 두 라벨 transition 동일
- [ ] κ 증가 시 lag-1 persistence 증가
- [ ] 한 행의 theoretical marginal은 동일
- [ ] v2b equilibrium occupancy가 π와 일치
- [ ] 위치별 occupancy에 early transient가 없음
- [ ] v2b에서 cross-channel alignment만 κ에 따라 증가

---

## 10. Phase 3 — benchmark validity gate

`benchmarks/validation.py`와 `scripts/validate_benchmark_v2.py`를 구현한다.

## 10.1 Gate A — single-row leakage

유효 행만 펼치되 entity ID를 유지한다.

검사:

### Continuous marginal

- amount: two-sample KS statistic
- raw gap: two-sample KS statistic

### Categorical marginal

- receiver TVD
- `dt_bin` TVD

### Position-wise marginal/stationarity

- 위치별 hidden burst occupancy
- 위치별 raw-gap KS 또는 practical distance
- 위치별 receiver category TVD
- early/middle/late segment별 marginal
- 라벨 간 위치별 difference와 simultaneous entity-bootstrap band

전체 행을 합친 marginal이 같아도 t=0 부근에만 transient leakage가 있을 수 있으므로 별도 gate로 저장한다.

### Joint single-row classifier

한 행의:

- `amount_log`
- `dt_bin`
- `receiver`

만으로 y를 예측한다.

최소 두 classifier:

1. one-hot + logistic regression
2. shallow tree/HistGradientBoosting

train entities의 행으로 학습하고 test entities의 행으로 평가한다. 행을 임의 분할하지 않는다.

entity-level bootstrap으로 AUROC CI를 계산한다.

### Positive control: pair/context classifier

single-row classifier가 0.5라는 것만으로는 “신호가 제거된 것”과 “신호가 순차 context로 이동한 것”을 구분할 수 없다.

동일 script에서 연속 context classifier를 실행한다.

#### Pair classifier

연속 두 행에서:

- `gap_{t-1}`, `gap_t`
- `receiver_{t-1}`, `receiver_t`
- `receiver_repeat = 1[receiver_t=receiver_{t-1}]`
- 두 행의 amount
- cross feature `short_gap_t × receiver_repeat`

를 구성한다.

entity-disjoint train/test를 사용하고, 같은 entity의 pair가 양쪽에 섞이지 않게 한다.

#### Context-length classifier

context length:

```text
m ∈ {1, 2, 4, 8}
```

에 대해 고정된 간단한 classifier 또는 동일 architecture를 사용해 AUROC curve를 만든다.

필수 해석:

- m=1: AUROC practical-equivalent to 0.5
- v2a: m=2부터 positive-control signal
- v2b: `receiver_repeat`와 `short_gap`의 공동 정렬은 인접 두 행에서 정의되므로 **m=2부터 positive-control signal이 검출되어야 함**
- m∈{1,2,4,8} 전체 AUROC와 entity-bootstrap CI를 하나의 curve로 보고
- 더 긴 context가 persistence를 누적해 AUROC를 높이는지, 또는 m=2 이후 포화되는지 정량 보고

v2b의 기본 positive-control hard gate:

```text
AUROC_context1 95% CI가 practical-equivalence 구간 [0.48, 0.52] 안에 존재
AUROC_context2 95% CI lower bound > 0.52
AUROC_context2 point estimate target >= 0.55
```

v2b에서 m=2가 실패하고 m=4/8만 통과하면 benchmark가 원래 정의한 “공동 정렬은 정확히 인접 pair부터 관측 가능하다”는 positive control과 불일치하므로 Gate A FAIL이다. 이 경우 우선 pair feature, cutoff 적용, receiver transition indexing, mask를 검사한다.

창 길이에 따른 **엄격한 표본 AUROC 단조 증가**는 hard gate로 두지 않는다. Bayes-optimal classifier는 더 긴 창에서 짧은 창의 정보를 무시할 수 있지만, 유한 표본의 고정 classifier AUROC와 bootstrap CI는 작은 역전을 보일 수 있기 때문이다. 대신 다음을 적용한다.

```text
AUROC_context4 >= AUROC_context2 - 0.02
AUROC_context8 >= AUROC_context2 - 0.02
```

그리고 m=1→2의 유의한 상승, m=2 이후의 추세 또는 포화를 모두 보고한다. 0.02보다 큰 지속적 하락은 구현 오류, overfitting 또는 classifier capacity 문제로 진단하고 자동 PASS시키지 않는다.

### Gate A 기본 tolerance

single-row equivalence는 모든 κ에서:

```text
KS(amount) <= 0.02
KS(raw_gap) <= 0.02
TVD(receiver) <= 0.02
TVD(dt_bin) <= 0.02
abs(single_row_AUROC - 0.5) <= 0.02
single_row_AUROC 95% CI 전체가 practical-equivalence 구간 [0.48, 0.52] 안에 존재
position-wise occupancy에 체계적 early/late drift 없음
```

positive-control은 κ=1에서:

```text
v2a: 사전 등록한 pair/context positive-control의 95% CI lower bound > 0.52
v2b: context2의 95% CI lower bound > 0.52
target point estimate >= 0.55
```

이어야 한다. κ=0에서는 pair/context classifier도 0.5여야 하며, κ 증가에 따라 context signal이 단조 증가하는지 별도 보고한다. κ별 경험적 AUROC의 작은 국소 역전만으로 실패시키지는 않되, κ=0 대비 κ=1 effect와 순서 제약 trend test를 함께 보고한다.

표본 수 때문에 p-value는 매우 작을 수 있으므로 p-value만으로 실패시키지 않는다. practical effect size를 기준으로 한다.

Gate가 간헐적으로 실패하면 seed를 바꾸기 전에 population implementation과 seed stream을 확인한다.

## 10.2 Gate B — conditional i.i.d. bootstrap

`generators/empirical_conditional_iid.py`를 가장 먼저 구현한다.

절차:

1. train valid rows를 `y_entity`별 pool로 분리
2. evaluation label과 length를 먼저 고정
3. 각 sequence의 각 valid 위치를 같은 y row pool에서 독립 복원추출
4. 순서·transition·entity state를 사용하지 않음
5. seed 1–10 실행

2/4/8-bin 모두 평가한다.

- 8-bin: primary gate
- 4-bin: confirmatory gate
- 2-bin: descriptive continuity check

Gate PASS/FAIL은 4-bin과 8-bin을 기준으로 한다. 2-bin은 반드시 보고하지만, median split이 tail/variance 신호를 뭉갤 수 있으므로 단독 판정에 사용하지 않는다.

정의:

```python
improvement_over_c1 = C1_gap - iid_gap
equivalence_margin = max(0.005, 0.15 * C1_gap)
```

각 κ>0, 4-bin과 8-bin의 scenario primary channel에 대해:

- v2a: velocity/gap/fanout
- v2b: joint_alignment

```text
95% CI upper bound of improvement_over_c1 <= equivalence_margin
```

이어야 한다.

단순히 “p>0.05”를 equivalence 증거로 사용하지 않는다.

2-bin 결과가 4/8-bin과 방향상 크게 충돌하면 gate를 자동 통과시키지 말고 bin occupancy와 distribution tail을 진단한다.

## 10.3 Gate C — parametric DGP oracle

`generators/parametric_dgp_oracle.py`:

- real train 행 재사용 금지
- 동일 DGP parameter로 fresh sequence 생성
- evaluation sampling plan의 label/length를 사용
- 새로운 emission/transition randomness 사용

정의:

```python
oracle_excess = oracle_gap - C0_gap
tolerance = max(0.005, 0.15 * C1_gap)
```

각 κ와 4/8-bin scenario primary endpoint에서:

```text
95% CI upper bound of oracle_excess <= tolerance
```

이어야 한다.

## 10.4 Gate D — coupling strength

real benchmark 자체에서:

- receiver repeat rate의 class difference
- receiver maximum run length의 class difference
- gap lag-1 autocorrelation의 class difference
- velocity/fanout coherence signal

이 κ에 따라 증가해야 한다.

기본 acceptance:

```text
Spearman correlation(κ, C1-C0 structural gap) >= 0.90
```

v2a의 amount negative-control channel은 κ에 따라 체계적으로 증가하면 안 된다.

v2b에서는 velocity/gap/fanout/amount 네 channel이 모두 설계 검증용 negative control이다. 각 channel의 4/8-bin `abs(C1-C0)` point estimate가 기본 0.005 이하이고, κ에 따른 증가 trend가 없어야 한다. 0.005는 pilot Monte Carlo와 N을 확정하기 전에 config에 사전 등록하며, CoF/baseline 결과를 본 뒤 늘리지 않는다. primary `joint_alignment`만 아래 dynamic-range gate를 통과해야 한다.

### Dynamic-range/power gate

자기상관·공동 결속 신호는 v1의 강한 marginal 분리보다 작을 수 있다. 최대 κ에서 측정 가능한 범위를 확보한다.

4-bin과 8-bin scenario primary endpoint 각각에 대해:

```text
dynamic_range = C1 - C0
```

판정:

```text
dynamic_range < 0.01       → FAIL: 현재 N/π/ρ/W/L에서 측정 불가
0.01 <= range < 0.02       → WEAK: power analysis 후 N 증가를 우선 검토
dynamic_range >= 0.02      → PASS target
```

`WEAK` 상태는 overall gate PASS가 아니다. 다음 순서로만 조정한다.

1. metric/bin occupancy와 구현 오류 확인
2. N 증가 power analysis
3. W/L이 정의된 순차 현상을 포착하는지 확인
4. 그 후에만 π, `rho_max`, duration, repeat probability 범위 검토

조정에는 DGP oracle, C0/C1, pair/context classifier만 사용한다. CoF 또는 baseline 성능을 보고 DGP parameter를 조정하지 않는다.

parameter 변경 시 config version을 올리고 이전 gate artifact를 보존한다.

## 10.5 Gate E — 2/4/8-bin robustness

다음 모두 저장한다.

- bin thresholds
- real/synthetic bin occupancy
- bin별 fraud count
- bin별 prevalence
- macro gap
- real-occupancy-weighted gap
- effective bin count

empty bin을 `p=0`으로 조용히 처리하지 않는다.

기본 정책:

```yaml
rate_smoothing:
  prior_alpha: 0.5
  prior_beta: 0.5
minimum_bin_count: 25
empty_bin_policy: invalid
```

minimum count 미만이면 해당 run을 invalid로 표시하고 원인을 보고한다.

판정 체계:

- 8-bin primary
- 4-bin confirmatory
- 2-bin descriptive
- 4-bin과 8-bin 중 하나라도 핵심 equivalence/oracle gate에 실패하면 overall FAIL
- 2-bin만 실패하거나 약할 경우 median-split resolution 문제로 진단하되 결과는 숨기지 않음

## 10.6 gate output

```text
artifacts/benchmark_v2/gates/
├── gate_report.json
├── gate_report.md
├── row_marginals.csv
├── positionwise_stationarity.csv
├── positionwise_stationarity.png
├── single_row_classifier.csv
├── context_classifier_curve.csv
├── iid_bootstrap_2bin.csv
├── iid_bootstrap_4bin.csv
├── iid_bootstrap_8bin.csv
├── dgp_oracle.csv
├── sequential_statistics.csv
└── plots/
```

`gate_report.json`:

```json
{
  "schema_version": "benchmark_gate_v2.0",
  "overall_status": "PASS",
  "gate_a_single_row_leakage": "PASS",
  "gate_a_positionwise_stationarity": "PASS",
  "gate_a_context_positive_control": "PASS",
  "gate_b_iid_equivalence_to_c1": "PASS",
  "gate_c_dgp_oracle_near_c0": "PASS",
  "gate_d_monotonic_coupling": "PASS",
  "gate_d_dynamic_range": "PASS",
  "gate_e_bin_robustness": "PASS"
}
```

### Phase 3 실행

```bash
python -m scripts.generate_benchmark_v2 \
  --config configs/benchmark_v2/main.yaml

python -m scripts.validate_benchmark_v2 \
  --config configs/benchmark_v2/main.yaml \
  --data-root data/benchmark_v2 \
  --output-root artifacts/benchmark_v2/gates
```

### Phase 3 tests

`tests/test_benchmark_v2_gate.py`에 최소한 다음을 구현한다.

1. `test_single_row_split_is_entity_disjoint`
2. `test_single_row_auc_equivalence_rule`
3. `test_positionwise_stationarity_detects_naive_duration_restart`
4. `test_context_windows_never_cross_entity_or_padding`
5. `test_v2b_context_one_has_no_pair_feature`
6. `test_v2b_context_two_contains_short_repeat_cross_feature`
7. `test_v2b_context_two_positive_control_rule`
8. `test_longer_context_non_degradation_tolerance`
9. `test_iid_bootstrap_equivalence_rule`
10. `test_dgp_oracle_rule`
11. `test_v2b_channel_negative_control_rule`
12. `test_primary_dynamic_range_rule`
13. `test_gate_report_fails_closed_on_missing_metric`

classifier용 threshold/cutoff/scaler는 train entity에서만 fit되어야 하며, test statistic이나 label별 quantile을 참조하면 test가 실패해야 한다.

### Phase 3 acceptance criteria

- [ ] Gate A PASS
- [ ] equilibrium residual과 위치별 stationarity test PASS
- [ ] Gate A positive-control context signal PASS
- [ ] v2b context1 equivalence 및 context2 positive-control hard gate PASS
- [ ] Gate B PASS for 4/8 bins; 2-bin reported
- [ ] Gate C PASS
- [ ] Gate D monotonicity and dynamic-range PASS
- [ ] v2b channel-wise negative-control dynamic range PASS
- [ ] Gate E PASS
- [ ] `overall_status=PASS`

PASS 전에는 Phase 6의 CoF full training 금지.

---

## 11. Phase 4 — 공통 generator interface와 baseline

## 11.1 `generators/sampling_plan.py`

모든 generator가 동일한 label과 length를 사용하도록 plan을 한 번 생성·저장한다.

```python
@dataclass(frozen=True)
class SamplingPlan:
    y_entity: np.ndarray
    lengths: np.ndarray
    valid_mask: np.ndarray
    plan_id: str
```

plan은 real test의:

- exact label vector를 그대로 쓰거나
- 동일 prevalence로 stratified fixed sample

중 하나를 config에서 선택한다. primary는 **real test의 exact label/length vector**를 권장한다.

각 `(kappa, evaluation_seed)`에서 plan을 생성하고 모든 generator가 같은 파일을 읽는다.

```text
artifacts/benchmark_v2/sampling_plans/kappa_1.00/seed_001.npz
```

## 11.2 empirical conditional i.i.d.

이 adapter는:

- real row를 재사용하는 empirical oracle diagnostic
- 학습형 generator가 아님
- 결과 표에서 별도 구역에 표시

v2에서 C1 근처에 있어야 한다.

## 11.2.1 empirical conditional block-bootstrap ladder

`generators/empirical_conditional_block.py`를 구현한다.

목적:

> 어느 정도의 연속 context를 보존해야 real coherence에 접근하는가?

block length:

```text
b ∈ {1, 2, 4, 8, full}
```

절차:

1. target label을 sampling plan에서 먼저 읽음
2. 같은 label의 train entity에서 contiguous block을 표본
3. target length가 찰 때까지 block을 연결
4. entity 경계를 넘는 block 금지
5. `b=1`은 empirical conditional i.i.d.와 수치적으로 동등해야 함
6. `full`은 가능한 경우 하나의 real sequence/window를 재표본하는 상한 참조선

산출물:

```text
artifacts/benchmark_v2/gates/context_ladder/
├── block_01/
├── block_02/
├── block_04/
├── block_08/
├── block_full/
└── context_length_curve.csv
```

보고:

- 4/8-bin structural coherence gap
- direct sequence metrics
- `C1-gap` improvement
- context length에 따른 curve

이 ladder는 empirical row/sequence를 재사용하는 diagnostic이며 learned generator table과 분리한다.

`IMPLEMENT_AND_SMOKE`에서도 b=1/2/4/8/full CPU diagnostic을 구현·실행한다.

## 11.3 conditional CTGAN

primary 구현은 class-specific model 두 개다.

```text
CTGAN_y0.fit(rows where y=0)
CTGAN_y1.fit(rows where y=1)
```

sampling:

- plan에서 y와 length를 먼저 읽음
- y별 필요한 총 row 수 계산
- 해당 class model에서 정확한 수만큼 sample
- 각 sequence에 독립 조립
- max-over-L label 생성 금지

모델 입력 column:

- amount_log
- dt_bin
- receiver

entity ID, position, latent state는 입력 금지.

`dt_bin`, `receiver`는 discrete column으로 지정한다.

seed 고정:

```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

CTGAN version이 `set_random_state`를 제공하면 함께 사용한다. 실제 적용 여부를 manifest에 기록한다.

## 11.4 conditional TVAE

CTGAN과 동일하게 class-specific model 두 개를 사용한다.

동일:

- epoch
- batch size
- train row pool
- sampling plan
- seed policy

## 11.5 learned class-conditional Markov baseline

`generators/class_conditional_markov.py`:

- `dt_bin` initial distribution과 transition matrix를 y별 추정
- receiver initial distribution과 transition matrix를 y별 추정
- Laplace smoothing config
- amount는 y별 empirical distribution 또는 Gaussian fit
- length/label은 sampling plan 사용

이 baseline은 sequence-aware이며 중요하다. v2a에서는 Markov가 강한 것이 예상되므로 CoF 패배만으로 C2를 기각하지 않는다. 그러나 primary v2b에서도 CoF가 row baseline만 이기고 모든 일반 sequence baseline보다 열세라면 C2 model contribution은 크게 축소해야 한다.

config:

```yaml
markov:
  laplace_alpha: 1.0
  amount_model: empirical
```

scenario별로 최소 두 형태를 보고한다.

### Channel-independent Markov

- gap/dt_bin transition과 receiver transition을 별도 fit
- 두 channel을 조건부 독립으로 sample
- v2b의 cross-channel alignment를 의도적으로 모델링하지 않음

### Joint observed Markov

- observable state를 최소한 `(dt_bin_t, receiver_repeat_t)`의 joint state로 구성
- 전체 receiver category×dt_bin Cartesian product를 무작정 사용하지 않음
- sparse transition smoothing과 unseen-state fallback을 명시

v2a에서 Markov가 oracle에 근접하는 것은 예상 가능한 calibration 결과다. v2b에서는 channel-independent와 joint Markov의 차이가 cross-channel coupling 난이도를 보여준다.

## 11.6 추가 sequence baseline

정식 논문 실험에서는 Markov 외에 최소 하나의 learned neural sequence baseline을 추가한다.

우선순위:

1. 기존 환경에서 재현 가능한 CPAR 또는 autoregressive sequence model
2. 간단한 class-conditional GRU/Transformer autoregressive generator
3. 설치·재현이 가능하다면 TimeGAN 계열

이 작업은 benchmark/CoF 코드를 침범하지 않는 별도 adapter로 구현한다.

v2b의 non-geometric duration을 직접 겨냥하는 HMM/HSMM을 추가할 수 있으면 별도 strong baseline으로 보고한다. 이 모델은 correctly specified에 가까울 수 있으므로 CoF가 반드시 이겨야 한다는 사전 조건을 두지 않는다.

최소 interface:

- entity label 조건
- mask/length 지원
- mixed numerical/categorical output
- sample 저장

`IMPLEMENT_AND_SMOKE`에서는 Markov까지 필수, neural sequence baseline은 skeleton+smoke까지 허용한다. `FULL_EXPERIMENT`에서는 하나 이상의 neural sequence baseline full run이 필요하다.

## 11.7 v1 conditional diagnostic

`scripts/run_v1_conditional_diagnostic.py`를 별도로 만든다.

실행 대상:

- empirical conditional i.i.d.
- conditional CTGAN
- conditional TVAE

목적:

- v1의 feature-only baseline이 불공정했음을 정량 확인
- v2 main table에 병합하지 않음
- `artifacts/v1_conditional_diagnostic/`에만 저장

### 고정 예산

- hyperparameter sweep 금지
- 기존 coherence baseline과 맞는 단일 고정 config 사용
- empirical i.i.d.는 5–10 sample seed
- CTGAN/TVAE는 우선 1 model seed씩, 시간이 허용되면 최대 3 seed
- 총 투자 상한: **한 근무일 또는 8 GPU-hours 중 먼저 도달하는 시점**
- technical failure가 상한 안에 해결되지 않으면 실패 원인과 로그를 보존하고 종료

예상대로 conditional CTGAN/TVAE가 feature-only v1 baseline보다 크게 개선되어도:

- v2 main performance table에 넣지 않음
- “v1 benchmark confounding diagnostic”으로만 사용
- CoF 우월성 근거로 사용하지 않음
- 추가 튜닝으로 성능 최고치를 찾지 않음

### Phase 4 tests

`tests/test_generator_contract.py`:

1. 모든 adapter의 output shape/dtype
2. sampling plan label 정확히 보존
3. length/mask 정확히 보존
4. pad 값 0
5. 같은 seed 결정론적 동작
6. 다른 seed에서 sample 변화
7. empirical i.i.d.가 entity transition을 사용하지 않음
8. row adapter가 position/entity id를 feature로 사용하지 않음
9. class-specific adapter가 올바른 class pool을 사용
10. block bootstrap b=1이 empirical i.i.d.와 동등
11. contiguous block이 entity 경계를 넘지 않음

### Phase 4 acceptance criteria

- [ ] empirical i.i.d. adapter PASS
- [ ] block-bootstrap context ladder PASS
- [ ] conditional CTGAN smoke PASS
- [ ] conditional TVAE smoke PASS
- [ ] Markov adapter smoke PASS
- [ ] 모든 adapter가 같은 sampling plan 사용
- [ ] output contract test PASS

---

## 12. Phase 5 — coherence/evaluation v2

## 12.0 `eval/behavior_summaries_v2.py`

기존 `soft_g`의 채널별 summary만으로는 v2b의 cross-channel 결속을 측정할 수 없다. v2 전용 named behavior summary를 만든다.

```python
def compute_behavior_summaries_v2(
    batch: SequenceBatch | SyntheticBatch,
    *,
    tau: np.ndarray,
    short_gap_threshold: float,
    window_width: float,
) -> Mapping[str, np.ndarray]:
    """
    Returns one value per entity for:
      velocity, gap, fanout, amount, joint_alignment
    """
```

### `joint_alignment`

valid adjacent transition `t-1 → t`에서:

```python
gap_proxy_t = tau[dt_bin_t]
short_t = 1[gap_proxy_t <= short_gap_threshold]
repeat_t = 1[receiver_t == receiver_{t-1}]
```

정의를 지금 고정한다.

#### Short-gap cutoff

```text
short_gap_quantile = 0.50
```

- pooled real train의 valid transition gap proxy만 사용
- label을 사용하지 않음
- test/synthetic 사용 금지
- `tau[dt_bin]` proxy를 real과 synthetic 모두에 동일 적용
- cutoff value, quantile, fit sample count, binning hash를 `behavior_summary_reference.json`에 저장
- CoF/baseline 결과를 보고 quantile 변경 금지
- 동점이 많은 discretized proxy에서는 `<= cutoff` 규칙을 고정하고 실제 short 비율을 metadata에 기록

entity별:

```python
joint_alignment = mean(
    (short_t - mean(short))
    * (repeat_t - mean(repeat))
)
```

즉, short-gap indicator와 receiver-repeat indicator의 within-entity covariance다.

- denominator는 `n_transition`을 사용하는 population covariance
- sample covariance의 `n-1`을 사용하지 않음
- 첫 position은 repeat 정의가 없으므로 항상 제외

#### 최소 유효 길이

기본:

```yaml
joint_alignment:
  short_gap_quantile: 0.50
  min_valid_transitions: 7
  covariance_denominator: population
```

즉, valid length가 최소 8이어야 primary `joint_alignment`에 포함된다.

- v2 controlled data는 length≥16이므로 모두 포함되어야 함
- real data에서 제외되는 sequence 수와 label별 제외율을 반드시 보고
- 최소 길이 미만을 0으로 채우지 않음
- excluded/invalid mask를 metric report에 저장

구현 요구:

- 첫 position 제외
- 두 position이 모두 valid인 transition만 사용
- minimum valid transitions 미만이면 invalid로 표시하고 이유 기록
- `short_gap_threshold`는 pooled real train valid gap proxy의 median으로 label 없이 fit
- real/synthetic에 동일 threshold 사용
- v2b primary coherence channel은 `joint_alignment`
- v2a primary channels은 기존 velocity/gap/fanout

추가 hard-evaluation sensitivity:

- joint occurrence rate `mean(short_t * repeat_t)`
- lagged alignment `corr(short_t, repeat_{t+1})`

`joint_alignment`을 기존 `soft_g`의 4차원 return에 조용히 끼워 넣지 않는다. 기존 teacher/model contract와 v1 결과를 보호하기 위해 v2 evaluation module에서 named summary로 관리한다. 나중에 coherence loss에 사용하려면 명시적인 별도 ablation으로 추가한다.

이번 구현에서는 differentiable/soft `joint_alignment`을 만들지 않는다. λ>0 coherence-loss 실험을 다시 설계할 때 별도 phase와 test를 두고 구현한다.

#### v2b 설계 검증용 negative controls

v2b에서는 다음 channel-wise coherence를 버리지 않고 모두 보고한다.

- velocity
- gap
- fanout
- amount

의도된 결과:

- channel-wise `C1-C0` dynamic range는 0 또는 매우 작음
- `joint_alignment`에서만 κ에 따라 dynamic range 증가

이 negative-control pattern이 나타나야 “각 채널 자체가 아니라 공동 정렬만 라벨 신호”라는 DGP 설계가 검증된다. channel-wise gap이 크게 분리되면 v2b implementation leakage 또는 univariate-dynamics mismatch로 판정한다.

## 12.1 `eval/coherence_v2.py`

legacy `eval/tstr.py::coherence_gap`을 삭제하지 않는다. v2 함수를 별도로 만든다.

```python
@dataclass(frozen=True)
class CoherenceReference:
    thresholds_by_channel: Mapping[str, np.ndarray]
    n_bins_requested: int
    n_bins_effective: Mapping[str, int]
    ...


def fit_coherence_reference(
    g_real: np.ndarray,
    y_real: np.ndarray,
    *,
    n_bins: int,
    min_bin_count: int,
) -> CoherenceReference:
    ...


def coherence_report(
    g_real: np.ndarray,
    y_real: np.ndarray,
    g_synth: np.ndarray,
    y_synth: np.ndarray,
    reference: CoherenceReference,
) -> Mapping[str, Any]:
    ...
```

rate는 Jeffreys smoothing을 적용한다.

```python
p = (n_positive + 0.5) / (n_total + 1.0)
```

보고:

- channel별 macro gap
- channel별 real-occupancy-weighted gap
- v2a structural mean: velocity/gap/fanout 평균
- v2b primary: joint_alignment
- all-channel mean
- amount negative-control
- bin occupancy와 effective bins

### Primary endpoint

v2의 primary endpoint:

```text
v2a: 8-bin macro coherence gap, mean over velocity/gap/fanout
v2b: 8-bin macro coherence gap on joint_alignment
```

4-bin은 confirmatory endpoint, 2-bin은 descriptive continuity analysis다. 2-bin 단독 결과로 success/failure를 판정하지 않는다.

이 정의는 CoF full result를 보기 전에 config와 `docs/benchmark_v2/preregistered_analysis.md`에 기록한다.

## 12.2 C0

C0는 real-vs-real entity split이다.

- controlled v2: entity가 독립이므로 stratified entity split
- real AMLSim/Sparkov: 같은 원 entity의 여러 window가 양쪽 split에 섞이면 안 됨
- 10개 split seed 실행
- C0 distribution 저장

## 12.3 C1

- real feature/sequence 유지
- entity label vector를 permutation
- exact prevalence 보존
- 10개 shuffle seed
- C1 distribution 저장

## 12.4 직접 순차 지표

coherence-gap만 사용하지 않는다. `eval/sequence_metrics.py`에:

- gap lag-1 autocorrelation
- receiver repeat rate
- receiver mean/max run length
- transition-matrix distance
- burst clustering index
- cross-channel alignment:
  - short-gap indicator와 receiver-repeat indicator의 covariance/correlation
  - joint occurrence rate
  - lagged cross-correlation

를 구현한다.

raw gap이 없는 synthetic sample은 `tau_k[dt_bin]`을 gap proxy로 사용하고 metadata에 명시한다.

각 metric에서:

```text
real class contrast = stat(y=1) - stat(y=0)
synth class contrast = stat(y=1) - stat(y=0)
contrast error = |real contrast - synth contrast|
```

를 보고한다.

## 12.5 fidelity

row marginal fidelity와 sequence fidelity를 분리한다.

Row:

- amount KS/Wasserstein
- dt_bin TVD
- receiver TVD

Sequence:

- transition matrix divergence
- autocorrelation error
- run-length distribution distance

TVD histogram은 real range 밖 synthetic 값을 별도 overflow bin에 포함한다. 범위 밖 값을 버린 뒤 renormalize하지 않는다.

## 12.6 bootstrap

- resampling unit은 entity sequence
- row bootstrap 금지
- 최소 2,000 resamples
- bootstrap seed 저장

### Phase 5 acceptance criteria

`tests/test_coherence_v2.py`:

- exact hand-computed toy example
- hand-computed joint_alignment toy example
- joint_alignment가 `n_transition` denominator의 population covariance와 정확히 일치
- joint_alignment padding/first-position exclusion
- train pooled valid transitions의 median만으로 short-gap cutoff를 fit
- cutoff fit 함수가 label 인자를 받지 않고 test/synthetic data를 참조하지 않음
- real/synthetic에 동일한 frozen cutoff와 `<=` tie rule 적용
- `min_valid_transitions=7` 미만은 invalid이며 0으로 대체되지 않음
- invalid sequence 수와 label별 제외율이 report에 기록됨
- independent gap/repeat에서 joint_alignment≈0
- shared state에서 joint_alignment>0
- v2b에서 channel-wise velocity/gap/fanout dynamic range≈0이고 joint_alignment만 κ에 따라 증가
- 2/4/8-bin
- duplicate quantile threshold
- empty/low-count bin invalid 처리
- Jeffreys smoothing
- macro/weighted 구분
- shuffled label C1

`tests/test_artifact_store.py`:

- atomic directory creation
- resolved config 저장
- manifest 필수 key
- sample NPZ 존재
- incomplete run 표시

---

## 13. Phase 6 — CoF adapter 연결

## 13.1 기존 모델을 먼저 재사용

CoF architecture를 처음부터 다시 쓰지 않는다.

초기 primary config:

```yaml
cof:
  coherence_lambda: 0.0
  cfg_dropout: 0.15
  guidance_scale: 2.0
  discrete_mask_max: 0.7
  start_from_mask: true
  feedback_discrete: true
  feedback_after: 0.3
  loss_positions: all_valid
```

목적은 새 benchmark에서 기존 mechanism이 작동하는지 먼저 확인하는 것이다.

## 13.2 `generators/cof_seqgen_adapter.py`

`fit`:

- `SequenceBatch`를 기존 tensor 형식으로 변환
- `y_entity`를 valid position에 broadcast
- model seed 완전 고정
- resolved hyperparameter 저장
- final checkpoint 저장
- best-checkpoint 기준을 명시

`sample`:

- `SamplingPlan`의 exact y/length/mask 사용
- sampler에 `valid_mask` 전달
- evaluation label은 model label head sample이 아니라 conditioning `y_entity`
- sample의 모든 channel과 mask 저장

## 13.3 training/sample mask mismatch 기록

기존:

- train 최대 70% mask
- sample 시작 100% mask

primary에서는 기존 설정을 유지하되 manifest와 method report에 명시한다.

후속 ablation:

```yaml
mask_schedule:
  - train_max_0.7_sample_full
  - train_max_1.0_sample_full
```

CoF primary 결과를 보고 mask schedule을 선택하지 않는다.

## 13.4 필수 sample 저장

모든 정식 run:

```text
artifacts/benchmark_v2/runs/
└── joint_semimarkov_v2b/
    └── kappa_1.00/
        └── cof/
            └── seed_001/
                ├── config.resolved.yaml
                ├── manifest.json
                ├── checkpoint.pt
                ├── synthetic_test.npz
                ├── sampling_plan.npz
                ├── metrics_2bin.json
                ├── metrics_4bin.json
                ├── metrics_8bin.json
                ├── sequence_metrics.json
                ├── fidelity.json
                └── train.log
```

`manifest.json` 필수 key:

```json
{
  "schema_version": "cofseq_run_v2.0",
  "status": "complete",
  "git_sha": "...",
  "git_dirty": true,
  "python_version": "...",
  "torch_version": "...",
  "cuda_version": "...",
  "gpu_name": "...",
  "dataset_hash": "...",
  "config_hash": "...",
  "sampling_plan_id": "...",
  "model_seed": 1,
  "sampling_seed": 1001,
  "bootstrap_seed": 2001,
  "kappa": 1.0,
  "n_bins": [2, 4, 8],
  "window_width": 7.0,
  "padding_side": "right",
  "checkpoint_path": "...",
  "sample_path": "...",
  "diagnostics": {
    "gap_lag1_autocorr_y0": 0.0,
    "gap_lag1_autocorr_y1": 0.0,
    "receiver_repeat_rate_y0": 0.0,
    "receiver_repeat_rate_y1": 0.0,
    "receiver_mean_run_length_y0": 0.0,
    "receiver_mean_run_length_y1": 0.0,
    "receiver_max_run_length_y0": 0.0,
    "receiver_max_run_length_y1": 0.0,
    "cross_channel_alignment_y0": 0.0,
    "cross_channel_alignment_y1": 0.0,
    "full_sequence_metrics_path": "sequence_metrics.json"
  }
}
```

위 diagnostics는 sample 저장 직후 자동 계산한다. 수동 후처리에 의존하지 않는다. 값이 계산 불가능하면 0으로 채우지 말고 `null`과 명시적 reason을 기록한다.

### Phase 6 smoke

- N_train≤512
- training 100–500 step
- T_steps≤5
- seed 1
- one κ only

검사:

- loss finite
- checkpoint 생성
- sample 생성
- mask/length 보존
- 2/4/8-bin metric 생성
- artifact manifest complete

## 13.5 CoF 실패 시 원인 분리 순서

새 benchmark에서 기존 CoF가 실패해도 즉시 architecture를 다시 쓰지 않는다. 다음 순서를 고정한다.

1. **`soft_g` 감지 가능성**
   - real v2 data에서 κ/class 차이를 `soft_g`와 direct sequence metrics가 실제로 감지하는지
   - DGP oracle이 C0에 접근하는지
2. **Markov/sequence baseline 재현**
   - v2a에서 learned Markov가 persistence를 복원하는지
   - v2b에서 joint Markov/neural baseline이 cross-channel alignment를 복원하는지
3. **CoF 생성 sample의 직접 순차 통계**
   - gap lag-1 autocorrelation
   - receiver repeat rate
   - run-length distribution
   - cross-channel alignment
4. **CFG class separation**
   - 동일 noise/sampling plan에서 y=0과 y=1 조건이 실제 output distribution을 분리하는지
   - κ=0에서 spurious split을 만들지 않는지
5. **discrete corruption 기여**
   - masking/feedback가 transition learning을 돕는지
   - train 70% mask/sample 100% mask mismatch 영향
6. **loss position과 architecture**
   - 마지막에만 `all_valid` vs `corrupted_only`, layer/width/schedule을 ablation

각 단계의 진단을 artifact로 저장하고, 앞 단계가 실패한 상태에서 뒤 단계 hyperparameter sweep을 하지 않는다.

---

## 14. Phase 7 — 정식 실험

`FULL_EXPERIMENT`에서만 실행한다.

## 14.1 모델 세트

Diagnostic/reference:

- C0
- C1
- parametric DGP oracle
- empirical conditional i.i.d.

Learned row baselines:

- conditional CTGAN
- conditional TVAE

Sequence baselines:

- channel-independent class-conditional Markov
- joint observed class-conditional Markov
- 최소 하나의 neural sequence baseline
- 가능하면 HMM/HSMM strong baseline

Proposed:

- CoF-SeqGen

## 14.2 seed

최소:

```yaml
model_seeds: [1, 2, 3, 4, 5]
sampling_seed_offset: 1000
bootstrap_seed_offset: 2000
```

가능하면 10 seed.

모든 learned model은 5개 이상 독립 학습 seed를 가져야 한다.

## 14.3 κ 순서

계산 자원을 아끼기 위해:

1. v2a κ=0/1 calibration
2. v2b κ=0 smoke/full sanity
3. v2b κ=1 primary
4. v2b κ=0.7
5. v2b κ=0.5
6. v2b κ=0.3
7. 필요한 v2a remaining κ

그러나 final summary에는 모든 κ를 동일하게 보고한다.

각 scenario의 κ=1에서:

- conditional i.i.d.≈C1
- DGP oracle≈C0
- sequence metric에 명확한 label contrast

가 먼저 확인되지 않으면 CoF full curve를 진행하지 않는다.

## 14.4 공정성

모든 learned generator:

- 동일 train split
- 동일 evaluation sampling plan
- 동일 label prevalence
- 동일 length vector
- epoch뿐 아니라 optimizer step과 observed-row budget 기록
- tuning budget 기록
- seed 수 동일

CTGAN/TVAE의 30 epoch fidelity와 100 epoch coherence 같은 혼용 금지.

## 14.5 통계

`eval/stats_v2.py`:

- model별 seed mean, SD, 95% CI
- Welch t-test
- Hedges' g effect size
- bootstrap difference CI
- primary 비교의 방향과 one-/two-sided 여부 사전 기록
- κ/channel 다중 비교 시 Holm correction

Primary hypothesis:

```text
In joint_semimarkov_v2b at κ=1.0,
CoF structural 8-bin macro coherence gap
is lower than each learned row-independent baseline.
```

Markov/neural sequence baseline 비교는 별도:

```text
CoF vs sequence-aware alternatives
```

CoF가 row baseline만 이기고 v2b의 Markov/neural sequence baseline을 이기지 못하면:

- benchmark contribution은 유지 가능
- CoF 우월성 주장은 축소

## 14.6 정식 실행 예시

실제 CLI는 이 interface를 만족하도록 구현한다.

```bash
python -m scripts.run_benchmark_v2 \
  --config configs/benchmark_v2/baselines.yaml \
  --data-root data/benchmark_v2 \
  --artifact-root artifacts/benchmark_v2/runs \
  --scenarios markov_persistence_v2a joint_semimarkov_v2b \
  --kappas 0.0 0.3 0.5 0.7 1.0 \
  --generators empirical_iid conditional_ctgan conditional_tvae markov_independent markov_joint \
  --seeds 1 2 3 4 5 \
  --device cuda

python -m scripts.run_benchmark_v2 \
  --config configs/benchmark_v2/cof.yaml \
  --data-root data/benchmark_v2 \
  --artifact-root artifacts/benchmark_v2/runs \
  --scenarios markov_persistence_v2a joint_semimarkov_v2b \
  --kappas 0.0 0.3 0.5 0.7 1.0 \
  --generators cof \
  --seeds 1 2 3 4 5 \
  --device cuda

python -m scripts.summarize_benchmark_v2 \
  --artifact-root artifacts/benchmark_v2/runs \
  --output-root artifacts/benchmark_v2/summary
```

---

## 15. Phase 8 — 실제 데이터 재계산

benchmark v2와 모델 실험 이후 수행한다.

## 15.1 padding

`data/build_sequences.py`:

- config에 `padding_side`
- v2 canonical은 `right`
- 기존 left-pad data는 덮어쓰지 않음
- 새로운 output directory에 재구축

```python
dest_slice = slice(0, seg_len)
```

모든 real data loader가 `entity_ids`를 보존하게 한다.

## 15.2 dataset-specific W

하드코딩 제거:

```yaml
datasets:
  amlsim:
    window_width: 7.0
    window_unit: step
  sparkov:
    window_width: 60.0
    window_unit: minute
```

Sparkov의 7분/60분/1주 중 primary 정의를 연구적으로 하나 선택하고 사전 기록한다. 다른 값은 sensitivity로만 실행한다.

현재 `W_SPARKOV_MINUTES=60`과 다른 script default가 충돌하므로 단일 resolved config가 source of truth가 되어야 한다.

## 15.3 C0/C1

- mask-aware `soft_g`
- entity-disjoint reference split
- 2/4/8-bin
- 10 seeds
- bin occupancy 저장

## 15.4 effective κ

수작업 CSV 금지.

`scripts/recompute_real_metrics_v2.py`가:

1. synthetic κ curve의 C1 distribution 읽음
2. real C1과 prevalence 차이 기록
3. interpolation 수행
4. uncertainty interval 계산
5. 결과 JSON/CSV 생성

하도록 한다.

cross-dataset prevalence 차이 때문에 effective κ는 “approximate mapping”으로만 표시한다.

---

## 16. 추가 테스트 정리

## 16.1 stale phase4 test

현재 `tests/test_phase4_precheck.py` 일부가 존재하지 않는 `info["g_std"]`를 기대한다.

임의로 테스트를 삭제하지 않는다.

1. 현재 intended `CoFSeqGen.compute_loss` return contract를 문서화
2. test가 오래된 contract를 검사하는지 확인
3. intended contract가 `L_diff`, `L_coh`, `L_label`이면 test를 그 contract로 업데이트
4. `g_std` 검증이 필요하면 모델 buffer를 직접 검사

## 16.2 전체 test 명령

```bash
pytest -q tests/test_soft_g_grad.py
pytest -q tests/test_soft_g_mask.py
pytest -q tests/test_benchmark_v2_dgp.py
pytest -q tests/test_benchmark_v2_gate.py
pytest -q tests/test_generator_contract.py
pytest -q tests/test_sampler_padding.py
pytest -q tests/test_coherence_v2.py
pytest -q tests/test_artifact_store.py
pytest -q
```

전체 suite가 오래 걸리면 timeout과 마지막 완료 test를 report에 기록한다. 일부 test만 성공했는데 전체 성공이라고 쓰지 않는다.

---

## 17. 결과 해석 decision table

| 결과 | 해석 | 허용되는 다음 행동 |
|---|---|---|
| single-row AUROC > 0.52 | 행 단위 label leakage | DGP 수정, 모델 학습 금지 |
| 위치별 occupancy 또는 관측 marginal이 초반에 drift | equilibrium residual 초기화 오류 가능 | duration sampler와 mask/indexing 수정, 모델 학습 금지 |
| single-row≈0.5, pair/context도≈0.5 | 신호 자체가 너무 약하거나 metric 불일치 | dynamic range/N/W/L 점검 |
| v2b context1≈0.5, context2 CI lower≤0.52 | 의도한 pair-level 공동 정렬 신호가 검출되지 않음 | pair feature/cutoff/receiver indexing 점검, Gate A FAIL |
| v2b context4/8이 context2보다 0.02 초과 하락 | classifier/implementation 불안정 | capacity·regularization·split 점검, 자동 PASS 금지 |
| iid bootstrap가 C1보다 실질적으로 낮음 | 순차 식별 실패 | DGP/metric 수정 |
| DGP oracle가 C0보다 멂 | metric 또는 generator 오류 | metric/DGP 진단 |
| v2a Markov가 C0 근처, CoF가 열세 | expected calibration outcome 가능 | v2a에서 CoF 우월 주장 금지, v2b 확인 |
| v2b joint Markov가 C0 근처, CoF가 실패 | CoF 또는 inductive-bias 한계 | 고정 진단 순서로 분석 |
| CoF가 row baseline만 이김 | sequence modeling 필요성 지지 | C2 우월성은 제한적으로 표현 |
| CoF가 Markov/neural baseline도 이김 | CoF 기여 강화 | 정식 통계·ablation |
| 2-bin만 성공, 4/8-bin 실패 | coarse-bin artifact 가능 | robustness claim 금지 |
| v2b velocity/gap/fanout/amount 중 하나의 C1−C0가 크게 분리 | channel marginal/dynamics가 라벨 간 같지 않음 | v2b leakage/univariate-dynamics mismatch 검사 |
| κ=0에서 model gap 상승 | spurious class conditioning | 별도 limitation/ablation |
| κ=1 dynamic range <0.01 | 5-seed 판별력 부족 | 모델 학습 금지, N/power 우선 |

---

## 18. Notion/논문 주장 관리

코드 작업 중 Notion을 자동 수정하지 않는다. 대신:

```text
docs/benchmark_v2/claim_status.md
```

를 만든다.

초기 상태:

| Claim | Status |
|---|---|
| v1 row-independent structural impossibility | rejected by empirical conditional i.i.d. diagnostic |
| benchmark v2 isolates temporal coupling | pending gate |
| conditional row models fail on v2 | pending full baseline |
| CoF preserves v2 coupling | pending full experiment |
| CoF outperforms sequence baselines | pending |
| real effective κ | invalid until mask-aware recomputation |

각 claim은 대응 artifact 링크를 가져야 한다.

Sajja(2026)의 proposition을 논문에 인용하기 전에 arXiv PDF 원문의 proposition statement와 assumptions를 직접 확인하고 페이지/절을 기록한다.

---

## 19. 재현성 파일

추가:

```text
environment.yml
requirements-freeze.txt
docs/benchmark_v2/runbook.md
docs/benchmark_v2/preregistered_analysis.md
docs/v2_implementation_report.md
```

### `runbook.md`

포함:

- environment 생성
- data generation
- gate validation
- smoke experiment
- full baseline
- full CoF
- summary/plot
- real data recomputation

### `preregistered_analysis.md`

CoF full 결과 전에 작성·commit:

- primary κ
- primary bins
- primary channels
- primary baseline comparisons
- seed 수
- exclusion/invalid-run rule
- statistical test
- multiple comparison correction

---

## 20. 최종 산출물

`IMPLEMENT_AND_SMOKE` 완료 시:

```text
docs/v2_implementation_report.md
artifacts/benchmark_v2/gates/gate_report.{json,md}
artifacts/benchmark_v2/smoke/...
```

`docs/v2_implementation_report.md` 필수 내용:

1. 최종 Git status와 SHA
2. 생성·수정 파일 목록
3. architecture/interface 설명
4. mask bug 수정 내용
5. benchmark v2 수학적 정의
6. gate별 실제 수치와 PASS/FAIL
7. 실행한 모든 command
8. test 결과
9. smoke 결과와 artifact 경로
10. 미실행 full-run command
11. 알려진 한계와 blocker
12. 기존 v1 자산이 보존되었음을 확인

`FULL_EXPERIMENT` 완료 시 추가:

```text
artifacts/benchmark_v2/summary/
├── main_table.csv
├── channel_table.csv
├── sequence_metrics.csv
├── statistics.csv
├── kappa_curve_2bin.png
├── kappa_curve_4bin.png
├── kappa_curve_8bin.png
├── artifact_index.json
└── research_conclusion.md
```

---

## 21. Definition of Done

다음 항목이 모두 충족되어야 “코드 수정 완료”다.

### 코드

- [ ] benchmark v2가 별도 deep module로 구현됨
- [ ] 모든 generator가 공통 interface 사용
- [ ] mask가 data→model→sampler→metric 전체 경로에 전달됨
- [ ] legacy v1 harness/result 보존
- [ ] 모든 결과가 v2 전용 경로에 저장됨

### benchmark

- [ ] single-row marginals practical equivalence
- [ ] single-row AUROC≈0.5
- [ ] equilibrium residual-duration 초기화가 survival formula와 일치
- [ ] 위치별 burst occupancy와 관측 marginal에 early/late drift 및 라벨 차이 없음
- [ ] v2b context2 AUROC의 95% CI lower bound>0.52; context1은 [0.48,0.52] equivalence
- [ ] context 1/2/4/8 AUROC curve와 CI 저장; 4/8이 context2보다 0.02 이상 하락하지 않음
- [ ] `joint_alignment` cutoff가 pooled real train valid transition에서 label 없이 한 번만 fit·동결됨
- [ ] `repeat_t=1[c_t=c_{t-1}]`, 첫 위치 제외, population covariance, 최소 7 transitions 규칙 준수
- [ ] 길이 미달 entity는 0 대체 없이 invalid 처리되고 label별 제외율 보고
- [ ] conditional i.i.d.≈C1 at 4/8 bins; 2-bin 결과 별도 보고
- [ ] DGP oracle≈C0
- [ ] coupling strength가 κ에 따라 증가
- [ ] κ=1 dynamic range가 hard minimum 0.01 이상, target 0.02 이상
- [ ] v2b channel-wise velocity/gap/fanout/amount는 negative control로 보고되고 dynamic range≈0
- [ ] v2b에서는 joint_alignment만 κ에 따라 분리
- [ ] v2a Markov calibration과 v2b joint semi-Markov 결과 분리
- [ ] block-bootstrap context-length curve 생성

### baseline

- [ ] empirical conditional i.i.d.
- [ ] conditional CTGAN
- [ ] conditional TVAE
- [ ] class-conditional Markov
- [ ] channel-independent와 joint Markov 분리
- [ ] neural sequence baseline skeleton/smoke

### CoF

- [ ] sampling plan label/length 준수
- [ ] padding-safe sampler
- [ ] checkpoint 저장
- [ ] synthetic sample 저장
- [ ] 2/4/8-bin metric 저장
- [ ] lag-1 autocorrelation/run-length/cross-channel alignment가 manifest에 자동 기록

### 통계·재현성

- [ ] 최소 5 independent seed 설계
- [ ] Welch + effect size + bootstrap CI 구현
- [ ] resolved config/seed/Git SHA/environment 저장
- [ ] stale test 정리
- [ ] 전체 test 결과 보고

### 연구 주장

- [ ] v1 structural-impossibility claim 폐기 또는 legacy limitation으로 표시
- [ ] v2 gate 통과 전 positive claim 금지
- [ ] row baseline과 sequence baseline 결론 분리
- [ ] real effective κ는 재계산 전 invalid 표시

---

## 22. 구현 중 판단이 필요한 경우의 우선순위

1. 연구 식별 가능성
2. 데이터·결과 보존
3. 재현성
4. 공정한 비교
5. interface 단순성
6. 계산 효율
7. 기존 코드와의 호환성

호환성을 위해 silent bug를 유지하지 않는다. 특히 `valid_mask=None` 기본값으로 padding 문제를 숨기지 않는다.

---

## 23. 마지막 지시

- 코드가 실행된다는 것만으로 완료 처리하지 말 것.
- benchmark gate 수치가 연구 질문을 실제로 분리하는지 확인할 것.
- gate 실패를 모델 성능으로 덮지 말 것.
- full sample과 provenance가 없으면 정식 결과로 인정하지 말 것.
- CoF를 재작성하기 전에 기존 CoF가 새 benchmark에서 무엇을 하는지 먼저 측정할 것.
- 가장 단순한 Markov baseline을 반드시 포함할 것.
- 구현 결과를 과장하지 말고, 통과하지 못한 gate와 실패한 test를 그대로 보고할 것.

최종 핵심:

> **이번 작업의 중심은 CoF 코드를 처음부터 다시 쓰는 것이 아니다. v2a Markov calibration과 v2b joint semi-Markov challenge로 구성된 benchmark suite, 그리고 공정한 generator seam을 먼저 확립하고 기존 CoF를 그 위에서 다시 평가하는 것이다.**
