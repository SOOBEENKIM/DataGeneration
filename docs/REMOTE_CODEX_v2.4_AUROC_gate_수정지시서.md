# v2.4 후보 수정 지시서 — 단일 행 AUROC gate의 귀무분포 보정 교체

## 0. 최상위 원칙 (변경 불가)

1. v2.0/v2.2/v2.3의 config·데이터·artifact·preregistration은 수정·삭제하지 않는다. 새 산출물은 전부 `artifacts/benchmark_v2_4/` 및 `configs/benchmark_v2/main_v2_4_candidate.yaml` 경로에 격리한다.
2. 학습형 생성기(CTGAN/TVAE/neural/CoF) 결과는 여전히 어떤 결정에도 사용하지 않는다. 이 수정은 순수 측정도구 보정이다.
3. DGP 파라미터, continuous primary endpoint, 유지된 v2.2/v2.3 PASS 증거는 변경하지 않는다.
4. 아래 Step A의 진단 결과를 본 뒤 Step B의 절차·기준을 바꾸는 것은 금지한다. 이 문서의 규칙을 Step A 실행 전에 commit하라.
5. 각 Step 종료 시 명령·소요시간·PASS/FAIL·artifact 경로를 `docs/v2_implementation_report.md`에 누적 기록한다.

## 1. v2.3 FAIL의 원인 진단 (이미 확정된 사실, 재확인만)

v2.3 calibration 산출물의 독립 검산 결과:

- clean 400 cell-seed의 단일 행 AUROC: mean 0.4998, sd 0.0074 → **벤치마크 데이터 자체는 깨끗함** (검사 대상 성질은 참).
- 실패 원인 1 — 규칙 A의 기하학적 불능: CI 반폭 평균 0.0142 vs margin ±0.02 → CI가 통째로 들어가려면 점추정치가 0.5±0.0058 이내여야 하는데 점추정치 sd가 0.0074. 개별 cell 통과율 54%, 4-cell 동시 요구로 10%로 붕괴 → familywise false-fail 0.90–0.99.
- 실패 원인 2 — 검출력 부족: 2% 누출이 AUROC를 평균 +0.008밖에 안 움직이는데 노이즈 sd가 0.0074라, 현재 평가 N에서는 어떤 규칙도 2%@0.80을 달성할 수 없음 (최고 하한 0.0592).

결론: 규칙 교체(원인 1)만으로는 부족하고, 평가 전용 N 확대(원인 2)가 병행되어야 한다.

## 2. Step A — 기존 artifact 재분석 (재실행 없음, CPU 수 분)

목적: (a) 보정 방식이 false-fail을 해결함을 기존 데이터로 확증, (b) Step B의 N을 **관측 효과크기 기반 power 분석**으로 사전 결정.

`scripts/reanalyze_auroc_gate_v2_4.py` 신규 작성:

1. **통계량 정의(고정)**: seed s, classifier c에 대해
   `T(s,c) = max over 4 cells(scenario×kappa) of |test_auroc − 0.5|`
2. `clean_seed_results.csv`의 method A 행만 사용(점추정치는 method 간 동일). **짝수 index seed 50개**로 T의 95퍼센타일 임계값 t̂_c를 classifier별로 계산, **홀수 index seed 50개**에서 false-fail 빈도를 산출해 exact 95% CI와 함께 기록한다. (n=50이라 상한≤0.05 판정은 불가 — 이 수치는 확증이 아니라 **진단**으로만 기록. 판정용 calibration은 Step B에서 수행.)
3. **Power 분석 → 필요 N 산출(핵심)**: `injected_leakage_results.csv`에서 feature×classifier×magnitude별 관측 AUROC shift Δ의 분포를 추정한다. AUROC shift는 모집단 성질이라 N에 근사 불변, 노이즈 sd는 σ_N ≈ σ_현재 × sqrt(N_현재/N)로 축소된다. 각 feature/classifier에 대해
   `P(|Δ + noise| > t_N) ≥ 목표power` (2%→0.80, 5%→0.90, t_N은 해당 N의 familywise 95퍼센타일)
   를 만족하는 최소 N 배수를 계산하고, **가장 어려운 feature(예상: gap 또는 receiver-category)가 결정하는 N_required에 안전계수 1.25**를 곱해 `N_audit`로 고정한다. 산출 근거를 `docs/benchmark_v2/auroc_power_analysis_v2_4.md`에 기록.
4. N_required가 비현실적(예: 32× 초과)으로 나오는 feature가 있으면, 그 feature에 한해 **분산 축소 대안**(두 classifier AUROC 평균 통계량, 또는 반복 cross-fit 평균)의 필요 N을 같은 방식으로 병기하고, 그래도 초과 시 해당 feature의 2% 기준을 descriptive로 강등하는 것을 **Step B 실행 전에** 이 문서의 개정으로 명시·commit한 뒤 진행한다(결과 관찰 후 변경 금지).

## 3. Step B — 확대 N에서 보정 calibration 재실행 (1회)

`scripts/calibrate_auroc_gate_v2_4.py`:

- **평가 전용 N**: Step A가 산출한 N_audit (train N 31,951은 불변; receiver 4N 선례와 동일하게 audit 데이터만 확대).
- **방법 1종만**: 위 T 통계량 + calibration 분위수 임계값. v2.3의 A/B/C 비교, seed별 bootstrap CI(200회), 1% 누출 주입은 **모두 제거**한다(귀무분포 방식에서 불필요 — 지난 7.4시간의 주요 비용 제거).
- **Seed 설계**: calibration용 clean 100 seed(임계값 산출) + validation용 clean 100 seed(판정). 판정 기준: validation false-fail exact 95% 상한 ≤ 0.05, feature/classifier별 2% power exact 하한 ≥ 0.80(강등된 feature 제외), 5% ≥ 0.90. Orientation은 validation entity에서만.
- **엔지니어링 요구(필수)**: (a) worker당 `OMP_NUM_THREADS`/`MKL_NUM_THREADS`를 `총코어/worker수`로 제한 — v2.3의 2–3× 메모리 경합 제거, (b) cell-seed 단위 **즉시 append 저장**(중간 체크포인트) + 재시작 시 완료분 skip, (c) 1-seed smoke로 task당 시간 실측 후 총 예상시간을 로그에 기록하고 시작.
- 예상 규모: bootstrap 제거·방법 1종·2magnitude로 task당 비용이 급감하므로, N_audit 4–8×를 감안해도 v2.3(7.4h)과 비슷하거나 짧아야 정상. 크게 초과 예상 시 시작 전에 보고.

## 4. Step C — v2.4 gate 조립과 이후

1. `scripts/build_v2_4_cpu_gate.py`: 유지된 v2.2/v2.3 PASS 증거(continuous endpoint, iid/block ordering, oracle, signed receiver, positionwise, bin validity, fanout 재분류, channel-only controls)를 참조하고 **AUROC 항목만 Step B 결과로 교체**. 하나라도 FAIL이면 `learned_smoke_authorized=false` 유지.
2. 전항목 PASS 시: `docs/benchmark_v2/preregistered_analysis_v2_4.md` 작성·commit(이 문서의 규칙 + Step A power 분석 + Step B 임계값을 그대로 봉인) → **그 후에만** GPU 1-seed smoke(conditional CTGAN/TVAE/neural/CoF, GPU당 1모델) 승인.
3. Step B가 FAIL이면(강등 반영 후에도 power 미달): 결과를 기록하고 강제 중단, 사용자에게 보고. 임계값·margin·N을 결과 관찰 후 추가 조정하지 않는다.

## 5. 완료 보고 항목

- Step A: 홀수-seed false-fail 진단치, feature별 필요 N 표, 채택된 N_audit
- Step B: validation false-fail 상한, feature/classifier별 2%/5% power 하한, 임계값
- Step C: gate_report.json 전항목, learned_smoke_authorized 값, 최종 commit SHA

## 6. Step A 이전 보충 규칙 (2026-07-29, 변경 불가)

이 절은 Step A 실행 전에 추가되며 위 규칙을 구체화한다.

1. `N_audit`은 learned generator의 train N이 아니다. AUROC audit 전용
   train/validation/test 세 split을 모두 같은 배수로 확대한다. 기존
   generator training N=31,951과 모델 평가 계획은 변경하지 않는다.
   Audit 데이터는 별도 seed/path/provenance를 사용한다.
2. Step A의 `1/sqrt(N)` power 계산은 train/validation fitting noise까지
   같은 비율로 줄어든다는 보장이 없으므로 **optimistic lower bound**로
   표시한다. Step B에서 audit train/validation/test를 함께 확대한다.
3. Injected leakage는 audit train/validation/test에 동일한 사전정의
   operator로 적용한다. Categorical leakage는 class-conditional category
   probability의 정확한 증가로, continuous gap/amount는 고정 contamination
   mixture로 정의한다. Direction/category operator 목록은 모든 seed에
   대해 동일하다.
4. 특정 feature의 2% AUROC 기준을 descriptive로 강등하더라도 직접
   row-marginal guard는 hard gate로 유지한다: gap은
   KS/effect-size/emission contract, receiver는 signed cluster-frequency,
   amount는 KS/effect-size/emission contract를 사용한다.
5. Calibration quantile은 interpolation 없는 order statistic이다.
   `rank = ceil((n_calibration + 1) * 0.95)`로 정의하며,
   `n_calibration=100`이면 정렬된 값의 96번째를 사용한다.
6. Calibration/validation seed 범위, source commit SHA, resolved config
   SHA-256, relevant code SHA-256을 artifact manifest에 기록한다.
7. Step A 결과가 32배를 초과하면 미리 정한 분산 축소 대안을 순서대로
   평가한다. 그래도 초과하는 feature의 2% 기준만 descriptive로
   강등할 수 있으며, 선택과 hard marginal guard를 Step B 전에 commit한다.
8. Step B 이후에는 N, threshold 정의, leakage operator, feature별
   hard/descriptive 분류를 다시 변경하지 않는다.

## 7. 최종 통계 설계 amendment (Step A 실행 전, 이전 rank 규칙 supersede)

이 절은 위의 classifier별 `T(s,c)`, 50/50 분할, rank 49/50 및 Step B의
rank 96/100 규칙을 모두 대체한다. 해당 규칙으로 생성된 결과가 있으면
`pre_amendment` artifact로 보존하되 v2.4 최종 근거나 `N_audit` 결정에
사용하지 않는다.

1. Seed `s`마다 통합 familywise 통계량 하나만 사용한다.

   ```text
   T(s) = max over 2 scenarios x 2 kappa x 2 classifiers
          |test_AUROC(s, cell, classifier) - 0.5|
   ```

   Clean false-fail과 leakage detection은 모두 단일 사건
   `T(s) > t_hat`으로 정의한다. Classifier별 power 표는 유지하지만
   classifier별 threshold는 만들지 않는다.
2. Step B의 calibration clean seed는 200개, 독립 validation clean
   seed는 200개다. `t_hat`은 200개 calibration `T`의 maximum, 즉
   interpolation 없는 rank 200/200이다.
3. Validation clean 200개에서 Clopper-Pearson two-sided exact 95% CI의
   false-fail 상한이 0.05 이하여야 한다.
4. Feature/classifier별 leakage power에서도 target classifier의 injected
   네 cell과 다른 classifier의 clean 네 cell을 합친 동일한 8-cell
   maximum을 사용한다. 검출 사건은 오직 `T(s) > t_hat`이다.
5. Step A의 immutable source에는 clean seed가 100개뿐이므로 rank 200을
   직접 계산했다고 주장하지 않는다. 현재-N의 보수적 proxy threshold는
   다음 두 값의 maximum으로 사전 고정한다.

   - 100 source seed에서 관측된 통합 `T`의 maximum
   - 각 8개 cell-classifier deviation에 대해 `|mean|`의 95% upper
     uncertainty, chi-square one-sided 95% upper SD, 그리고
     Bonferroni tail `1/(2*8*201)`을 사용한 Gaussian upper prediction
     bound 중 maximum

   이 proxy를 `1/sqrt(N)`로 축소하는 계산은 여전히 optimistic lower
   bound이며, 관측 효과 shift는 고정한다.
6. 최대 필요 배수에 기존 안전계수 1.25를 적용한다. Audit 전용
   train/orientation-validation/test 세 split에 같은 배수를 적용하며
   learned generator train N=31,951과 실제 모델 평가 계획은 바꾸지
   않는다.
7. Step B의 1-seed smoke는 runtime과 peak memory 외에 base-N과
   selected-N nested diagnostic의 clean deviation을 기록한다. 단일
   seed로 variance floor를 추정할 수 없음을 명시하고, 관측 편차가
   `1/sqrt(N)` projection보다 큰 위험 신호인지 기록한다.
8. 이전 절의 두-classifier AUROC 평균 대안은 통합 maximum `T(s)` 정의와
   양립하지 않으므로 supersede한다. 필요 시 각 classifier 내부의
   repeated-cross-fit 평균만 분산축소 대안으로 사용할 수 있으며,
   최종 통계량은 계속 8개 값의 maximum이어야 한다.
