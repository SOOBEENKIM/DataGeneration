# CS-SAF: 모델·수식·데이터·선행연구·기여 주장 감사

2026-09-16 기준. 이번에 실제 읽은 구현, 저장된 실행 근거, 저자 논문/공식 문서를
대조한 기록이다. 모든 관련 문헌을 망라한 독창성 증명이나 외부 구현의 완전한 검증은 아니다.
최신 실험은 [고정 checkpoint 분해](route_decomposition_v1_report_2026_09_16.md)이며,
결론과 원시 요약은 [evidence JSON](route_decomposition_v1_result.json)에 있다.

**연구를 발전시킬 근거는 있지만, 기존 baseline보다 우수한 새 모델이 완성됐다는
근거는 아직 없다.** 지금까지 확인한 핵심은 활성 신호 부재가 아니라, 유용한 이력 보정과
gap 의존성을 함께 담은 경로가 비활성 집단에서도 불필요하게 변한다는 문제다.
이 문제를 목적함수·표현·일반화 측면에서 분리하고, 해결 후보를 공정하게 검증해야 한다.

## 1. 현재 모델이 실제로 정의하는 분포

현재 CS-SAF는 **자가회귀 조건부 시퀀스 생성기**다. 저장소가 TabDiff에서 출발했다는
사실과 현재 모델이 diffusion인지는 다른 문제다. 이 CS-SAF 실행은 diffusion 학습이 아니다.
observed static context s와 제공된 길이 L에 대해, 이력 H_t는 t 이전 사건만 포함한다.

```text
h_t = shifted_GRU(events_<t; static_initial_state(s))
c_t = concat(h_t, static_embedding(s))

p(sequence | s,L)
 = p(m_1 | s) p(v_1 | s,m_1)
   * product_(t=2..L) p(k_t | h_t) p(m_t | c_t,k_t) p(v_t | h_t,k_t,m_t)

p(m_t=j | c_t,k_t)
 = q_t * 1[j=m_(t-1)] + (1-q_t) f_theta(j | c_t)
q_t = sigmoid(b_theta(c_t) + r_s(c_t,k_t))

r_s(c,k) = sum_d w_(s,d) tanh(Wc_s c + bc_s)_d
                        * tanh(Wg_s E_gap(k))_d / sqrt(16)
```

v는 signed-log/표준화한 numeric value다. 첫 사건에는 gap 손실이 없고 mark/value
손실은 있다. p(v)는 변환 공간의 Gaussian이다. 위 식은 모델의 분해를 설명하며,
아래 실제 최적화 목적함수의 head별 평균 가중치까지 동일하다는 뜻은 아니다.
생성 시 parent/길이 계획을 제공하므로 현재 연구는 parent 분포와 길이 분포까지
학습하는 완전한 relational generator의 성공을 입증하지 않는다.

| 구성 | 현재 구현/범위 | 코드 근거 |
|---|---|---|
| 이력 | strictly-past GRU, hidden 128, 길이 2–32의 entity 전체 한 번 | [CausalHistoryEncoder](../../models/cof_seqgen_saf.py), [loss_terms](../../models/cs_saf.py) |
| static | 하나의 알려진 이진 범주, embedding 8; GRU 초기화와 직접 context 입력 | [CSSAF](../../models/cs_saf.py) |
| gap | train에서 선택한 31개 대표값의 categorical decoder | [GapSupportState/fit_train_only_gap_support](../../models/cof_seqgen_saf.py) |
| copy route v2 | 알려진 context별 rank 16 bank 두 개; 활성 여부는 입력하지 않음 | [CSSAFv2](../../models/cs_saf_v2.py) |
| 용량 | 총 133,549개, route 5,440개; v1의 공유 rank 32와 총수 동일 | [v2 계약](revision_v2_preregistration.md) |
| fresh mark | c에 의존하는 categorical head, 현재 gap 직접 입력 없음, reserved 출력 제외 | [mark_distribution](../../models/cs_saf.py) |
| value | history, 현재 gap, 현재 mark 조건부 Gaussian | [_value_parameters](../../models/cof_seqgen_saf.py) |
| 진단 | 동일 가중치에서 모든 gap bin의 입력 반응, factual likelihood와 분해 | [분해 구현](../../experiments/cs_saf_route_decomposition.py) |

bank를 나눠도 GRU, gap embedding, base/fresh head는 공유된다. 따라서 v2는 모든
집단 간 gradient 영향을 차단하지 않는다. 총 파라미터 수를 맞춘 대신 집단별 rank는
32에서 16으로 줄어든다. v2의 실패/개선을 오직 공유 여부의 효과로 단정하면 안 된다.

### Copy 확률과 관측 repeat 확률의 구별

관측값은 `I[m_t=m_(t-1)]`이고 latent copy coin은 아니다.

```text
p_repeat = q + (1-q) f_theta(previous | c)
```

fresh 표본도 이전 mark와 같을 수 있다. 현재 모델은 관측 repeat likelihood로
보조 손실을 계산한다. latent coin을 정답으로 넣는 누출은 없다.
또한 flexible fresh head 아래 q를 실제 latent copy 확률과 동일시할 수 없다.
한 이력에서 관측 mark 분포만 주어지면 `q <= p(previous)`인 여러 q/f 조합이
같은 분포를 나타낼 수 있다. fresh head가 gap과 무관하다는 제한이 있어도,
그 head의 확률이 알려진 DGP 값이라는 보장은 없다. DGP의 균등 fresh=1/64라는
가정으로 정의한 oracle q와 학습된 q의 비교는 이 차이를 명시해야 한다.
주요 정확도 주장은 **관측 가능한 전체 mark 조건부 분포/TV와 repeat 예측**으로
뒷받침하고, 내부 q range만으로 성공을 정하지 않아야 한다.

## 2. 손실과 이번 분해의 이론적 의미

일반 손실 U의 base는 gap/mark/value 각각의 유효 target 평균 NLL을 더한다.
엄밀히 동일 가중치의 전체 시퀀스 log likelihood 합과 같지는 않다.
entity-uniform minibatch에서 고정된 train transition 분모를 사용한 보조 손실은:

```text
U = base
A = base + (1/T) sum_(train nonfirst events) BCE_repeat
B = base + (1/2) sum_(s=0,1) (1/T_s) sum_(events in s) BCE_repeat
```

실제 A/B minibatch 추정에는 entity 수/배치 크기 보정이 들어간다.
[loss-control 구현](../../experiments/cs_saf_loss_control.py)과
[기본 objective](../../models/cs_saf.py)가 이에 대응한다.
A−U는 보조 손실 추가 효과, B−A는 전체 평균 계수를 유지한 집단별 배분 효과를
분리한다. 이는 realized gradient norm까지 같게 만든 실험은 아니다.
repeat BCE는 mark NLL과 정보를 공유하므로 새로운 독립 정답이 추가되는 것도 아니다.
B는 동일 집단 평균이지 worst-group max를 최소화하는 GroupDRO가 아니다.

기존 [6-fit 결과](loss_control_v1_report_2026_09_16.md)는 A와 B가 rare-null 반응을
더 키웠음을 보여준다. 그러나 U도 실패했고 B는 active repeat BCE에서 이득을 보였다.
따라서 “balanced loss가 유일한 원인” 또는 “balanced loss는 항상 나쁘다”는 결론은
둘 다 부적절하다. 현재 primary B의 필수성을 전제로 후속 설계를 진행하면 안 된다.

이번에는 `r=mu+delta`, `mu(c)=E_(train gap | s)[r(c,g)]`를 정의했다.
이것은 logits의 대수적 분해이고 **중심화만 하면 원 모델 F의 예측은 그대로다**.
mu는 선택한 context-marginal 기준에 의존한다. history-conditional 평균,
uniform-bin 평균, 확률 평균과 다르며 완전한 functional-ANOVA도 아니다.

고정 c에서 sigmoid의 미분이 최대 1/4이므로:

```text
range_g q(c,g) <= range_g delta(c,g) / 4
range_g p_repeat(c,g) = (1-f_theta(previous | c)) * range_g q(c,g)
```

이는 알려진 link 함수의 초등적 성질이지 새 정리가 아니다. 중심화는 delta의
range를 줄이지 않는다. 실제 학습 편향/규제 변경이 있어야 null 반응을 줄일 수 있고,
그 변경이 active 의존성도 함께 훼손할 가능성을 평가해야 한다.

`copy_base`는 c의 선형 함수인 반면 route의 평균은 c에 비선형 함수가 될 수 있다.
유용한 이력 보정이 route로 들어갈 구조적 여지가 있다는 뜻이다. 하지만 c를 만드는
GRU 자체가 비선형이므로 base의 표현력이 반드시 부족하다는 증명은 아니다.
**비선형 history-only 대조군**과 규제 없는 같은 분리 구조 대조군이 후속 비교에 필요하다.

관측 입력 gap를 바꾼 반응도 인과적 `do(gap)` 효과가 아니다. DGP의 latent gap
regime이 gap와 active copy를 함께 바꾸므로, 현재 분석은 **예측적 조건부 의존성**을
평가한다. 역사적인 `intervention`/`noncausal cell` 이름은 이 범위로 해석한다.

## 3. 데이터 전처리와 공정성 점검

| 점검 | 코드/실행에서 확인한 내용 | 남는 한계 |
|---|---|---|
| train-only fit | [SAFTensorizer.fit](../../data/cof_seqgen_saf_tensorizer.py)이 train entity를 먼저 고른 뒤 vocab/scaler/support를 fit | 모든 미래 데이터셋에서 자동 무누출이라는 보장은 아님 |
| split | 고정 entity 분할 유지, 현 cache train/validation ID 겹침 0 | validation은 이미 반복 사용, 확증적 test가 아님 |
| 시간 이력 | GRU 출력 한 칸 shift; 첫 gap NaN; 절대 원점 대신 gap 사용 | teacher-forced 분석은 누적 생성 오류를 평가하지 않음 |
| 범주 | 관측 label 0/1→code 3/4, PAD/MISSING/UNK와 분리; mark 64종 | 미지의 집단/open vocabulary 일반화 미검증 |
| gap bin | train 경험 질량으로 대표값을 고르고, 인접 대표값의 중점으로 boundary 구성 | 동일 빈도 bin이라고 가정할 수 없음; 이번 pi_s는 실제 bin count로 계산 |
| support | 대표값은 실제 train gap에서 선택, roundtrip PASS | DGP는 연속 gap이므로 31값은 양자화 근사; population support 회복/연속밀도 보장은 아님 |
| numeric | train-only signed_log1p_zscore, 역변환 경로 존재 | 변환 공간 NLL을 원시 단위의 density NLL로 직접 비교할 수 없음 |
| prevalence | [materializer](../../data/cs_saf_prevalence.py)가 기존 label 난수 stream으로 nested context view 구성 | 여러 prevalence는 독립 데이터 반복이 아님 |
| active marks | kappa=1에서 새 active entity만 기존 gap regime 조건으로 mark 재생성 | latent는 DGP 구성에만 사용; 학습 입력/손실에는 전달하지 않음 |
| amount | 이 controlled DGP의 독립 numeric value는 mark 재생성 때 유지 | mark와 value가 의존하는 다른 DGP에는 같은 절차를 그대로 적용할 수 없음 |
| length/가중 | entity 길이 2–32, overlapping window 없이 전체 사용 | 임의 길이·다중 static·일반 relational schema를 검증한 것이 아님 |

실행 evidence에서 31,951 train entity(희소 1,545), 6,846 validation(희소 331),
train transition 734,984를 재확인했다. 이번 분해의 기준 빈도는 nonfirst train
transition으로만 계산했고 모든 checkpoint/validation에 고정했다.
외부 baseline 비교에서는 같은 split/정적 조건/길이 계획/생성 수/접근 정보가 필요하다.
연속 gap baseline과 31-bin 모델의 raw likelihood를 같은 단위로 직접 순위 매기면 안 된다.
통일된 원시 gap fidelity와 조건부 관측 분포 오차를 함께 비교해야 한다.

## 4. 선행연구와 실제로 남는 차별점

아래는 2026-09-16에 확인한 저자 논문 또는 공식 문서다. 논문에 보고된 다른
데이터셋의 성능 수치를 여기 baseline 실험 결과로 가져오지 않았다.

| 선행연구 | 이미 다룬 내용 | CS-SAF가 추가로 증명해야 할 부분 |
|---|---|---|
| [TabDiff, ICLR 2025](https://arxiv.org/abs/2410.20626) | 수치/범주 혼합 tabular diffusion | 현 CS-SAF는 다른 생성기 계열. 저장소 기반만으로 개선판/우위라고 할 수 없음 |
| [TabDiT, 2025](https://arxiv.org/html/2504.07566v2) | row VAE와 latent diffusion transformer로 이질적 가변 길이 tabular sequence 생성 | 혼합형 시퀀스 생성 자체는 차별점 아님; context별 의존성 정확도와 null 선택성을 실증해야 함 |
| [REaLTabFormer, 2023](https://arxiv.org/abs/2302.02041) | parent table 및 parent-conditioned relational sequence 생성 | static 조건부/자가회귀 생성만으로는 새롭지 않음 |
| [TabularARGN, 2025](https://arxiv.org/html/2501.12012v2) | context와 history를 쓰는 tabular autoregressive 생성, sequential 평가 | 기존 연구가 전부 주변분포만 본다는 주장은 부정확; 동일 조건에서 좁은 실패 모드를 비교해야 함 |
| [SDV PARSynthesizer/CPAR](https://docs.sdv.dev/sdv/sequential-data/modeling/parsynthesizer) | probabilistic autoregressive 시퀀스 합성 | 직접적인 실용 baseline. 동일 조건/예산의 비교가 필요 |
| [Enguehard et al., 2020](https://proceedings.mlr.press/v136/enguehard20a.html), [Waghmare et al.](https://arxiv.org/abs/2210.15294) | 시간과 mark의 결합/조건부 의존성을 명시한 neural point process | time→mark 의존성을 처음 제안했다는 주장 불가. 이 계열도 생성모델이며 예측만 한다고 구분하면 안 됨 |
| [See et al., ACL 2017](https://aclanthology.org/P17-1099/) | copy와 vocabulary generation을 혼합한 pointer-generator | 대상/복사 위치는 다르지만 copy/new 혼합 자체는 독창성 근거가 아님 |
| [Lengerich et al., AISTATS 2020](https://proceedings.mlr.press/v108/lengerich20a.html) | main/interaction 식별과 functional-ANOVA purification | 이번 train-reference centering은 새 분해 이론이 아님; 실제 진단과 검증된 해결책이 기여여야 함 |
| [Sagawa et al., GroupDRO](https://arxiv.org/abs/1911.08731) | group robust objective와 과적합/규제 문제 | B는 equal-group average로 다른 목적함수. 그 문헌이 이번 실패의 원인을 입증하지는 않음 |
| [Umesh et al., HFGF, 2025 preprint](https://arxiv.org/abs/2507.19211) | 지정된 functional/logical rules를 이용한 dependent feature 생성/복원 | 알려진 결정 규칙 대신 관측 이력에서 확률적 전이를 배운다는 구분은 가능하나 성능 근거 필요 |
| [Sajja, 2026 preprint](https://arxiv.org/abs/2604.13125) | fraud synthetic data의 timing/burst/graph/velocity 행동 보존 평가 | 평균 fidelity/utility가 행동 실패를 놓친다는 주장만으로는 차별성 부족. 본 연구의 rare-context active/null 조건부 분포 오차에 초점을 좁혀야 함 |
| [Kang, UAI 2026](https://proceedings.mlr.press/v337/kang26a.html) | relevance-weighted conditional diffusion으로 희소 회귀 구간 합성 | 희소 표본 가중치와 생성의 결합만으로 독창성 부족; 현재 목표인 이력 조건부 의존성과 null 보존의 차이를 증명해야 함 |
| [Tugnoli et al., UAI 2026](https://proceedings.mlr.press/v337/tugnoli26a.html) | TabPFN 생성에 DAG/PDAG 조건을 반영, 구조/ATE 보존 평가 | TabPFN은 생성 불가라고 일반화하면 안 됨. 본 연구는 알려진 causal graph/ATE를 목표로 하는 방법이 아님 |

특히 Sajja 문헌은 arXiv의 **DMLR 제출 preprint**로 확인했으며 게재 확정 논문으로
표기하지 않는다. 그 논문의 정리/수치까지 우리가 독립 검증했다는 뜻도 아니다.
새 문헌은 기존 frozen 실험 계약을 소급 변경할 이유가 아니라, 현재 차별성 설명을
정확히 제한하고 이후 비교 설계를 갱신할 이유다.

## 5. 실제 baseline 실행 상태

근거는 [registry](../../generators/cof_seqgen_saf_baselines.py),
[잠금 계약](../../configs/benchmark_v2/cof_seqgen_saf_baseline_lock.yaml),
[이전 실행 기록](../benchmark_v2/cof_seqgen_saf_execution_recovery_status_2026_08_28.md)와
cofseq 환경의 설치 메타데이터 확인이다. 옛 `NOT_ATTEMPTED` 문서 하나만 보고 현재
상태를 판단하지 않았다. 이번 작업에서는 baseline을 새로 학습하지 않았다.
현재 worktree의 문서상 CPAR artifact 경로에는 과거 원본 report가 없으므로,
그 실행 이력은 versioned 보고서에 근거한다. 이번에 그 CPAR 실행을 독립 재현하거나
원본 checksum을 재검증한 것은 아니다. 설치 메타데이터와 경로 확인은 최신 evidence의
`supplemental_context_audit`에 따로 기록했다.

| 대상 | 실제 확보된 상태 | 현재 CS-SAF와 공정한 비교 완료? |
|---|---|---|
| 내부 U/A/B | 같은 architecture/초기값/순서/예산의 pi=.05 단일 seed 6 fits | 해당 loss 비교만 완료; 새 방법 우위 아님 |
| CPAR | SDV 1.38.0 wrapper/환경, 이전 SAF 연구에서 kappa=1 20-epoch 단일 full fit 기록 | 아니오; 현재 CS-SAF와 동일 프로토콜의 최종 비교 아님 |
| REaLTabFormer | 0.2.4 설치, wrapper와 smoke 검증 기록 | 아니오 |
| TabularARGN | pinned upstream/adapter 경로; 주 cofseq 환경에 mostlyai 미설치, 별도 runtime 필요 | 아니오 |
| TabDiT | 잠근 소스에서 실행 가능한 train/sample generator가 registry상 BLOCKED, reported-only | 아니오; 실행한 baseline처럼 성능표에 넣지 않음 |
| empirical sequence sampler | 이전 memorization/control 구현 및 실행 기록 | 구조적 대조군이며 새로운 신경망 baseline 우위 근거 아님 |
| CTGAN/TVAE/GaussianCopula | CTGAN 0.12.1/SDV 환경 존재; flat control 역할 | 주 sequential 모델 순위와 분리해야 함 |
| TabPFN | 기존 계약에서는 utility estimator 역할 | 그 역할 선택이 TabPFN 생성 연구의 부재를 뜻하지 않음 |

향후 외부 실행 시 버전/설치/adapter/공식 구현 가능성을 다시 고정해야 한다.
주 baseline은 CPAR/REaLTabFormer/TabularARGN 같은 실제 순차 생성기여야 하고,
flat 모델만 이겨서 sequential SOTA를 주장할 수 없다. 내부 비교에는 충분한
history-only 용량, routed U, 같은 용량의 centered-unregularized 모델이 필요하다.

## 6. 현재까지의 주장과 근거 수준

| 주장 | 현재 판정 |
|---|---|
| train 데이터/gap 표현에 구별 가능한 신호가 있다 | oracle audit이 지지. 학습 성공과 구별 |
| support alignment가 선택한 train 대표값 밖 출력을 막는다 | 구조 및 실행 확인. 연속분포의 정확한 복원과 구별 |
| CS-SAF가 active current-gap 반응을 학습했다 | 저장된 pilot에서 확인. 정확한 oracle 분포를 회복했다는 뜻은 아님 |
| active를 보존하면서 모든 null에서 안전하다 | v1/v2/U/A/B pilot 모두 해당 조건에서 실패 |
| 희소화 때문에 의존성이 희석된다 | 아직 입증하지 못함. prevalence×seed 정확도 분석 미완료; 최신 주요 실패는 spurious null response |
| route 전체를 끄면 안 되는 이유가 있다 | 고정 가중치 아래 평균 성분의 유용성 확인 |
| null residual은 항상 해롭다 | 사전등록 조건 미충족: train/validation 부호가 다름 |
| centered history/residual 모델이 해결한다 | 아직 후보. 새 학습·성능 결과 없음 |
| balanced auxiliary가 필수이며 우수하다 | 미입증. active 이득과 null 악화의 tradeoff 관찰 |
| 외부 baseline보다 좋고 논문 기여가 확정됐다 | 미입증. 현재 직접 비교/다중 seed/실데이터/held-out 결과 없음 |

이전 protocol의 `conditional mechanism dilution`은 연구 가설로 남긴다.
현재 관찰을 그 가설의 성공 사례로 재명명하지 않는다. 관측된 문제를 더 정확히 쓰면
**“희소 context의 조건부 의존성을 보존하면서, 비활성 context의 가짜 의존성을
억제하는 선택성”**이다. 이 표현도 새 분야를 처음 정의했다는 주장은 아니다.

## 7. 발전시킬 수 있는 기여와 필요한 다음 실험

가능한 방법 후보는 `history-only correction + train-centered gap residual`을
명시하고 residual에만 규제를 주는 것이다. 이번 증거는 설계 방향을 제안하지만
구체적 penalty/lambda/학습 예산을 결정한 사전등록은 아니다.
다음 모델 계약에서 아래 혼동을 먼저 통제해야 한다.

1. **표현과 규제 분리:** 충분한 history-only head, 같은 용량의 분리 구조(규제 없음),
   그 구조+residual 규제를 비교. 단순 centering 재표현이면 함수 보존을 검사한다.
2. **oracle 없이 선택성 학습:** 양 집단에 같은 규칙을 적용. 알려진 active/null mask로
   끄고 켜는 방법은 제외. 미래 데이터의 활성 여부를 안다는 가정도 하지 않는다.
3. **학습 가능성과 일반화 분리:** 새 후보의 CPU 검증 후 등록된 pilot. response range
   통과만으로 승격하지 않고, 관측 가능한 조건부 분포 오차·null 오차·전체 fidelity를
   동시에 확인. 이번 validation 반복 사용을 탐색 단계로 기록한다.
4. **희소성 인과 설명에 필요한 통제:** prevalence 변화는 active 표본 수도 바꾼다.
   희소 비중 효과와 절대 표본 수 효과를 분리할 설계가 필요하다. 전체 prevalence
   grid/다중 seed의 실패도 보존하고 balanced 이득이 없으면 기여에서 제거한다.
5. **구조 편향과 외부 타당성:** 현재 copy DGP와 유사한 decoder의 이점을 인정하고,
   다른 fresh 분포/전이 규칙에서도 확인. CPAR/RTF/ARGN에 동일 조건과 공정한 계산
   예산을 제공하고, 실데이터의 행동·조건부·utility·privacy를 따로 검증한다.

후속 모델이 검증에 성공하면 가능한 기여는 단순한 gate/균형 손실의 조합이 아니라,
**유용한 이력 보정과 gap 변화를 구별해야 하는 실패 분석, 그 분석에 기반한 선택성
개선 방법, active/null의 관측 분포 정확도를 함께 검증하는 증거**의 결합이다.
개선 폭이 작거나 여러 seed/외부 baseline에서 반복되지 않으면 모델의 우위 주장을
축소해야 한다. 현재 결과는 계속 연구할 근거이지 성공이나 게재 가능성의 보장은 아니다.
