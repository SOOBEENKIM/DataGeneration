# CS-SAF 문헌 재검토와 연구 방향 판단 — 2026-09-20

> **공식 소스 확인 후 정정 (2026-09-20):** TabDiT HEAD `cbb2b9a`에는 평가 코드와 생성 데이터만 있고, VAE/DiT 학습·생성 구현이 없습니다. 아래의 TabDiT 직접 실행 계획은 코드 확보를 전제로 한 보류 항목입니다. 우선 실행 대표 모델은 공식 TabularARGN 2.4.0으로 변경했습니다. 이 제약은 TabDiT의 성능 한계가 아닙니다. [소스 목록](external_audit_v1/source_inventory.json).

이 문서는 문헌 검토와 설계 판단이다. 새 모델의 성능 보고서나 실험 사전등록이 아니다. 검토 시점의 연구 기준은 `research/cs-saf`의 `b9d4162`이며, 기존 결과·실패 판정을 바꾸지 않는다.

## 먼저 결론

**현재 E와 간격별 보정 결과만으로 새로운 생성 방법의 우수성을 주장하기에는 부족하다. 그러나 연구 질문을 더 정확하게 정의하고 새로운 학습 원리를 검증할 여지는 있다.**

핵심 문제를 “기존 모델은 이력을 사용하지 않는다” 또는 “관계를 보존하지 않는다”로 쓰면 안 된다. 이미 이질적인 순차 표를 생성하는 모델, 시간–행동의 의존성을 모델링하는 모델, 관계 보존을 평가하는 연구가 있다. 앞으로 검토할 질문은 다음과 같다.

> 이력에 조건화한 한 단계 예측의 개선이 실제 반복 생성에서 관계 보존으로 이어지지 않을 때, 개별 변수의 예측과 변수 간 의존성의 수정을 분리하면서, 생성 이력의 변화까지 고려해 관계 왜곡을 줄일 수 있는가?

이 질문 자체의 완전한 신규성을 확인한 것은 아니다. 아래 문헌과 구체적으로 다른 알고리즘·분석·결과를 만들어야 방법론 기여가 된다. **E를 살리는 것이 목적은 아니며, E는 비교 대상으로 남긴다.**

## 1. 검토 범위와 읽기 수준

- 기준일: 2026-09-20. 2024–2026년의 순차 표 생성, marked temporal point process, 생성 이력의 분포 차이, 의존성/주변분포 분리, 금융 시계열·거래 데이터 생성을 중심으로 검색했다. 필요한 과거 기초 연구도 포함했다.
- arXiv 원문, 학회 proceedings, 저널 출판사, 저자 공식 코드/페이지를 근거로 사용했다. 검색 결과의 요약만으로 구조나 정리를 확정하지 않았다.
- **본문 검토:** 아래에 명시한 방법·실험·한계 절을 읽은 경우. 모든 부록 증명을 독립 검증하거나 코드를 재현했다는 뜻은 아니다.
- **부분 검토:** 초록, 공식 서지, 본문 일부 또는 검색에 노출된 저자 원문만 확인한 경우. 방법의 완전한 이해나 성능 재현을 주장하지 않는다.
- 이 목록은 관련성이 높은 문헌을 추린 결과이며 모든 CS·금융 논문을 빠짐없이 조사한 체계적 문헌고찰은 아니다. 서로 다른 데이터·예산의 논문 수치를 하나의 성능 순위로 합치지 않았다.

## 2. 현재 결과와 먼저 연결하기

[최근 실험 보고서](gap_calibration_v1_report_2026_09_20.md)에 따른 자유 생성 반복관계 L1이다. 낮을수록 좋다.

| 활성 집단 비율 | U+기존 보정 | U+간격 보정 | E+기존 보정 | E+간격 보정 |
|---|---:|---:|---:|---:|
| 5% | 0.031552 | 0.028078 | 0.035293 | 0.029126 |
| 10% | 0.030656 | 0.028086 | 0.035597 | 0.029669 |
| 25% | 0.028794 | 0.025740 | 0.030726 | 0.026597 |
| 50% | 0.018003 | 0.014090 | 0.018022 | 0.014730 |

- 같은 보정으로 U와 E 모두 개선됐다. E의 별도 이력 계수가 있어야만 얻는 효과는 아니다.
- 같은 보정 후 E의 생성 오차는 U보다 평균 3.3–5.6% 높다. 그러나 차이의 참고 95% 구간은 모든 비율에서 0을 포함하므로, E가 언제나 통계적으로 열등하다고 단정하지 않는다.
- E는 활성 조건부 행동분포 정확도와 비활성 반응에서 U보다 유리한 결과를 유지한다. **조건부 예측과 자유 생성의 순위가 다르다.**
- 새 보정은 U/E 모두에서 비활성 조건의 평균 반응을 높였다. 관계 강화와 억제를 함께 해결했다는 결과는 아니다.
- 인공 데이터 생성 seed42와 반복 검토한 validation을 재사용했다. 5개 학습 시드와 5개 생성 난수는 새 데이터에서의 독립 재현과 다르다.
- 외부 CPAR 비교는 이미 40회 수행했다. 다만 학습 예산·선정 기준이 다르고 현재 인공 법칙이 내부 copy 구조에 유리하다. “외부 비교가 전혀 없다”도, “외부 모델 대비 우수성이 충분히 입증됐다”도 부정확하다.
- 현재 구현은 GRU 기반 gap→mark→amount 자기회귀 생성기이다. 원래 CoF-SeqGen의 모든 계획을 구현한 diffusion 모델이 아니다. 기본 U도 과거 이력을 사용한다. E는 반복확률 경로에 이력 보정 계수를 추가한 내부 변형이다.

## 3. 직접적인 CS 비교 대상

### 3.1 TabDiT — ICLR 2025

**Diffusion Transformers for Tabular Data Time Series Generation**, Garuti et al.

- **기여/방법:** 각 거래 행을 별도 VAE로 압축하고, 행 잠재표현의 시퀀스를 DiT로 함께 생성한다. 이질적인 수치·범주형 필드와 가변 길이를 다루기 위한 숫자 표현, 행 내부 AR decoder, 종료행 표현을 포함한다.
- **근거:** 금융 거래를 포함한 6개 공개 데이터, 조건부/비조건부 생성, 시퀀스 판별·하위 예측 효용, AR·REaLTabFormer 등의 비교.
- **우리에게 의미:** 가장 직접적인 강한 외부 비교 대상이다. “시퀀스에 diffusion을 붙인다”나 “이질적 거래의 관계를 학습한다”는 것만으로 차별화되지 않는다.
- **읽기:** 본문 §4–5 및 결론. 행 내부 AR과 시간축 전체 잠재 시퀀스 diffusion을 구분했다.
- [학회 원문](https://proceedings.iclr.cc/paper_files/paper/2025/file/e90ba1fc564a69809d7391bf76a5f087-Paper-Conference.pdf) · [본문](https://arxiv.org/html/2504.07566) · [공식 저장소—현재 평가 코드만 공개](https://github.com/fabriziogaruti/TabDiT)

### 3.2 TabularARGN — 2025 preprint

**TabularARGN: A Flexible and Efficient Auto-Regressive Framework for Generating High-Fidelity Synthetic Data**, Tiwald et al.

- **기여/방법:** 필드와 시간축 양쪽의 자기회귀 생성, LSTM 이력 표현, 정적 parent context, 불규칙·가변 길이 시퀀스 지원. 모든 사용 모드가 행 독립인 모델이 아니다.
- **근거:** 순차 표와 parent–child 데이터 비교, 효율·coherence 평가. 최신 본문의 순차 비교에는 REaLTabFormer, RC-TGAN, ClavaDDPM이 포함된다.
- **우리에게 의미:** U보다 훨씬 일반적인 시퀀스 대조군이다. 전체 입력 이력·정적 정보·길이 조건을 맞춰 비교해야 한다.
- **읽기:** §3.5–3.6, §4.2. 별도 최종 학회 채택은 이 검토에서 확인하지 않았다.
- [원문](https://arxiv.org/html/2501.12012) · [공식 구현 문서](https://mostly-ai.github.io/mostlyai-engine/)

### 3.3 Preventing Conflicting Gradients in Neural Marked Temporal Point Processes — TMLR 2025

Bosser and Ben Taieb.

- **기여/방법:** 시간과 mark 손실의 기울기 충돌을 분석하고 decoder 또는 encoder까지 파라미터를 분리한다. 파라미터 분리는 시간과 mark가 독립이라는 가정과 다르다.
- **근거:** 여러 TPP 구조와 5개 실제 데이터, 용량 증가·목적함수 영향의 대조 실험.
- **우리에게 의미:** “규제와 예측 특징을 분리한다” 또는 “시간과 행동의 경로를 나눈다”만으로 새 기여를 주장할 수 없다. 실제 gradient 충돌과 단순 용량 효과를 분리해야 한다.
- **읽기:** §3–4 방법, 실험 설계. 모든 결과를 자체 재현하지 않았다.
- [TMLR 논문](https://openreview.net/pdf?id=INijCSPtbQ) · [본문](https://arxiv.org/html/2412.08590)

### 3.4 ADiff4TPP — TMLR 2026

**ADiff4TPP: Asynchronous Diffusion Models for Temporal Point Processes**, Mukherjee et al.

- **기여/방법:** event 잠재표현에서 앞선 사건과 뒤 사건에 서로 다른 노이즈 진행을 부여한다. 비동기 diffusion/flow 학습으로 순서를 반영하면서 여러 미래 사건의 생성·예측을 다룬다.
- **근거:** 다음 사건 및 긴 구간 예측, 동기/자기회귀형 스케줄 등 ablation.
- **우리에게 의미:** 반복 생성 문제를 joint event generation으로 푸는 대안이 이미 있다. 현재 전체 거래 schema에 바로 호환되는지는 별도 확인해야 한다.
- **읽기:** §3–5. 최종 상태는 TMLR 표지로 확인했으며 다른 학회 submission과 혼동하지 않는다.
- [최종 논문](https://openreview.net/pdf?id=bwnZW4wXh4) · [본문](https://arxiv.org/html/2504.20411)

### 3.5 EdiTPP — ICLR 2026

**Edit-Based Flow Matching for Temporal Point Processes**, Lüdke et al.

- **기여/방법:** 사건의 삽입·삭제·교체를 연속시간 Markov chain으로 모델링하고 flow matching으로 학습한다. 순서와 가변 사건 수를 생성 과정의 연산에 반영한다.
- **근거:** 실제 7개·인공 6개 데이터, 5개 시드, 기존 TPP 생성기 비교.
- **우리에게 의미:** 출력의 유효성을 생성 연산에 반영한 예다. 검토한 버전은 주로 unmarked event times이므로, amount·다범주 mark를 가진 우리 과제의 직접 baseline으로 그대로 부르지 않는다.
- **읽기:** §3 방법, §4 설계와 주요 비교.
- [저자 공식 페이지](https://www.cs.cit.tum.de/daml/editpp/) · [원문](https://arxiv.org/html/2510.06050)

### 3.6 TabDiff — ICLR 2025

**TabDiff: a Mixed-type Diffusion Model for Tabular Data Generation**, Shi et al.

- **기여/방법:** 연속형·범주형의 서로 다른 확산 과정과 학습 가능한 feature별 노이즈를 한 tabular 생성 모델에 결합한다.
- **근거:** 여러 표 데이터에서 fidelity·utility 등 비교.
- **우리에게 의미:** 혼합형 표현의 참고 문헌이다. flat table 성능을 시간축 의존성 처리 성능으로 해석하면 안 된다. 현재 CS-SAF와는 모델 계열도 다르다.
- **읽기:** 초록·본문 개요 중심의 부분 검토. 전체 학습·sampler 증명 재검토는 하지 않았다.
- [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5c882988ce5fac487974ee4f415b96a9-Abstract-Conference.html) · [본문](https://arxiv.org/html/2410.20626)

### 3.7 TabCascade — ICML 2026

**Cascaded Flow Matching for Heterogeneous Tabular Data with Mixed-Type Features**, Mueller, Gruber and Fok.

- **기여/방법:** 범주/수치의 거친 구조를 먼저 생성하고 조건부 연속 flow로 값을 세분화한다. 이산 원자와 연속 분포가 섞이는 문제를 명시적으로 다룬다.
- **근거:** 표현·수송 오차 분석 및 구성요소 비교. 정리에는 명시적인 표현/근사 가정이 있다.
- **우리에게 의미:** support와 분포 형태의 문제를 수식과 구조로 연결하는 예다. 우리 31개 gap 대표값 표현은 원래 연속 분포의 완전한 복원이 아니다.
- **읽기:** §4 방법·정리, §5 비교 설계. [저자 arXiv v3 서지](https://arxiv.org/abs/2601.22816)와 [ICML 2026 공식 논문 목록](https://icml.cc/Downloads/2026)에서 정식 제목을 확인했다.
- [원문](https://arxiv.org/html/2601.22816) · [공식 코드](https://github.com/muellermarkus/tabcascade)

### 3.8 Unified Flow Matching for Long Horizon Event Forecasting — 2025 preprint

Xiao Shou.

- **기여/방법:** 연속 시간 간격과 이산 mark를 flow matching 체계에서 예측한다. 긴 구간 생성의 효율을 강조한다.
- **한계/의미:** 결론에 미래 사건 사이의 조건부 독립 가정이 명시된다. “joint”라는 명칭만으로 모든 미래 의존성을 보존한다고 읽어서는 안 된다.
- **읽기:** 방법 개요·결과·결론의 가정. 최종 채택 venue는 확인하지 않았다.
- [저자/서지](https://arxiv.org/abs/2508.04843) · [본문](https://arxiv.org/html/2508.04843)

## 4. 관계 보존과 생성 이력 문제의 가까운 연구

### 4.1 TabStruct — ICLR 2026

**TabStruct: Measuring Structural Fidelity of Tabular Data**, Jiang, Simidjievski and Jamnik.

- **기여/방법:** 다른 변수로 각 변수를 예측하는 global utility와 조건부 독립 구조 보존을 연결하여 관계 소실·가짜 관계를 평가한다.
- **근거:** 13개 생성기, 29개 데이터(인공 SCM 포함). global utility와 구조 지표 관계는 실증 결과이며 보편적인 동치 정리로 읽지 않는다.
- **우리에게 의미:** “없는 관계까지 평가한다”는 것만으로 새 평가 기여가 되지 않는다. 시퀀스 생성 이력, 희소 집단, 오류가 누적되는 조건에 대한 추가 분석이 필요하다.
- **읽기:** §3 지표와 §4 설계·주요 결과.
- [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/447ac93bf22099aa346a45577376492d-Abstract-Conference.html) · [원문](https://arxiv.org/html/2509.11950)

### 4.2 Seq2Synth — 2026-08 v2 preprint

**Seq2Synth: Benchmarking Temporal Fidelity in Synthetic Sequential Tabular Data**, Kwon et al. v1 제목은 **Do Generative Models Keep Time?**였다.

- **기여/방법:** 시간 표현·불규칙성·개체 간 의존성·schema로 적용 가능한 평가를 정하고, timestamp·같은 시점의 집단 구조·개체 내 궤적·관계형 구조를 나눠 평가한다.
- **근거:** 13개 데이터 중 7개 core와 8개 생성기. 정적 분포 평가의 순위가 시간 평가와 다름을 보인다. 궤적 기반 utility/privacy도 포함한다.
- **우리에게 의미:** “행 단위 분포는 좋지만 시퀀스는 틀린다”는 문제 제기는 이미 가까운 선행연구가 있다. 실제 생성의 오류를 설명하고 줄이는 방법이 더 필요하다.
- **주의:** 본문의 TabDiT 설명을 그대로 채택하지 않았다. Seq2Synth는 시간축 AR처럼 설명하지만 TabDiT 원문은 시퀀스 잠재 diffusion과 행 내부 AR을 구분한다. 구현 확인 없이 실패 원인을 AR 탓으로 단정하지 않는다.
- **읽기:** §3 평가, §4 결과, §5 논의. 검색 요약의 YNAB/MIMIC 등은 v2의 데이터 목록과 달라 사용하지 않았다.
- [v2 원문](https://arxiv.org/html/2607.15606v2) · [공식 코드](https://github.com/KiwanKwon/Seq2Synth)

### 4.3 TACTiS-2 — ICLR 2024

**TACTiS-2: Better, Faster, Simpler Attentional Copulas for Multivariate Time Series**, Ashok et al.

- **기여/방법:** 주변분포와 의존성 copula를 나누고 두 단계로 학습한다. 무제약 공동 학습의 식별·유효성 문제를 다룬다.
- **근거:** 방법 정리와 실험. 정리의 연속 주변분포·충분한 모델 용량 등의 가정은 이산 mark에 자동으로 이전되지 않는다.
- **우리에게 의미:** “주변분포와 관계를 분리한다”, “주변분포를 동결한다” 자체는 새 기여가 아니다. 실제 생성의 피드백과 관계 수정 비용까지 다뤄야 한다.
- **읽기:** §2–4 방법·정리 중심. 결과 전체 재검증은 하지 않았다.
- [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/63796148c99205adb0fcac069cc714d4-Abstract-Conference.html) · [본문](https://arxiv.org/html/2310.01327)

### 4.4 Self Forcing — NeurIPS 2025

**Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion**, Huang et al.

- **기여/방법:** 모델 자신이 생성한 이전 프레임으로 학습을 진행하고 생성 분포를 맞춘다. 계산 가능하게 만드는 짧은 diffusion·KV cache·gradient 절단이 결합된다.
- **우리에게 의미:** 자기 생성 이력으로 학습하는 아이디어 자체는 새롭지 않다. 이산 행동·희소 관계에서 학습 신호를 어떻게 추정하고 편향을 통제할지가 별도 문제다. 영상 teacher 기반 손실을 그대로 가져올 수 없다.
- **읽기:** §3.2–3.3 학습 알고리즘과 목적함수 중심.
- [NeurIPS 논문](https://proceedings.neurips.cc/paper_files/paper/2025/file/f4823f831af67a3ef15e41a85434422a-Paper-Conference.pdf) · [본문](https://arxiv.org/html/2506.08009)

### 4.5 Amortized Vine Copulas for High-Dimensional Density and Information Estimation — 2026 preprint

- **기여/방법:** PIT 후 2차원 histogram을 신경망으로 추정하고 IPFP/Sinkhorn으로 주변 제약을 맞춘 copula를 vine으로 결합한다. 학습한 추정기를 재사용하는 amortization이 핵심이다.
- **우리에게 의미:** 의존성 행렬에 신경망과 Sinkhorn을 붙이는 것만으로 새 구조가 되지 않는다. 유한 반복의 주변 제약 오차도 보고해야 한다.
- **읽기:** §3 방법 중심. 이 검토에서 독립 구현·성능 재현은 하지 않았다.
- [원문](https://arxiv.org/html/2604.20568)

### 4.6 Preserving Temporal Dynamics in Time Series Generation — 2026 preprint

- **기여/방법:** GAN 출력에 경험적 차분 분포를 이용한 MCMC형 보정을 적용한다. 여러 GAN·시계열에서 동학 관련 지표를 비교한다.
- **검토상 한계:** Algorithm 1은 시점별 실제 관측값을 사용하며, 일반 비대칭 proposal의 MH 보정비가 그대로 드러나지 않는다. 정확한 target 수렴·자유 생성 호환성을 별도 검증해야 한다.
- **우리에게 의미:** “생성 후 관계를 보정한다”도 기존 방향이다. 우리 비교에는 평가 시점 정답을 사용하지 않는지와 실제 생성 조건이 동일한지 확인해야 한다.
- **읽기:** §IV 방법·알고리즘 및 §V 평가 정의. 논문의 이론 주장을 독립 검증된 사실로 채택하지 않았다.
- [원문](https://arxiv.org/html/2604.27182)

### 4.7 COT-GAN — NeurIPS 2020, 기초 문헌

- **기여/방법:** 시간의 정보 제약을 가진 causal optimal transport를 적대적 순차 생성 목적함수로 구현한다. mini-batch Sinkhorn 편향을 줄이기 위한 방법도 제시한다.
- **우리에게 의미:** “순차 분포를 OT로 맞춘다”도 이미 있는 접근이다. 여기서 causal은 시간상 이용 가능한 정보 제약이며 사기 행동의 인과효과를 식별한다는 뜻이 아니다.
- **읽기:** 공식 초록·서론·알고리즘 관련 일부. 새로운 전면 재독은 하지 않았다.
- [공식 논문](https://proceedings.neurips.cc/paper/2020/file/641d77dd5271fca28764612a028d9c8e-Paper.pdf)

## 5. 금융 저널·AI 금융 논문에서 배울 점

### 5.1 Time-Causal VAE — SIAM Journal on Financial Mathematics, 2026

**Time-Causal VAE: Robust Financial Time Series Generator**, Acciaio, Eckstein and Hou.

- **기여/방법:** 시간상 미래 정보를 사용하지 않는 encoder/decoder와 flow prior를 구성하고 causal Wasserstein과 학습 오차를 연결한다.
- **이론·결과:** 가정 아래 생성 분포의 차이가 다단계 최적화 가치에 주는 영향을 분석한다. Black–Scholes, Heston, 경로 의존 변동성 및 S&P500/VIX에서 분포·의사결정·다양성을 평가한다.
- **주의:** Lipschitz/convexity/유계 전략 등 가정이 있다. 일방향 causal 거리의 한쪽 부등식과 adapted 거리의 양쪽 부등식을 혼동하면 안 된다. 모든 탐지기의 성능 보장은 아니다.
- **우리에게 의미:** 모델 제약→오류의 의미→실제 활용을 연결하는 좋은 논문 구성이다. 사기 탐지 연구라면 마지막 연결을 실제 held-out 탐지 효용으로 검증해야 한다.
- **읽기:** §2 구조, §3 정리와 가정, §4 평가 설계 및 시장 적용.
- [저널](https://epubs.siam.org/doi/abs/10.1137/24M1711650) · [공개 본문](https://arxiv.org/html/2411.02947)

### 5.2 Generation of synthetic financial time series by diffusion models — Quantitative Finance, 2025

Takahashi and Mizuno.

- **기여/방법:** 수익률·spread·거래량을 wavelet 이미지로 변환해 DDPM으로 생성한다.
- **결과/한계:** AAPL 분봉 데이터에서 tail·자기상관·장중 패턴·교차상관을 비교한다. 공개 원고에서는 날짜를 섞어 train/validation을 나누고 10σ winsorization을 사용한다. 원래 극단 tail 전체나 시간 밖 일반화를 입증한 것으로 확대하지 않는다.
- **우리에게 의미:** 금융 관계를 구체적인 수치로 평가하는 예다. 모든 지표에서 압도적이지 않아도 장점과 한계를 명확히 제시할 수 있다. 다만 이것이 우리 실패 기준을 사후 변경할 근거는 아니다.
- **읽기:** §3–4.
- [저널](https://www.tandfonline.com/doi/abs/10.1080/14697688.2025.2528697) · [공개 본문](https://arxiv.org/html/2410.18897)

### 5.3 CoFinDiff — IJCAI 2025 AI4Tech 특별 트랙

**CoFinDiff: Controllable Financial Diffusion Model for Time Series Generation**, Tanaka et al.

- **기여/방법:** 추세와 실현 변동성을 조건으로 wavelet 기반 금융 경로 diffusion을 학습하고 극단 구간을 더 자주 학습한다.
- **근거:** 일본 주식 11개 종목 분봉, 조건 일치·stylized facts·다양성·deep hedging 평가. 모든 비교에 동일한 강도의 최신 baseline이 포함된 것은 아니다.
- **우리에게 의미:** 생성 정확도 다음에 구체적 금융 목적을 배치한다. 규칙을 만족했다는 결과와 실제 의사결정 효용을 구분한다. rare upsampling 자체도 신규성이 아니다.
- **읽기:** §3 구조·§4 설계·§5 결과 일부. deep hedging 전체 수치를 재검산하지 않았다.
- [공식 proceedings](https://www.ijcai.org/proceedings/2025/1040) · [본문](https://arxiv.org/html/2503.04164)

### 5.4 Probabilistic Multivariate Time Series Forecasting with Diffusion Copulas — 2026 preprint

- **기여/방법:** 이력 조건부 marginal mixture와 diffusion copula로 주변분포와 의존성을 분리한다. PIT와 density-ratio/score 추정 후 결합 표본을 생성한다.
- **우리에게 의미:** 금융의 joint risk를 위해 주변과 의존성을 나누는 접근도 이미 진행 중이다. 단순 copula 추가는 충분한 차별화가 아니다.
- **읽기:** §3 방법 중심. 논문 보고 성능을 독립 검증하지 않았다.
- [원문](https://arxiv.org/html/2605.19685)

### 5.5 Behavioral fraud benchmark — 2026 preprint

**Synthetic Tabular Generators Fail to Preserve Behavioral Fraud Patterns: A Benchmark on Temporal, Velocity, and Multi-Account Signals**, Sajja.

- **기여/방법:** 시간 간격·burst·계정 간 motif·velocity 규칙의 보존을 real-data noise floor와 비교한다.
- **우리에게 의미:** 행동–사기 관계가 사라지는 문제 제기도 이미 직접적인 선행연구가 있다. 평가 범위와 방법이 실제로 다른지 구분해야 한다.
- **한계:** 생성기별 학습 표본/비율 차이와 synthetic entity 구성 가정이 있다. 특정 실행 결과를 근거로 TabularARGN의 순차 기능 전체가 행 독립이라고 판단하지 않는다. proxy entity가 실제 계정임을 보장하지도 않는다.
- **읽기:** 평가 문제·한계·관련 방법 일부. 원문의 강한 불가능성 주장까지 검증한 것은 아니다. DMLR 제출 표시는 채택이 아니다.
- [원문](https://arxiv.org/html/2604.13125)

## 6. 추가 확인했지만 핵심 근거로 과장하지 않은 문헌

| 문헌 | 확인 수준과 관련성 | 1차 출처 |
|---|---|---|
| Conditional Diffusion Models for Imbalanced Tabular Regression / TabOversample, UAI 2026 | 공식 초록·서지. rare-target 가중 학습·generate/filter·DRO 연결. 9개 데이터·13개 baseline·10개 seed는 저자 보고; PDF 접근 실패로 증명 전부 미검토 | [PMLR](https://proceedings.mlr.press/v337/kang26a.html) |
| Norm-Salvaged Embedding, ICAIF 2025 | ACM 서지·검색에 노출된 방법 부분. 조건 정렬을 위한 표현 구조. 전문 접근 한계로 전체 방법 검증 제외 | [ACM DOI](https://doi.org/10.1145/3768292.3770342) |
| Conditional Generative Modeling for High-dimensional Marked TPP, KDD 2025 | 초록·도입 및 공식 서지. 고차원 mark의 조건부 implicit 생성 | [본문](https://arxiv.org/html/2305.12569), [ACM](https://doi.org/10.1145/3690624.3709258) |
| Generative AI for Banks: Benchmarks and Algorithms for Synthetic Financial Transaction Data | 2024 원고·WITS 2024 관련 서지. fidelity·utility·privacy·graph 등 다면 평가. 2026 재게시를 새로운 2026 방법으로 세지 않음 | [원고](https://arxiv.org/abs/2412.14730), [저자 SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6027795) |
| AMLNet, 2025 preprint | 초록·개요. 규칙/지식 기반 AML 생성과 탐지. 합성 AML 성공을 실제 은행 성능으로 읽지 않음 | [본문](https://arxiv.org/html/2509.11595) |
| Efficiently Generating Correlated Sample Paths from Multi-step Time Series Foundation Models, 2025 preprint | 초록·개요. 주어진 marginal에서 상관 경로 생성. 현재 제안과의 copula 관련 겹침을 추가 확인할 문헌 | [본문](https://arxiv.org/html/2510.02224) |

## 7. 문헌을 읽고 바꿔야 할 contribution 설명

| 기존에 하기 쉬운 주장 | 현재 판단 | 필요한 변화 |
|---|---|---|
| 기존 생성기는 거래를 독립적으로 본다 | 일반적으로 틀림 | TabDiT·TabularARGN 등 순차 모델을 직접 비교 |
| 이력을 더 넣어 관계를 학습한다 | U도 이력을 사용하며 E의 생성 우위 미확인 | 이력 용량과 의존성 학습의 효과를 분리 |
| 그룹별 규제로 없는 관계를 억제한다 | 필요한 관계까지 약화한 실험이 있음 | 반응 크기 자체보다 실제 관계 분포의 오차를 학습 대상으로 검토 |
| gap 구간 보정이 새로운 핵심 방법이다 | U에도 효과적인 일반 보정 | 강한 대조군으로 고정하고 그 이상의 이득을 증명 |
| 한 단계 TV가 좋으므로 생성기가 좋다 | 현재 결과와 맞지 않음 | 실제 생성 이력의 분포와 전체 궤적 관계를 함께 분석 |
| 관계 오류를 평가하면 새 benchmark다 | TabStruct·Seq2Synth·fraud benchmark와 겹침 | 희소성·조건부 관계·피드백의 상호작용에 새로운 재현 가능한 분석 필요 |
| diffusion이나 copula로 바꾸면 AI 학회급이다 | 성립하지 않음 | 기존 방법이 해결하지 못한 제약과 효율적인 알고리즘·증거 필요 |

## 8. 권고하는 한 가지 설계 방향과 기여의 조건

후보는 **이력 조건부 개별 분포와 gap–mark 결합을 분리하고, 실제 생성 이력에서 생기는 장기 관계 오차를 이용해 결합을 수정하는 모델**이다. [구체적 설계 메모](method_reconstruction_design_2026_09_20.md)에 수식, 정보 흐름, 기존 방법과 겹치는 부분, 검증·중단 조건을 적었다.

기여 후보는 “copula + rollout”이라는 조합명이 아니다. 아래 세 가지를 함께 해결하는 원리와 증거여야 한다.

1. **문제 분석:** 같은 이력에서의 예측 오류와 생성 이력의 분포 변화가 실제 관계 오류에 어떻게 기여하는지, 다른 생성 법칙과 외부 모델에서도 검증한다.
2. **방법:** 관계를 수정할 때 개별 분포까지 무제한으로 바꾸지 않으면서, 이후 이력으로 전파되는 효과를 반영하는 계산 가능한 업데이트를 제시한다. 현재 초안의 수송 제약·mirror step 자체는 알려진 수학이며 새 정리가 아니다.
3. **증거:** 단순 보정, 분리 구조만, rollout 학습만, 동일 예산의 외부 모델보다 유의미한 이득을 보이고 실제 금융 과제에서 그 의미를 확인한다.

이 설계로 좋은 결과가 나올지는 아직 모른다. 특히 국소 주변분포를 유지해도 생성 이력이 달라지면 전체 분포는 달라진다. 그 한계를 숨기지 않고 다루는 것이 연구 깊이의 일부다.

## 9. AI 학회 제출 가능성을 판단하는 기준

- **현재:** 좋은 실패 분석과 제한된 개선은 확보했지만, 새로운 생성 방법의 독립적인 우수성은 입증하지 못했다. 현재 결과만으로 상위 AI 학회 수준이라고 평가하지 않는다.
- **방법론 지향:** 새 원리/알고리즘이 명확하고 강한 가까운 baseline, 여러 법칙·데이터, 공정한 ablation에서 재현돼야 한다. 정리 개수나 파라미터 수가 학술 깊이를 대신하지 않는다.
- **AI 금융 지향:** 방법의 범위가 좁더라도 금융의 구체적 실패를 해결하고 실제 독립 테스트 효용을 보여주는 방향은 검토할 수 있다. 단순한 금융 데이터 적용만으로 충분하다는 뜻은 아니다.
- **분석 연구 지향:** 새 구조의 이득이 없더라도 여러 모델·데이터에서 일반적인 오류와 진단법을 입증하면 다른 기여가 가능하다. 현재 인공 법칙 한 종류의 관찰만으로 그 기여까지 확보한 것은 아니다.

따라서 이전 보정의 독립 재현은 기존 결과를 마무리하는 데 필요하지만, 그것만 성공한다고 방법론 기여 문제가 해결되지는 않는다. 먼저 새 설계의 작은 반증 가능한 검증을 하고, 살아남은 후보 하나에 대해서만 독립 데이터·외부 비교를 확대하는 것이 타당하다.
