# 금융·순차 tabular 생성 원문 독해 기록

기준일 2026-09-22. 현재 연구와의 연결은 CS-SAF 보고서와 종료된 ARGN 관계 파일럿을 기준으로 한다. 기존 모델·실험은 변경하지 않는다. 공개 원문의 방법·실험·결론을 읽은 범위를 페이지로 기록하며, 모든 부록 증명의 독립 검증이나 코드 재현을 뜻하지 않는다. 원문을 확보하지 못한 논문은 정독 완료로 표시하지 않는다.

## S01 FinDiff — 2023, 서베이 [72]

- 원문: https://arxiv.org/abs/2309.01472 ; 본문 PDF 1–7쪽 정독, 참고문헌 별도.
- 수치값과 범주 임베딩을 연결하여 Gaussian DDPM으로 생성하고, 범주는 최근접 임베딩으로 복원한다. 이 논문의 timestep은 diffusion 단계이며 고객 거래 시간의 이력 인코더가 아니다.
- Credit Default·Philadelphia Payments·비공개 Fund Holdings, 70/30 분할, 5 seeds. TSTR 효용은 5개 분류기의 평균 accuracy다. 행 fidelity라는 명칭은 열 쌍의 상관/분할표 비교다(5–6쪽).
- 수치 전처리 영향이 크다. Fund Holdings의 column fidelity는 표준화 .534, quantile .764(7쪽 표3). 모델 구조의 효과와 분리해야 한다.
- 비판적 확인: 5쪽 식7은 실제 최근접 거리 DCR인데 6쪽 표2·본문은 작은 DCR을 더 좋은 프라이버시로 해석한다. 거리가 작다는 정의와 보호 해석이 일치하지 않는다. 원문 주장을 프라이버시 보장으로 인용하지 않는다.
- 연구 연결: 혼합형 필드 표현의 출발점. 고객별 이력·자유 생성 경로 관계의 실증은 없다.

## S02 Imb-FinDiff — ICAIF 2024, 서베이 [74]

- 원문: https://sdm.lbl.gov/oapapers/icaif2024-schreyer.pdf ; 1쪽 초록 및 2–7쪽 본문 정독(그림의 숨은 LaTeX 메타데이터 제외).
- 실제 저자: Marco Schreyer, Timur Sattarov, Alexander Sim, Kesheng Wu. 서베이 저자 정보와 다르다.
- 수치·범주·diffusion step·클래스 임베딩을 투영하고 Hadamard product로 결합한다. 노이즈 MSE와 라벨 예측 MSE의 합을 학습한다(3–4쪽).
- 4개 데이터, 80/20 분할, 소수 클래스 합성 10만건과 실제 표본 10만건을 결합한다. Random/SMOTE/ADASYN/무증강 비교. 표3의 평균 F1 .57 대 SMOTE .47은 서로 다른 분류기·seed의 집계이며 paired 유의성 검정이 아니다(5–6쪽).
- 주의: 직접 FinDiff 대조 및 라벨 손실 단독 ablation이 없어 추가 모듈의 독립 기여가 분리되지 않는다. 극소수 30/200건 클래스 설정과 표본 추출 절차도 확인 대상이다.
- 결론은 temporal dependencies를 향후 과제로 명시한다(7쪽). 고객별 거래열 생성의 직접 해결책으로 읽지 않는다.

## S03 Fraud DDPM — ISIJ 2024, 서베이 [66]

- 원문: https://www.isij.eu/system/files/download-count/2024-11/5534_Fraud_detection.pdf ; 본문 1–12쪽 정독.
- 정상/사기 클래스별 별도 Gaussian diffusion을 학습하고 표본을 합쳐 분류기를 학습한다. 자료형을 수치로 부호화하는 설명은 있으나 세부 codec이 충분히 특정되지 않는다(3–7쪽).
- Credit Card Fraud, Online Retail, E-commerce, IEEE-CIS를 제시하고 LR/RF/XGBoost/MLP를 언급한다. 표4는 방법별 단일 결과표로, 데이터×분류기별 수치·반복 분산·평가 선택 절차를 분해해 주지 않는다(8–10쪽).
- 따라서 보고된 F1 .89를 모든 데이터에서의 검증된 효과로 확대하지 않는다. 온라인 거래열을 생성하는 모델이나 시간 밖 검증의 증거도 제시되지 않는다.
- 연구 연결: 사기 증강 목적의 참고. 엄밀한 관계 생성 benchmark의 강한 근거로 삼기에는 실험 보고가 제한적이다.

## S04 EmDT — 2026 v2, 서베이 [44]

- 원문: https://arxiv.org/abs/2603.13566v2 ; 본문 1–14쪽 정독.
- 사기 표본을 UMAP→KMeans 3개 군집으로 나누고 군집별 DDPM을 학습한다. feature 위치 sinusoidal embedding과 transformer를 사용하며 군집 비율대로 생성한다(5–8쪽).
- European Credit Card 단일 데이터, 수치 29개(PCA 28+Amount). Time 제외(10쪽). 60/20/20 분할, 10 runs, validation F1로 Optuna, XGBoost 평가, 사기 표본을 두 배로 증강한다.
- F1: 무증강 .800, SMOTE .834, EmDT .849; 군집 없는 EmDT .829(11·13쪽). 분산을 제시하지만 이 값만으로 paired 유의성을 판정하지 않는다.
- TabDDPM은 기본 hyperparameter, EmDT는 탐색이므로 탐색 예산의 차이를 점검해야 한다. DCR은 train/test 최근접 비교 확률이며 FinDiff의 원시 DCR과 동일 지표가 아니다.
- 연구 연결: 희귀 패턴 내부의 이질성·표현 스케일에 대한 참고. 거래 시간·고객 이력·실제 혼합형 범주 데이터의 일반화는 여기서 입증되지 않는다.

## S05 DP-FedTabDiff — 2024/2025 v2, 서베이 [73]

- 원문: https://arxiv.org/abs/2412.16083v2 ; 본문 1–7쪽 정독.
- FinDiff 기반 로컬 DP-SGD와 federated averaging을 결합한다. 4개 데이터, 비IID 범주 분할, client 수·로컬 update 수·aggregation·privacy budget을 비교한다. 고객 이력 인코더를 제안하는 연구는 아니다.
- TSTR 분류 accuracy, 열/열 쌍 fidelity, Anonymeter 공격을 함께 평가한다. ε=1에서 비DP 대비 공격 위험 약 34% 감소와 utility 약 15%, fidelity 약 14% 감소를 보고한다. 초록의 성능 보존 표현보다 본문의 상충 관계가 중요하다.
- 로컬 update 증가가 항상 유리하지 않으며 분할 이질성과 통신 빈도의 상호작용이 있다. 가장 좋은 설정을 일반 법칙으로 확대하지 않는다.
- DP 인접성은 레코드 변경을 기준으로 읽어야 한다. 한 고객의 여러 거래 전체를 보호하는 고객 단위 DP는 별도의 주장이다. 전처리·전체 반복 회계의 독립 감사는 하지 않았다.

## S06 DP-FinDiff — 2025, 서베이 [75]

- 원문: https://arxiv.org/abs/2512.00638 ; 본문 1–6쪽 및 부록 9–15쪽 정독.
- 혼합 필드 임베딩 DDPM에 per-example clipping과 DP-SGD를 사용한다. FA는 feature 평균 손실을 합으로 바꾸고, AT는 diffusion step 샘플링을 epoch별 power-law로 바꾼다. AT는 관측 손실에 따라 자동 적응하는 방식과 구분된다.
- 5개 데이터·5 seeds·ε=.2/1/10. 주 baseline 대비 큰 향상과 추가 모듈 효과를 구분해야 한다. 본문 평균 utility 개선은 FA .13%, AT 1.22%, 결합 2.20%로 보고한다(5쪽). 모든 데이터·ε에서 최고인 것은 아니다.
- 부록 B2는 importance correction을 사용하지 않는다고 명시한다. 따라서 AT는 균등 timestep 목적의 불편 추정만을 개선하는 것이 아니라 학습 가중치를 바꾼다.
- 독립 수식 점검: 동일 파라미터·동일 feature 수 d라면 g_sum=d·g_mean이다. gradient norm의 상대분산 Var/Mean²와 왜도는 상수배에 불변이다. FA가 이것을 축소한다는 4쪽/B1의 설명은 그대로 성립하지 않는다. 고정 clipping threshold에서 유효 clipping과 최적화 경로가 달라져 실험 효과가 생길 가능성과는 별개다.
- 연구 연결: 학습 목적의 가중치·스케일·privacy accounting 참고. 순차 거래 관계 보존의 증거는 아니다.

## S07 Privacy Risks and Tradeoffs — WWW Companion 2026, 서베이 [103]

- 원문: https://arxiv.org/abs/2602.09288 ; 본문 1–8쪽, 부록 10–11쪽 정독.
- 6개 금융 관련 정적 표 데이터에서 Gaussian Copula, TabDiff, CTGAN, TVAE 및 DP-CTGAN/TVAE를 비교한다. DP diffusion을 직접 비교한 실험은 아니다.
- DP-CTGAN은 학습 외 전처리·조건부 표본 추출·gradient penalty까지 수정한다. 공개 min/max 범위 가정이 있다. DP는 training loop만의 속성이 아니라 pipeline 전체의 속성임을 보여준다.
- 높은 주변분포 fidelity가 높은 TSTR utility를 보장하지 않는다. TabDiff도 모든 데이터의 downstream utility에서 최고는 아니다. 데이터 균형화는 목표 분포 자체를 바꾸므로 원래 분포 재현과 구분한다.
- DCR은 uniform 기준으로 정규화한 거리와 train/test overfitting 지표다. FinDiff의 raw DCR과 숫자를 직접 비교하지 않는다. canary 기반 shadow 공격이 .5 근처라는 결과는 해당 공격의 실패이며 공격 불가능성의 증명이 아니다.
- 대부분 3개 학습 seeds, 균형화 실험은 1 seed. 소수 클래스 비율은 약 6.68–30%여서 극단적 사기 희소성을 직접 대표하지 않는다.
- 연구 연결: 주변분포·관계·효용·프라이버시를 분리해 평가해야 한다는 근거. 고객별 rollout을 검증한 논문은 아니다.

## S08 CTDF — ICAIF 2025 / arXiv 공개 2026, 서베이 [9]

- 원문: https://arxiv.org/abs/2606.28674 ; 본문 1–9쪽 정독, 9쪽 그림5 원본 이미지 교차 확인.
- 사전학습 TabDiff의 각 sampling step에 feasibility 연산을 삽입한다. 수치는 Euclidean projection, 범주는 금지 범주의 확률을 0으로 하고 재정규화한다. 결합 규칙은 입력 열을 고정하고 목표 열을 순차 보정한다(4–5쪽).
- Airbnb·Lending Club 사례에서 규칙 위반 0을 보고한다. 이는 지정 규칙 준수의 결과이며 정확한 조건부 분포 p(x|C), 일반 법적 적합성, 인과적으로 타당한 외삽의 증명은 아니다.
- 9쪽 그림5에서 CTDF의 PR-AUC .2570은 Full→SB .1704보다 높지만 F1 .0305는 .0734보다 낮다. 본문의 ‘모든 지표에서 최고’ 주장과 표가 불일치한다. 50만건 증강 대 약 1만건 비교의 규모 차이도 제약 연산 효과와 분리해야 한다.
- 연구 연결: 거래 조합의 hard support 보정에 참고. hard rule 위반 감소와 확률적 관계 재현, 희귀 패턴의 현실성은 별도 평가해야 한다.

## S09 TDCE — Frontiers 2026, 서베이 [97]

- 원문: https://doi.org/10.3389/frai.2026.1743495 ; 본문 1–13쪽 정독. 별도 supplementary proof는 독립 검증하지 않음.
- 미분 가능한 분류기의 예측을 뒤집는 counterfactual explanation을 생성한다. 연속 필드는 classifier/distance gradient로 reverse diffusion을 유도하고, 범주는 Gumbel-softmax와 log-density 근사를 통해 guidance를 전달한다. 불변 필드는 단계별 noisy blending 후 원값 복구.
- LCD/GMC/Adult/LAW, 각 balanced test 1,000건, 5 seeds. validity·L2·다양성·안정성·범주 JS·autoencoder 기반 IM1/IM2를 평가한다. 온도가 작으면 범주 gradient가 약해지고 크면 근사 오차가 커지는 trade-off가 있다.
- 예측 label flip과 target-class 유사성은 실제 개입의 인과 효과가 아니다. IM1/IM2도 학습한 autoencoder에 의존하는 plausibility 대리 지표다. 최단 거리·최고 다양성·최고 validity가 동시에 최적인 방법은 없다.
- 연구 연결: 필드를 고정하거나 반응을 유도하는 sampling 방법론 참고. 고객별 장기 거래열 생성의 직접 경쟁 모델은 아니다.

## S10 Latent Forest Diffusion — 2025, 서베이 [34]

- 원문: https://arxiv.org/abs/2511.16571 ; 본문 1–30쪽 정독, 참고문헌 31–35쪽 별도.
- PCA/MLP autoencoder/Transformer autoencoder로 압축한 공간에서 GBT 기반 flow를 학습한다. PCAForest는 분류까지 latent에서 수행하고, EmbedForest·AttentionForest는 복원한다. attention 축은 행 안의 feature다.
- 11개 데이터, RF/XGBoost, minority 증강률 25–300%, SMOTE/CTGAN/ForestDiffusion 비교. AttentionForest는 평균 utility가 좋고 계산 비용은 더 높다. PCA의 WD/DCR은 latent 공간에서 측정하여 원공간 값과 직접 비교할 수 없다(21·28쪽).
- 핵심 재현성 문제: 20쪽은 train-only 증강·고정 real test라고 쓰지만, 12–13쪽 Algorithm1은 전체 X에 표준화/PCA, 전체 XPCA에서 minority 추출, 전체 X에 증강 후 재분할로 적혀 있다. 서술을 그대로 구현하면 test 정보가 학습에 들어갈 수 있다. 코드 실행 없이 실제 누수를 확정하지 않는다.
- 표9는 증강률별 최댓값을 모은 표다. validation으로 선택한 단일 설정의 test 결과인지 분명하지 않아 성능 근거를 보수적으로 취급한다.
- 연구 연결: codec/표현 효과를 생성기 효과와 분리하는 비교 설계가 필요하다는 사례.

## S11 EntTabDiff — ICAIF 2024, 서베이 [53]

- 출판사: https://doi.org/10.1145/3677052.3698625 ; 저자 Changshuo Liu, Canyao Liu.
- **원문 미확보. 정독 완료 아님.** ACM 페이지와 PDF가 HTTP 403을 반환했고 공개 저자 원고를 찾지 못했다. 검색·서베이의 설명을 원문 독해로 대체하지 않는다.
- entity conditioning이 현재 연구와 직접 관련될 가능성이 높으므로 전체 리뷰의 남은 중요한 공백이다. 원문 확보 전에는 이력 사용 방식·시간 순서 모델링·평가 범위에 관한 확정 비교를 보류한다.

## R01 TabDiT — ICLR 2025

- 원문: https://arxiv.org/abs/2504.07566v2 ; 본문 1–10쪽 및 부록 14–23쪽 정독.
- 각 거래 행을 VAE로 압축한 뒤 전체 latent 행렬을 DiT가 함께 생성한다. AR은 **행 내부 필드 decoder**에 쓰이며 시간축 전체를 한 거래씩 AR 생성하는 구조와 다르다. 고객 속성은 별도 encoder와 adaLN/CFG로 조건화한다.
- 수치는 magnitude token+유효숫자 4개로 표현하며 EOS padding을 학습해 길이와 내용을 함께 생성한다. 따라서 혼합형·고객조건·가변길이 거래열 생성은 이미 직접 선행되어 있다.
- 6개 공개 데이터와 비공개 대규모 은행 데이터. 공개 주실험 3개 분할 반복. MLD-TS는 tsfresh 계열 특징+CatBoost로 real/synthetic 거래열을 구별한다. 분류 실패는 모든 조건부 관계가 맞다는 보장이 아니다. Berka 무조건 MLD-TS는 여전히 85.53%다(10쪽).
- 9쪽 ablation은 codec·행 내부 AR·길이 예측의 영향을 분리한다. 양자화 기준은 equal-width이고 모든 adaptive binning이 열등하다는 증거는 아니다. 140M DiT·200 sampling steps로 작은 GRU와 비용이 다르다(21쪽).
- 연구 연결: 가장 직접적인 diffusion baseline. 희귀한 gap×이전상태×다음행동 관계와 rollout 위치별 오류는 별도로 대조해야 한다.

## R02 TabularARGN — 2025 v2

- 원문: https://arxiv.org/abs/2501.12012v2 ; 본문 1–8쪽, 부록 13–18·20–23쪽 정독.
- 모든 필드를 범주 sub-column으로 부호화하고 any-order AR cross-entropy로 학습한다. 순차형은 LSTM 과거 이력, 현재 행의 앞 필드, 정적 parent context를 결합한다. 생성 길이를 먼저 예측하고 위치 index를 증가시킨다(4–6·14쪽).
- numeric-discrete/binned/digit codec을 선택하며 binned 복원은 bin 안의 uniform sampling이다(13쪽). 현재 D_bin의 경험적 bin 내부 복원과 동일하지 않지만 이산화 아이디어 자체는 선행되어 있다.
- Berka·Baseball·California 및 정적 데이터, default 설정, validation early stop. Berka 전체 accuracy .79, coherence .82를 보고한다(23쪽). 이는 downstream accuracy가 아니라 주변·쌍·연속 두 시점 분포 점수의 평균이다.
- 평가에서 수치는 decile, 범주는 상위 10개로 제한한다. coherence는 고객별 임의의 연속 두 사건에서 같은 필드의 관계를 본다(15–16쪽). 희귀 merchant 조합·다중 필드 조건부 관계·긴 rollout 오류를 모두 검증하지 않는다.
- 연구 연결: 필수 직접 baseline. 현재 작은 예산 파일럿의 실패로 충분히 학습한 ARGN 전체가 실패한다고 결론 내릴 수 없다.

## R03 Seq2Synth — CIKM 2026 채택, 최신 v3 2026-08-31

- 원문: https://arxiv.org/abs/2607.15606v3 ; 본문 1–7쪽, 부록 9–12·15–17·19–20쪽 정독. v2와 최신 v3를 구분하여 최신본으로 분석함.
- 시간 표현·규칙성·고객 간 의존·schema로 적용 지표를 선택한다. timestamp, 시점별 단면, 고객 내 dynamics, 관계형 cardinality를 분리하고 temporal utility/privacy를 더한다. 주변분포 평가만으로 temporal quality를 판단할 수 없다는 문제의식은 직접 선행된다.
- 13개 자료 중 7개 core, 8개 주비교 모델. ARGN/CPAR는 추가 single-table 실험에 있으며 core Berka의 ARGN 비교는 없다. 단순히 모든 최신 순차 모델이 동일 조건에서 실패했다고 읽지 않는다.
- raw timestamp 지표와 보정 후 다른 지표를 분리한다. 보간은 fidelity를 높일 수 있어 sparse/보정량도 보도록 한다. TabDiT는 배포된 생성 자료를 사용한다(12·15쪽). generator 재학습 조건이 완전히 통일된 비교는 아니다.
- 주의: TabDiT를 시간축 AR처럼 묶는 설명은 원 논문 구조와 구분해야 한다. n-gram 일치는 정보 노출 탐색 지표이며 회원 추론 성공이나 DP 위반의 증명이 아니다.
- 연구 연결: 기존 지표와 중복을 명확히 해야 한다. Berka의 일 단위 timestamp에는 정상적인 동시 날짜 거래가 가능하므로 uniqueness=1을 무조건 목표로 삼지 않는다.

## R04 TabStruct — ICLR 2026

- 원문: https://arxiv.org/abs/2509.11950v2 ; 본문 1–11쪽, 부록 25–27·32–33·44–48쪽 정독. 63쪽 전체 부록을 정독한 것은 아님.
- global utility는 각 필드를 나머지 필드로 예측하는 TSTR 성능을 real-trained 기준으로 정규화하고 평균한다. 단일 target 효용보다 다양한 조건부 관계를 보는 목적이다. SCM 자료에서는 알려진 CI 집합의 보존율도 측정한다.
- 13개 생성기, 6개 SCM+23개 실제 데이터, 10회 분할, generator objective로 tuning. global utility와 global CI의 Spearman .84를 보고한다. 저자도 이 연결을 경험적 결과이며 증명이 아니라고 명시한다(8쪽).
- 관측 예측 성능은 causal direction/개입 효과를 식별하지 않는다. 평균 성능은 희귀 조건에서의 관계 오류를 가릴 수 있고, 연속값 RMSE는 전체 조건부분포 평가도 아니다. CI test의 가정·검정력에도 의존한다.
- 연구 연결: ‘관계가 중요하다’는 포괄적 주장은 이미 선행된다. 48쪽은 temporal/event 데이터 확장을 향후 과제로 명시한다. 시간 순서·희귀 상태·생성된 이력 분포를 함께 고려하는 구체적 확장이 후보지만 자동으로 새 기여가 되는 것은 아니다.

## R05 ADiff4TPP — 2025 원고, TMLR 2026

- 읽은 원문: https://arxiv.org/abs/2504.20411v1 ; 본문 1–9쪽, 부록 18–22쪽. TMLR 게재는 [저자 소속기관](https://www.grad.ubc.ca/node/115396)에서 확인했으나 [최종본](https://openreview.net/forum?id=bwnZW4wXh4)은 접근 확인 화면으로 막힘. 아래 수치는 2025 원고 기준이며 최종본 대조는 미완료.
- 사건의 gap·mark를 β-VAE로 임베딩하고, 앞선 사건이 먼저 복원되도록 사건별 비동기 noise schedule을 갖는 DiT/flow matching을 학습한다. 관측 prefix를 고정하고 여러 미래 사건을 함께 생성한다. 관측·예측 길이는 설정된 최대 길이 N 안에서 가변이다.
- 5개 EasyTPP 데이터·5 seeds, 다음 gap RMSE/mark error와 horizon 5/10/20/30의 OTD를 평가한다. OTD 평균 개선 24.5%는 해당 baseline·cost 설정에 한정된다. time prediction 우위와 mark prediction 우위를 구분한다.
- 동일 계열의 synchronous/disjoint schedule 및 latent/mask ablation이 있어 생성 순서의 영향을 비교하는 데 유용하다. 전 금융 거래 schema, 금액·merchant의 희귀 결합이나 고객 상태별 관계 보존은 평가하지 않는다.
- 연구 연결: history conditioning, 시간·종류 결합, 장기 오류 감소를 새 주장으로 내세우기 전 비교해야 한다. OTD cost가 다른 LBDTPP의 절대 점수와 직접 비교하지 않는다.

## R06 LBDTPP — 2026-06-23 preprint

- 원문: https://arxiv.org/abs/2606.24982v1 ; 본문 1–15쪽과 부록 증명 17–18쪽 정독. PDF 상단의 2020년 표시는 미수정 LaTeX template이며 출판 연도가 아니다.
- 고정 sinusoidal gap embedding+고정 mark embedding을 더한다. 블록 간 AR, 블록 내부 Gaussian diffusion, KV cache를 사용한다. decoder는 positive gap과 argmax mark를 출력하며 종료 시각 T를 넘으면 자른다.
- 6개 비금융 중심 event 데이터, 10 seeds, validation tuning. 무조건 생성은 block 8, 조건부 미래 20건은 block 4. 일부 baseline 결과는 CDiff 논문에서 가져온다. block 1/2/4/8/16/20 ablation은 중간 block의 이점을 보여준다. ADiff4TPP/TabDiT와의 직접 비교는 없다.
- 정리는 모든 prefix에서 local approximation error와 prefix Lipschitz stability를 가정한다. 블록 길이가 b일 때 ε_block≤b·ε_event 및 ρ_block≤ρ_event라면 **오차 상한**이 작아진다. 실제 오차의 우열이나 학습 후 이 가정 성립을 보장하지 않는다.
- 독립 해석: latent→event bound의 Lipschitz decoder 조건은 argmax mark에 자동 성립하지 않는다. decoded real latent와 원본 사이의 reconstruction error도 별도다. 무조건 표본과 임의 실표본의 평균 OTD만으로 전체 분포 일치를 판정할 수도 없다.
- 연구 연결: ‘rollout 누적 오차’가 미개척이라는 주장은 불가. 추가 기여는 혼합 거래 필드·희귀 조건부 관계·실제 고객 데이터에서 무엇을 더 설명하는지에 달렸다.

## R07 TabCascade — ICML 2026, v3 2026-05-13

- 원문: https://arxiv.org/abs/2601.22816v3 ; 본문 1–9쪽, 부록 13–18·23–24·38–39쪽 정독.
- 범주와 수치의 coarse state z를 CDTD로 함께 생성하고, z·범주를 조건으로 연속값을 flow matching으로 복원한다. DT/GMM이 coarse state를 만든다. 점질량·결측은 명시적 상태이며 연속 생성 손실에서 제외한다. high-resolution 모델은 실제 coarse state로 teacher forcing한다.
- 12개 정적 데이터, 70/10/20, 3 training×10 sampling seeds. 표의 ±는 12개 데이터 간 표준편차다. 약 3M parameter·시간 예산을 맞추며 TabDiff transformer를 제거하는 등 원본 baseline 구조를 수정했다. 원저자 최적 설정과의 무제약 비교가 아니다.
- 평균 detection score .518→.787, MLE 상대오류 .039→.027. ‘효용 30% 향상’은 accuracy 30%p 증가가 아니다. cascade→z→FM→coupling의 단계별 ablation 중 z만 추가하면 오히려 악화한다.
- 독립 수식 점검: 부록의 ‘부분 구간이면 각 구간 분산≤전체 분산’은 일반적으로 틀리다. 일정 가중치에서는 전체분산 법칙으로 평균 조건부분산을 제한할 수 있으나, 상태별 학습 schedule의 가중치까지 포함한 일반 증명은 별도 확인이 필요하다. 실험 성능과 정리의 일반성을 분리한다.
- 연구 연결: D_bin을 해석할 때 coarse 관계와 bin 내부 조건부 복원을 구분하는 참고. 해당 논문은 고객 거래열 생성의 실증이 없다.

## R08 Synthetic Tabular Generators Fail to Preserve Behavioral Fraud Patterns — 2026-04-13 preprint

- 원문: https://arxiv.org/abs/2604.13125v1 ; 본문 1–24쪽, 부록 27–28쪽 정독.
- P1 gap/자기상관, P2 burst/활동기간, P3 공유 device/IP fan-out, P4 velocity rule을 real-real noise floor 대비 평가한다. 4개 생성기·IEEE-CIS/Amazon FDB를 사용하며 diffusion은 실험하지 않는다. P3 실제 비교는 fan-out 하나이고 모든 graph motif를 실증한 것은 아니다.
- 결정적 설정: IEEE-CIS에서 card1을 학습에서 제외하고 생성 행을 무작위 pseudo-entity에 배정한다(11쪽). ARGN의 순차 기능을 평가한 결과가 아니다. 원문 ARGN 한도는 **240분**(10쪽), 부록은 10 epochs(27쪽)다.
- 저자의 ‘올바른 entity 크기 분포이므로 degradation의 하한’ 주장은 성립이 보장되지 않는다. 무작위 배정은 실제로 학습 가능한 entity–event 관계를 파괴할 수 있다. card1은 고객 진실값으로 검증되지 않은 proxy다.
- 독립 이론 점검: 고정 attribute의 Poisson-binomial 분산 제한으로 서로 다른 attribute의 혼합 fan-out가 heavy-tail이 될 수 없다고 결론 내리는 것은 부당하다. 혼합분산에는 Var(E[F|attribute]) 항이 추가된다. 또한 임의 시간분포의 spacing 자기상관 일반 주장은 제시된 uniform 예만으로 증명되지 않는다.
- 연구 연결: 행동 평가 항목은 유용한 선행 아이디어. 포괄적 불가능성 주장·ARGN 실패 수치·분할에 민감한 비율을 현재 연구의 정당화로 그대로 사용하지 않는다.

## R09 EdiTPP — ICLR 2026, v3 2026-02-04

- 원문: https://arxiv.org/abs/2510.06050v3 ; 본문 1–10쪽, 부록 14–16·20–21쪽 정독.
- 연속 시간상의 사건 집합을 insert/delete/substitute하는 CTMC다. edit 위치를 64 bins와 uniform dequantization으로 표현하고, alignment 공간의 목표 edit rate를 Bregman loss로 학습한다. 무조건 학습한 모델에 관측 구간을 재조건화해 예측도 수행한다.
- 7 real+6 simulated unmarked TPP 데이터, 5 seeds, 20k updates, validation W1-IET 선택. unconditional은 4k 생성열의 MMD·count/IET distribution, conditional은 random time window의 event count·시간 거리를 본다. 빈 관측/빈 미래도 허용한다.
- 100 sampling steps를 맞춘 비교에서 edit 수·실행시간을 줄이며 전반적으로 경쟁력 있다. 모든 데이터·지표 최고는 아니다. 길이가 늘어나는 AR와 달리 사건 수 자체를 편집한다.
- 연구 연결: gap만 맞추면 충분한지, 고정 사건 수 조건과 고정 시간 구간 조건이 어떻게 다른지 검토하는 참고. **mark·merchant·amount가 없는 모델**이므로 현재 task의 완성된 직접 대체재는 아니다. bin 내부 uniform 복원 자체도 새 아이디어가 아니다.

## R10 PersonaLedger — 2026-01, v2 2026-01-20

- 원문: https://arxiv.org/abs/2601.03149v2 ; 본문 1–10쪽, 부록 15–17쪽 정독.
- persona와 ledger state·최근 7일 대화를 조건으로 LLM이 일별 거래를 제안하고, program이 잔액·결제·구독 등 규칙을 검사하고 상태를 갱신한다. 약 30M 거래·23k 사용자, merchant/type/time/amount 등을 생성한다.
- 부도성 유동성 부족 예측과 다른 합성 사용자의 하루 거래를 삽입한 identity-theft segmentation benchmark를 제공한다. 주실험은 생성된 데이터 안에서 예측 모델을 비교한다. 실제 은행 데이터 TSTR 전이 또는 learned generator와의 동일 조건 fidelity 비교가 아니다.
- 달력·persona별 그럴듯한 평균과 분산은 실제 고객의 조건부 행동분포 일치 증거와 다르다. 반복 실험 불확실성, 생성 비용, 규칙별 제거 실험도 본문만으로 충분하지 않다. 현실 데이터 없이 생성했다는 점을 별도의 DP 보장으로 표현하지 않는다.
- 연구 연결: 규칙 위반 없는 ledger와 통계적으로 충실한 거래열은 구분해야 한다. 재현 가능한 오류 주입·상태 전이 실험에 유용한 simulator 후보지만 실데이터 검증을 대체하지 않는다.

## R11 FINESSE — 2026-09-09 preprint

- 원문: https://arxiv.org/abs/2609.11993v1 ; 본문 1–8쪽 정독. 이번 검색에서 확인한 가장 최근의 직접 관련 원문이며, 전체 분야의 마지막 논문이라는 의미는 아니다.
- customer/merchant/bank agent와 잠재 merchant affinity·payment strategy 상태를 통해 거래·결제·계좌 상태·정책 변경 4개 stream을 연결한다. 위험 merchant와의 상호작용이 이후 fraud risk에 영향을 준다.
- fraud temporal graph, missed payment, balance forecast, next-event task를 제공한다. fraud는 시간순 70/15/15 분할, TGN/DyRep baseline. 실제 고객분포를 학습하여 재현한 생성 모델 비교는 아니다. 본문은 formal causal inference를 목적으로 하지 않는다고 명시한다(2쪽).
- baseline 예측 성능은 simulator 내부의 학습 가능성을 보여준다. 현실 전이·금융적 calibration·생성기 간 우열은 별도로 검증해야 한다. simulation hidden state를 정답으로 쓰는 통제 실험 설계에는 장점이 있다.
- 연구 연결: 관계 보존의 단위가 한 거래의 필드와 한 고객의 과거를 넘어 여러 event stream까지 확대되고 있다. 현재 연구에서 문제 단위를 명확히 정의해야 할 이유다.
