# 데이터셋 중심 문헌 검토와 연구 진행 재점검 — 2026-09-20

**현재 우선순위는 추가 구조·보정 개발보다 외부 거래 데이터에서 해결할 문제를 확인하는 것이다.** 자체 시뮬레이션은 정당한 연구 도구이지만, 같은 생성 법칙의 오류를 계속 수정하는 것만으로 금융 데이터 생성 방법의 기여가 확보되지는 않는다. 앞선 ‘반복 상태 × 간격 보정’ 제안은 아직 실행하지 않았으며, 외부 문제 확인보다 먼저 진행할 필수 단계로 두지 않는다.

이번 작업은 문헌의 데이터·평가 절 확인과 저장된 실험/전처리 기록의 감사다. 새 모델 학습, 보정 학습, 생성, 외부 데이터 성능 평가는 실행하지 않았다. 기존 실패 판정과 결과는 바꾸지 않는다.

## 1. 무엇을 확인했는가

아래 **14편**의 원문 또는 저자 공개 원고에서 데이터셋과 관련 실험 절을 확인했다. 최근 2025–2026년 연구와 가까운 순차·금융 생성 연구를 우선하고, 시뮬레이션의 용도를 확인하기 위해 TimeGAN·CTGAN·Quant GANs를 포함했다. 전체 분야의 전수조사나 모든 증명·코드의 재현을 뜻하지 않는다. 학술지, 본 학회, 워크숍, preprint를 구분했다. 출판사 원문이 제한되면 저자 원고를 읽고 출판 정보는 출판사/저자 저장소로 확인했다.

‘실제 데이터’는 **모델 학습·평가에 쓰는 원본의 출처**를 뜻한다. 그 데이터로 학습한 모델이 출력하는 합성 데이터와 혼동하지 않는다. Kaggle은 유통 플랫폼이며, 등록된 데이터가 모두 시뮬레이션인 것은 아니다.

### 거래열·관계형 표와 직접 가까운 연구

| 논문·출판 상태 | 원문에서 확인한 학습/평가 데이터 | 데이터의 성격과 우리 연구에 주는 의미 | 읽은 위치 |
|---|---|---|---|
| [TabDiT, ICLR 2025](https://arxiv.org/html/2504.07566) | Age1, Age2, Leaving, PKDD’99 Financial/Berka, Rossmann, Airbnb. 추가로 비공개 국제은행 거래 | 공개 관측 은행 거래·매출·이용 이력. 행별 분포와 거래열 판별/예측 효용을 함께 평가하는 직접 비교 문헌 | §5.1, Appendix E 및 D.2 |
| [TabularARGN, 2025 공개 원고](https://arxiv.org/html/2501.12012) | 순차/관계형: Berka, Baseball, California. 단일 표: Adult, ACS-Income, Default, Shoppers | Berka는 계좌–거래. California 가구–개인은 관계형 자료로, 모든 데이터가 시간순 금융 거래열인 것은 아님. 최종 채택 상태를 추정하지 않음 | §4 및 Appendix F |
| [REaLTabFormer, 2023 공개 원고](https://arxiv.org/html/2302.02041) | 관계형 비교에 Rossmann 매장–매출, Airbnb 사용자–세션 | 실제 관측 데이터. parent와 child의 관계 및 계절성을 비교. 두 데이터 모두 금융 사기 데이터는 아님 | 관계형 실험, Appendix B.2 |
| [Seq2Synth, 2026-08 v2 preprint](https://arxiv.org/html/2607.15606v2) | 13개: Walmart, PTB-XL, FreddieMAC, FannieMAE, H&M, Coupon, Google, Home Credit, Rossmann, Airbnb, Berka, CMAPSS, Citi Bike | 여러 관측 자료와 **시뮬레이션 CMAPSS**를 포함. 본문 core 7개와 전체 13개를 구분. 정적 분포 순위가 시간 관계의 순위를 대신하지 못한다는 평가 연구 | §3–4, Appendix D.3–D.6 |
| [FinDiff, ICAIF 2023](https://arxiv.org/pdf/2309.01472) | Credit Default(대만 신용카드 고객), Philadelphia Payments(2017 시 지급 내역), 비공개 Fund Holdings | 관측 금융 표. 혼합 수치·범주형 생성의 비교 대상이지만, 그 자체로 고객별 자유 거래열 생성 검증은 아님. [저자 출판 정보](https://github.com/sattarov/FinDiff) | §4.1–4.3 |
| [Generative AI for Banks, WITS 2024](https://arxiv.org/pdf/2412.14730) | 비공개 실제 거래 약520만 행 및 IBM이 시뮬레이션한 공개 거래 약421만 행 | 실제/시뮬레이션 금융 데이터를 함께 비교. 본문의 ‘IBM simulated’를 현재 사용한 AMLSim 파일과 동일하다고 단정하지 않음. 훈련 자료와 비교한 평가도 있어 독립 검증 절차를 그대로 복제하면 안 됨 | Algorithm Evaluation, pp.3–4 및 결과 |

Seq2Synth의 논문 표에 있는 Berka 900개 계좌, H&M 10,000명과 프로젝트의 같은 규모 표본은 **동일 표본이 아니다**. 계좌/고객 ID, 분할, 길이 제한을 맞추지 않고 논문 수치를 직접 비교해서는 안 된다. TabDiT의 방법은 거래행 잠재표현의 시퀀스 diffusion과 행 내부 decoder를 구분해야 하며, 다른 논문의 짧은 요약만으로 시간축 자기회귀 모델이라고 규정하지 않는다.

### 금융 시계열 저널·학회: 가까운 평가 원칙, 다른 데이터 단위

| 논문·출판 상태 | 원문에서 확인한 데이터 | 시뮬레이션/실증 역할과 관련성 | 읽은 위치 |
|---|---|---|---|
| [Time-Causal VAE, SIAM Journal on Financial Mathematics 2026](https://epubs.siam.org/doi/abs/10.1137/24M1711650) | **직접 생성한 Black–Scholes, Heston, 경로 의존 변동성 모델** + 실제 S&P500/VIX 일별 자료(2014-08–2024-08) | 통제 모형에서 조건부·경로 특성을 검증하고 시장 자료로 확장. 자체 생성 데이터 사용이 논문으로 부적절하다는 주장의 반례. 거래 사기 모델과 동일 과제는 아님 | [저자 원고](https://arxiv.org/html/2411.02947) §4.1–4.2 |
| [Generation of synthetic financial time series by diffusion models, Quantitative Finance 2025](https://www.tandfonline.com/doi/full/10.1080/14697688.2025.2528697) | Refinitiv의 AAPL/NASDAQ 분 단위 가격·스프레드·거래량, 2005–2014 | 실제 관측 시장 자료. 주변분포뿐 아니라 변동성 군집, 일중 패턴, 변수 간 관계를 평가. 고객 거래열의 직접 baseline으로 취급하지 않음 | [저자 원고](https://arxiv.org/html/2410.18897) §4.1–4.2 |
| [GANs and synthetic financial data: calculating VaR, Applied Economics 2025](https://www.tandfonline.com/doi/full/10.1080/00036846.2024.2365456) | 실제 S&P500, FTSE100 일별 지수 | 가격 수준이 비슷해도 수익률의 시차 특성이 달라질 수 있음을 분석. 겉보기 분포 일치와 목적에 맞는 관계 재현을 구분하는 사례 | §III Data characteristics/Results, §IV |
| [Quant GANs, Quantitative Finance 2020](https://www.tandfonline.com/doi/abs/10.1080/14697688.2020.1730426) | S&P500 실제 지수(2009-05–2018-12) | TCN 기반 생성, 분포·시차·변동성 관계를 GARCH 등과 비교. 기초 문헌이며 최신 거래열 모델은 아님 | [저자 원고](https://arxiv.org/pdf/1907.06673) §7 |
| [CoFinDiff, IJCAI 2025 AI4Tech](https://www.ijcai.org/proceedings/2025/1040) | JPX FLEX 11개 종목의 1분 자료, 2015–2021 | 실제 시장 관측에 조건부 diffusion을 학습. 추세·실현변동성 제어와 hedging 효용을 연결. gap–merchant 반복과는 다른 과제 | [원문](https://arxiv.org/html/2503.04164) §4.1–4.2 |

이 사례들의 평가를 우리 주장에 대응시키면 ‘거래 분포가 닮음 → 시간·행동 관계가 보존됨 → 그 관계가 필요한 과제에 도움이 됨’을 각각 보여야 한다. 가격 시계열 논문의 성공이 거래 생성에서 같은 구조가 효과적이라는 증거는 아니다.

### 인공 데이터 사용과 사기 탐지의 근거

| 논문·출판 상태 | 원문 데이터 | 해석 | 읽은 위치 |
|---|---|---|---|
| [TimeGAN, NeurIPS 2019](https://papers.nips.cc/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf) | 직접 만든 다변량 AR Gaussian, Sines + 실제 Google 주가, UCI Energy, 비공개 폐암 진료 사건열 | 시간/변수 상관을 바꾼 통제 실험과 여러 관측 자료 검증을 함께 수행 | §5.1–5.3 |
| [CTGAN/TVAE, NeurIPS 2019](https://papers.nips.cc/paper/8953-modeling-tabular-data-using-conditional-gan.pdf) | 직접 구성한 Gaussian mixture/Bayesian-network 데이터 **7개** + 실제 자료 **8개**(Adult, Census, Covertype, Intrusion, News, Credit, MNIST28/12) | 정답 분포를 아는 실험과 합성 자료로 학습한 예측기의 실자료 성능을 구분. 순차 생성 모델은 아님 | §5.1–5.2 |
| [Enhancing credit card fraud detection: highly imbalanced data case, Journal of Big Data 2024](https://link.springer.com/article/10.1186/s40537-024-01059-5) | 공개 시뮬레이션 카드 거래, **Sparkov**, 실제 European Credit Card Fraud | Sparkov를 연구에 사용하는 명확한 사례. 연구 대상은 특징 선택·사기 분류이며 거래열 생성기 우수성의 직접 근거는 아님 | Experimental results → Data used for experiments |

**문헌에서 도출되는 판단:** 자체 시뮬레이션도 쓰고 공개 시뮬레이션도 쓴다. 다만 직접 가까운 순차 표 생성 연구는 Berka 등 외부 관측 자료를 주요 실증 대상으로 사용한다. 현재 연구에 부족한 것은 시뮬레이터를 더 정교하게 만드는 일보다 **현재의 오류·개선이 외부 데이터에서 중요한지 연결한 증거**다.

## 2. 현재 데이터 선택과 전처리에서 바로잡을 점

[기존 수집/전처리 보고](../benchmark_v2/cof_seqgen_saf_data_collection_report_2026_08_26.md)에는 다음 자료의 materialization 완료가 기록되어 있다. 아래 숫자는 그 기록이며, 이번 문헌 검토에서 원격 원본 전체를 다시 해시 검사하거나 모델을 평가한 것은 아니다. 현재 이 checkout에는 `data/cof_seqgen_saf/`가 없으므로 실행 전 워크스테이션 manifest와 실제 파일을 다시 연결해야 한다.

| 데이터 | 성격 | 전처리 기록의 규모 | 적합한 역할과 제한 |
|---|---|---:|---|
| Berka full | 실제 은행 거래 | 4,500계좌 / 1,056,320거래 | 관측 금융 거래열 생성의 우선 실증. **사기 정답 라벨 없음**. 현재 mark는 `operation`, 없으면 `type`; 수취인 반복이 아님 |
| Sparkov fraudTrain | 외부 공개 시뮬레이션 | 983카드 / 1,296,675거래 | 가맹점·시간·금액·사기 라벨을 이용한 관계 및 탐지 효용. 실제 은행 사기 성능이라고 부르지 않음 |
| AMLSim | 외부 공개 시뮬레이션 | 9,999송금 계좌 / 1,323,234거래 | 송금 상대방·불법 거래 관계. 개별 거래열만으로 전체 금융 그래프 일관성을 입증할 수 없음 |
| H&M | 실제 구매 거래 | 10,000고객 / 236,028거래 | 금융 밖에서 일반화 확인. 일 단위 시각·동일 시각 구매의 순서 모호성을 처리해야 함 |
| 자체 controlled DGP | 정답 관계를 아는 자체 시뮬레이션 | 기존 κ0/1 조건 | 구현 검사·원인 분리·알려진 무관계 조건의 점검. 외부 효용의 대체물이 아님 |

출처: [Berka 공식 목록](https://relational.fel.cvut.cz/dataset/Financial), [Sparkov 생성기](https://github.com/namebrandon/Sparkov_Data_Generation), [AMLSim](https://github.com/IBM/AMLSim). 데이터 원본과 전처리 해시는 기존 수집 보고/설정에 기록되어 있다.

코드 확인에서 중요한 점은 다음과 같다.

- `data/cof_seqgen_saf_adapters.py`의 Berka adapter는 거래 날짜를 **일 단위**로 처리한다. 동일 날짜의 여러 거래에 초·분 단위의 참 순서가 있다고 가정하면 안 된다. 현재 mark는 거래 방식/유형이다. ‘같은 수취인 반복’이라는 용어를 그대로 이식할 수 없다.
- Sparkov adapter의 mark는 merchant다. 저장된 보고에 따르면 `unix_time`과 문자열 시각이 불일치하여 `trans_date_trans_time`을 기준으로 삼았다. 이 결정을 뒤집거나 두 시각을 섞으면 안 된다.
- 인공 데이터에서의 ‘관계가 없는 집단’은 생성 법칙으로 안다. 외부 자료에서는 자동으로 알 수 없다. 임의 집단을 비활성 정답으로 선언하여 이전 규제·판정 기준을 복제하면 안 된다.
- European Credit Card Fraud는 실제 관측 자료지만 익명 PCA 특징 중심이다. 공개 스키마에서 안정적인 고객 ID·해석 가능한 가맹점 이력이 없는 자료를 고객별 거래열 문제의 주 데이터로 강제하지 않는다.
- 현재 인공 데이터의 두 집단·유한 mark vocabulary·gap 지지점과 외부 schema는 다르다. **구조와 학습법을 고정하여 외부 자료에서 재학습**하는 것과 인공 데이터 checkpoint를 그대로 적용하는 것은 다르다. 전처리·차원·길이 처리를 검증하지 않은 CSV 교체는 공정한 비교가 아니다.

또한 **외부 실험을 한 번도 하지 않았다는 설명도 부정확하다.** [이전 Sparkov 저장 결과](../benchmark_v2/sparkov_external_validation_aggregate_forensic_v1.md)에서 구형 CoF는 관계 지표의 저장 순위가 가장 좋았지만, 주변분포·종합 점수는 empirical IID와 CTGAN보다 나빴다. 이는 현재 U/G 실험이 아니며, 현재 U/G가 Sparkov에서 잘될 것이라는 근거도 아니다. 외부 데이터를 쓰면 자동으로 결과가 좋아진다는 결론은 성립하지 않는다.

## 3. 수정이 얼마나 이어졌는가

CS-SAF oracle audit부터 현재까지 **주요 완료 결과 묶음 22개**를 아래처럼 확인할 수 있다. 하나의 묶음 안에 여러 시드·설정이 있고, 저장 checkpoint 진단에는 학습이 전혀 없는 경우도 있다. 따라서 **‘아키텍처를 22번 수정했다’나 ‘학습을 22번 했다’는 집계가 아니다.** 전처리, 문헌 검토, CPU smoke test, 이전 CoF 연구는 이 수에 넣지 않았다.

| 번호 | 완료 결과 묶음 | 성격·확인할 근거 |
|---:|---|---|
| 1 | [Oracle audit](oracle_audit_v1_report_2026_09_16.md) | 인공 관계의 식별 신호 점검 |
| 2 | [V1 pilot](pilot_v1_report_2026_09_16.md) | 초기 구조 학습 |
| 3 | [Checkpoint forensics](checkpoint_forensics_v1_report_2026_09_16.md) | 저장 모델 원인 진단 |
| 4 | [V2 pilot](v2_pilot_v1_report_2026_09_16.md) | 집단별 경로 분리 |
| 5 | [Loss control](loss_control_v1_report_2026_09_16.md) | 일반/전역/균형 손실 비교 |
| 6 | [Route decomposition](route_decomposition_v1_report_2026_09_16.md) | 이력 보정과 gap 변화 분리 진단 |
| 7 | [V3 pilot](v3_pilot_v1_report_2026_09_16.md) | 이력·중심화·잔차 규제 대조 |
| 8 | [V4 pilot](v4_pilot_v1_report_2026_09_16.md) | E 예측 구조를 유지한 규제 |
| 9 | [Replication](replication_v1_report_2026_09_16.md) | U/E/ER 시드·비율 확대 |
| 10 | [Follow-up](followup_v1_report_2026_09_17.md) | 규제 강도·비율·CPAR 비교 |
| 11 | [Rollout audit](rollout_audit_v1_report_2026_09_17.md) | 예측과 자유 생성의 차이 |
| 12 | [Replay/oracle](replay_oracle_v1_report_2026_09_17.md) | mark feedback과 공동 정답 평가 |
| 13 | [Calibration](calibration_v1_report_2026_09_20.md) | 저장 모델 반복 보정 |
| 14 | [U calibration control](calibration_u_control_v1_report_2026_09_20.md) | U에 동일 보정 적용 |
| 15 | [Generation repeats](generation_repeats_v1_report_2026_09_20.md) | 모델 고정·생성 난수 반복 |
| 16 | [Calibrated replay](calibrated_replay_v1_report_2026_09_20.md) | U/E 생성 이력 교차 평가 |
| 17 | [Gap calibration](gap_calibration_v1_report_2026_09_20.md) | 간격별 보정 용량 비교 |
| 18 | [Official external model audit](external_audit_v1/README.md) | 인공 자료에서 공식 ARGN 검증 |
| 19 | [Baseline adequacy](baseline_adequacy_v1/README.md) | ARGN 학습 연장·동일 단순 대조군 |
| 20 | [U/G/C structure](structure_v1/README.md) | 동일 용량의 구조 비교 |
| 21 | [A/B/P rollout calibration](rollout_calibration_v1/README.md) | 생성 관계 학습·예측 보호 비교 |
| 22 | [History/run diagnostic](history_diagnostic_v1/README.md) | 저장 결과의 이력·반복 상태별 진단 |

문제는 개별 비교에 아무 이유가 없었다는 것이 아니다. 이 정도의 탐색이 누적되었는데도 다음을 일찍 연결하지 못했다는 것이다.

1. **문제의 외부 타당성:** 최근 실험은 기존 인공 data seed42의 비율·학습·생성 시드를 바꾼 탐색이다. 재학습 시드를 바꿔도 새로운 데이터 생성 법칙/관측 모집단의 확인은 아니다.
2. **목표와 지표의 간격:** 원래 금융 행동 관계와 탐지 효용이 목표였지만, 실제 수정은 반복확률과 gap 반응 억제에 집중되었다. 그 지표 개선이 외부 탐지/예측에 필요한지의 근거가 약하다.
3. **반복된 검증 자료 활용:** 매 단계 사전등록은 유용하지만, 앞선 validation 결과를 보고 다음 구조를 만들면 연구 전체는 적응적 탐색이다. 개별 등록만으로 누적 선택 편향을 없앨 수 없다. 이는 test 누출이 발견되었다는 주장이 아니라, 현재 결과를 확증이라고 부를 수 없는 이유다.
4. **판정 기준과 용도의 연결:** 모든 국소 지표를 완벽하게 맞춰야 외부 데이터로 넘어갈 필요는 없다. 기존 실패를 성공으로 바꾸지는 않되, 외부 평가에서는 주요 효용과 허용할 비용을 사용 목적에 맞게 먼저 정해야 한다.
5. **구현 정확성과 방법의 타당성 혼동:** 검사 통과는 연구 기여·실데이터 적용성의 보증이 아니다. [기존 코드/수식 감사](history_diagnostic_v1/architecture_audit.md)는 점검한 경로에서 결과를 무효화하는 오류를 찾지 못했지만, 가중 성분 손실, 연속 학습 이력과 이산 생성 이력, G의 nonrepeat 확률 표현 제한을 남겼다. 이 중 무엇이 주원인인지 아직 분리되지 않았다.

현재 확실한 성과는 ‘조건부 예측 개선이 자유 생성 개선과 다를 수 있음’, ‘전체 평균이 맞아도 반복 상태별 오류가 남음’이라는 제한된 인공 데이터 진단이다. 예를 들어 실제 인공 validation 이력의 2–3회 반복 상태에서 U/G는 각각 약4.66/4.51%p 과소 예측하며, 생성 지속률도 oracle보다 약10%p 낮았다. **이 진단을 새로운 생성 방법의 우수성으로 보고해서는 안 된다.**

## 4. 다음 작업의 우선순위와 종료 조건

아래는 이번 검토의 **권고 프로토콜**이며 모델 실행 사전등록 완료를 뜻하지 않는다. 문헌 표를 만드는 것으로 외부 실증이 완료된 것도 아니다.

### 1단계: Berka와 Sparkov에서 문제를 먼저 확인

- 원격 원본·기존 split manifest·전처리 코드 버전을 확인한다. 최종 test는 탐색에 사용하지 않는다. 이번에 없는 자료를 새로 확보해야 하는지부터 구분한다.
- Berka는 거래 유형–시간 간격–금액/잔액, Sparkov는 가맹점/범주–시간 간격–금액–사기 라벨로 **행동의 정의를 데이터에 맞게 고정**한다. 반복 하나를 모든 금융 관계의 대리변수로 쓰지 않는다.
- 기존 train에서 관계와 표본 수를 정하고 validation에서 재현 여부를 확인한다. 정답 oracle TV 대신 관측 이력의 확률 예측 점수, 시간 조건부 전이, 연속 반복 분포, 금액 관계를 사용한다. Berka의 같은 날짜 순서·0 gap에 대한 민감도도 별도로 확인한다.
- 독립 실자료 표본끼리의 차이를 참고치로 계산하여, 모델의 차이가 표본 변동보다 큰지 평가한다. 이를 오차의 이론적 하한이나 동등성 증명이라고 부르지 않는다.
- **종료:** 해당 관계가 약하거나 불안정하거나 주요 용도와 무관하면 그 관계를 개선하는 새 모듈 개발을 시작하지 않는다. 반대로 인공 실험의 모든 항목 통과를 외부 문제 확인의 선행 조건으로 두지 않는다.

### 2단계: 현재 구조와 가까운 기존 생성기의 한 번의 공정한 외부 비교

- 현재 U/G와 기존 gap 보정의 구조·방식을 유지하고 외부 train에 재학습한다. 가까운 외부 순차 baseline은 **공식 TabularARGN**을 우선하고 CPAR 및 단순 관측 전이/반복 대조군을 둔다. CTGAN/FinDiff는 행 단위 참고 대조로 구분한다.
- 모델마다 같은 원본 필드, split, 예측 시점 정보, 생성 표본 수와 평가 코드를 사용한다. 외부 모델의 encoder까지 억지로 동일하게 만들지 않는다. 유효한 시퀀스 구성, 선택 규칙, 계산 예산을 실행 전에 고정한다.
- TabDiT는 직접 가까운 문헌이지만 [기존 공개 코드 감사](external_audit_v1/source_inventory.json)의 확인 commit에는 학습/생성 구현이 없다. 실행 가능성을 다시 확인하기 전 재현 실험 목록에 완료 모델처럼 넣지 않는다.
- 초기 비교는 고정된 소수 학습 시드로 제한한다. 데이터/길이별 자원 측정 후 실행 횟수와 시간을 등록하며, 결과를 보고 후보·계수를 계속 추가하지 않는다.
- **주 평가:** 데이터별로 사전에 고른 시간·행동 관계의 생성 오차. **보호 평가:** 관측 이력 예측, 금액·gap·행동 분포, 표본이 충분한 집단의 성능. 주요 개선 폭과 허용 비용은 train 내부 분할/용도/표본 변동을 근거로 정하고 모델 validation 결과에 맞춰 조정하지 않는다.
- **종료:** 단순 대조군 또는 외부 모델로 문제가 충분히 해결되면 현재 반복 보정/새 구조의 필요성을 주장하지 않는다. 모든 모델에 오류가 남고 크기·효용상 중요할 때만 그 오류의 원인을 분리한다.

### 3단계: 남은 문제 하나에 한정한 방법 개발과 효용 연결

- 외부에서도 반복 상태 의존 오류가 남을 때만 이전의 상태 × gap 대조군을 검토한다. 정보·용량이 같은 일반 모델보다 좋아야 구조 기여이며, 단순 보정으로 충분하면 복잡한 모듈의 기여는 없다.
- 사기 효용은 Sparkov의 미사용 외부 평가에서 실제 관측 원본 학습, 합성 학습, 원본+합성 학습을 비교한다. 여기서 ‘원본’도 Sparkov 시뮬레이션이다. PR-AUC와 고정 오탐률의 재현율 등 목적에 맞는 지표를 쓰며, 탐지 입력에 참 사기 라벨/미래 정보를 넣지 않는다. 새 고객 일반화와 동일 고객의 미래 탐지를 다른 과제로 보고한다.
- Berka는 거래열 예측·재현 효용을 먼저 확인한다. 대출 결과를 쓰려면 원래 대출 시점의 정보 제한을 지킨 별도 과제이며 사기 효용이라고 부르지 않는다. 실제 은행 사기 효용을 주장하려면 적합한 실제 사기 자료가 더 필요하다.
- 채택 후보는 별도 holdout·다른 데이터/생성 법칙에서 확인한다. 새로운 방법이 단순 대조군보다 추가 이득이 없으면 해당 방법 주장을 접는다. 진단 결과는 보존하지만 성공할 때까지 같은 평가 자료에서 수정하는 흐름을 반복하지 않는다.

**현재 연구 주장:** ‘관측 이력에서의 예측 정확도를 유지하면서, 생성된 금융 거래열의 시간·행동 관계와 해당 관계가 필요한 과제의 효용을 개선할 수 있는가?’는 연구 질문이다. **이를 해결한 새 아키텍처가 확보되었다는 주장은 아직 아니다.** 다음 구조는 외부 데이터에서 확인된 실패 원인에 따라 정해야 한다.
