# ibm_design_v2 — 두 설계 후보의 선행 연구 대조 (2026-11-11, 모델 변경·등록 전)

사용자 지시(2026-11-11): "진행"(두 설계 후보의 선행 연구 대조와 새 평가 틀 등록). 근거 자료는 [진단 결과](../ibm_gap_diag_v1/RESULTS.md). 이 문서는 AGENTS.md가 요구하는 네 항목(가장 가까운 선행 / 정확한 겹침 /
남은 문제의 증거 / 가르는 실험)을 후보별로 적는다. **여기서는 아무것도 구현·학습·생성하지 않았고, 새로움 주장도 없다.** 검색: 2026-11-11, 15회(extended 3회 포함), 직접 확인 4건(본문 2: Altman 2019·Handbook 시뮬레이터, 초록 2: 2509.18414·Banksformer). 그 밖의 고전 문헌은 이전 등록에서 확인한 것을 다시 적었다.

## 후보 1 — 사기 묶음(에피소드) 상태

**고칠 것.** 실제 사기는 고객당 한 번, 며칠(중앙값 3.1일·19.5사건) 동안, 끝나면 재진입이 없는 묶음이다(마지막 사기 뒤 51사건 이후 위험률이 기준 수준). 현재 전이표는 "이전 사기 있음"을 영구 상태로 두고
21사건 이후를 한 칸으로 묶어, 자유 생성에서 사기를 1.33–1.6배, 고객당 최대 133건(실제 27건)까지 낸다.

| 항목 | 내용 |
|---|---|
| 가장 가까운 선행(방법) | **명시적 지속 시간을 갖는 준마르코프 상태 모형**: Yu 2010(HSMM 개관), Chiappa 2014(명시적 지속 모형 정리), Dai et al. ICLR 2017(Recurrent HSMM — 구간을 RNN으로 생성), Dube et al. 2509.18414(시간 비동질 마르코프 "다음 활동" + 위험률 "지속 시간"으로 활동열을 **생성**; 순수 마르코프 대비 유의한 개선). |
| 가장 가까운 선행(자료 생성 원리) | 규칙 기반 시뮬레이터가 이미 "유한한 침해 기간"을 쓴다: Fraud Detection Handbook 시뮬레이터 시나리오 3("매일 고객 3명 선택, **이후 14일** 동안 거래의 1/3이 사기"), 시나리오 2(단말기 **28일**). 우리 자료의 생성기 Altman 2019(1910.03033)는 사기꾼 집단이 "특정 기간(몇 달–몇 년) 활동"하고 "정해진 일정·장소에서" 사기를 낸다고 적는다(카드당 건수·기간은 논문에 없음) — 달력 몰림(7–12월)과 맞는다. |
| 가장 가까운 선행(학습형 거래 생성기) | 읽은 범위에서 **없음**. Sajja 2604.13125는 burst·velocity 구조가 깨진다는 것을 재지만 처방이 없고("entity-aware autoregressive generation"을 향후 과제로 둠), 행 단위 사기 생성 모델(FraudDiffuse, EmDT 2603.13566, CTTVAE 2602.03641)은 고객열·묶음이 없다. |
| 내부 선행(기각된 시도) | `ibm_state_first_v1` M1: "마지막 사기 이후 사건 수·위치"를 포함한 6특징 **학습 분류기**(HGB)로 상태 과정을 바꿨다가 자유 생성에서 지속률·구간 수가 악화돼 기각("노출 편향"). 같은 정보를 쓰지만 형식이 다르다 — 그때는 자유 모의로 사전 검증하지 않았고, 묶음의 **종료**를 구조로 넣지 않았다. |
| 정확한 겹침 | 상태 과정 자체는 표준 준마르코프 모형이다(지속·종료 위험률을 거리 구간별로 추정). **상태 모형의 새로움은 주장하지 않는다.** |
| 겹치지 않을 수 있는 것(미확정) | 묶음 상태를 **다른 열의 조건**으로 쓰는 것 — 묶음 안에서의 가맹점 반복·집중(실제 .19 대 생성 .04, 상위 가맹점 집중도 7.7배)과 사기 시점. 학습형 고객 거래열 생성기에서 이를 한 것은 읽은 범위에 없으나, 없다고 단정하지 않는다. |
| 남은 문제의 증거 | D1: 자유 모의 1,729건 대 실제 1,303건, "이전 사기" 상태 체류 2.4배, 첫 사기 상대 위치 .45 대 .73; 봉인 test 사기율 3.30배; dev 생성 4개 중 1개가 732건. Sparkov에서도 같은 종류의 실패(사기 고객 수 6.75배). |
| 가르는 실험 | (1) **CPU 자유 모의 먼저**(D1의 모의기): 후보 표가 train의 사기 건수·구간 수·사기 고객 수·고객당 분포(최대 포함)·첫 사기 위치를 모두 바닥 범위 안으로 맞추는가. 건수만 맞추는 단순 표(이전 라벨만: 사기 고객 373명)와 구별돼야 한다. (2) 통과 시에만 GPU 생성: 사기율·진입률·24h 구간 수가 3배 안, 효용 비열화. (3) 제거 실험: 거리 구간 확장만 / + 종료 상태 / + 시작 위치 분포. |

## 후보 2 — 정상 행의 이력 참조 확장 (금액·최근성·간격)

**고칠 것.** 효용 격차(.10)가 전부 합성 정상 행에서 온다(합성 사기 + 실제 정상 = .597 = 상한). 정상 행에서 (고객, 가맹점)에 묶인 금액이 없고(재방문 금액 차이 중앙값 1.16 대 실제 .32), 가맹점 최근성이 없고
(직전과 같은 가맹점 .042 대 .175), 간격의 연속성이 없다(자기상관 .05 대 .19).

| 항목 | 내용 |
|---|---|
| 가장 가까운 선행(방법) | **복사·포인터 기제**: Vinyals 2015, Gu 2016(CopyNet), See 2017 — 이력에서 토큰을 복사할지 새로 생성할지의 혼합. **재소비 모형**: Anderson et al. 2014("최근성이 재소비의 가장 강한 예측 변수"), Benson et al. 2016(최근성 + 도착 간격 + 새로움 모형), RepeatNet(Ren 2019). **자기 여기 점과정**(Hawkes 계열, Shchur et al. 2021 개관) — 간격의 연속성. |
| 가장 가까운 선행(거래 생성기) | Banksformer(Nickerson et al. ECML-PKDD 2022): 거래 유형 → 날짜 → 금액 순 생성, 날짜 패턴이 목적(**본문 미확인** — 발표 자료·초록 기준; 금액의 조건화 방식 확인 못 함). UniTTab-AR / BankDiT(Garuti 학위논문, Luetto et al. MLJ 2025): 은행 거래 생성(**본문 미확인**, 접근 거부). REaLTabFormer(2302.02041): 창 전체를 문맥으로 금액 자릿수를 자기회귀 생성 — 명시적 기제 없이도 (고객, 가맹점) 안 금액 표준편차 1.10(실제 1.00)으로 **암묵적으로 상당 부분 담는다**(우리 측정). Karst et al. 2412.14730(은행 거래 생성 벤치마크): 다섯 모델 모두 표·그래프 충실도 평가이며 고객 이력 구조는 없음. |
| 자료 생성 원리 | Altman 2019: 소비자는 업종별 **개인 지출 분포**(평균·표준편차를 모집단에서 한 번 뽑아 고정)로 금액을 내고 단골 가맹점이 있다 — (고객, 업종/가맹점)에 묶인 금액은 자료의 설계에서 나온 구조다. |
| 내부 선행 | `ibm_merchant_memory_v1`의 포인터 커널은 빈도 대 최근성을 train hold-out 우도로 골라 빈도를 택했고, `ibm_merchant_marginal_v1`은 끈적임을 우도 +.011 nat로 기각했다. 우도 기준이 생성 통계의 결손을 재지 못했다(진단 결과에 기록). |
| 정확한 겹침 | 세 부분 모두 기제는 알려져 있다: 금액 = 수치 열에 대한 복사/새 생성 혼합(과거 금액 주변의 조건부 밀도), 최근성 = 재소비 모형의 표준 항, 간격 = 자기 여기. **기제의 새로움은 주장하지 않는다.** RTF가 암묵적으로 일부를 이미 한다는 점도 겹침이다. |
| 겹치지 않을 수 있는 것(미확정) | (a) "합성 거래열의 TSTR 격차가 사기 행이 아니라 **정상 행의 고객 이력 구조**에 있다"는 측정과 그것을 고치는 최소 부품. (b) 사기 쪽 부품(상태 먼저, 라벨 라우팅)과 정상 쪽 이력 참조를 한 원리("이력을 참조한다")로 묶는 것. 둘 다 Banksformer·UniTTab 본문을 읽기 전에는 확정할 수 없다. |
| 남은 문제의 증거 | D2: 행 교체(.492 → .597), 정상 행 판별기 AUC .825(금액 .19, 업종 .13, 간격 .09), 오탐·미탐 구성(합성 학습 탐지기가 "가 본 가맹점에서 평소와 다른 금액"의 사기를 놓침). Sparkov에서도 ARGN의 금액–가맹점/업종 결합이 약했다(업종 × 금액 쌍 TV 4.1–5.8배). |
| 가르는 실험 | (1) **CPU 모의 먼저**: train으로 추정한 부품이 정상 행 구조 표(가맹점 안·(고객, 가맹점) 안 금액 표준편차, 재방문 금액 차이, 직전 동일 가맹점, 간격 자기상관)를 실제 바닥 범위로 옮기는가. (2) 생성 후: 합성 + 합성 효용이 상한 쪽으로 가는가(행 교체 기준 .492 → .597이 상한선), 정상 행 판별기 AUC가 내려가는가, 표준 지표·가맹점 구조 비열화. (3) 제거 실험: 금액만 / 최근성만 / 간격만 / 전부. (4) 비교군: RTF + 전이표(암묵적 이력 참조를 가진 backbone). |

## 미해결(그대로 보고)

- Banksformer와 UniTTab-AR / BankDiT의 방법 절을 읽지 못했다(서버 연결 거부·403). 금액을 과거 금액에 조건화하는 명시적 기제가 있는지 **확인되지 않았다.** 후보 2의 기여 서술은 이 둘을 읽은 뒤에만 한다.
- Sparkov 생성기의 사기 기간 규칙은 코드를 확인하지 않았다(우리 측정만 있음).
- 따라서 지금 쓸 수 있는 것은 "두 후보 모두 진단 근거가 있고, 기제는 알려진 것이며, 학습형 고객 거래열 생성기에 적용한 선행은 읽은 범위에서 찾지 못했다"까지다.

## 출처

- Altman, Synthesizing Credit Card Transactions — https://arxiv.org/abs/1910.03033
- Fraud Detection Handbook, Transaction data simulator — https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/SimulatedDataset.html
- Dube et al., Hierarchical Semi-Markov Models with Duration-Aware Dynamics for Activity Sequences — https://arxiv.org/abs/2509.18414
- Dai et al., Recurrent Hidden Semi-Markov Model (ICLR 2017) — https://hanjun-dai.github.io/pdf/DaiDaiZhaetal17.pdf
- Sajja, Synthetic Tabular Generators Fail to Preserve Behavioral Fraud Patterns — https://arxiv.org/abs/2604.13125
- Nickerson et al., Banksformer (ECML-PKDD 2022) — https://mlanthology.org/ecmlpkdd/2022/nickerson2022ecmlpkdd-banksformer
- Luetto et al., One Transformer for All Time Series (MLJ 2025) — https://mlanthology.org/mlj/2025/luetto2025mlj-one/
- Karst et al., Generative AI for Banks — https://arxiv.org/abs/2412.14730
- Benson et al., Modeling User Consumption Sequences (WWW 2016) — https://cseweb.ucsd.edu/classes/fa17/cse291-b/reading/sequences-www2016.pdf
- Shchur et al., Neural Temporal Point Processes: A Review — https://arxiv.org/abs/2104.03528

---

## 추가 (2026-11-12) — 미해결이던 본문 확인: Banksformer · UniTTab · TabDiT, 그리고 같은 자료를 쓴 선행 1건

사용자 지시(2026-11-12): "채택하고 진행"(정상 행 부품 채택, 기여 서술 전에 미해결 선행부터 확인). 위의 "미해결" 첫 항목을 푼다. 위 본문은 고치지 않는다. 모델 변경·학습·생성 없음.
읽은 방법: Banksformer는 원 서버가 다시 연결을 거부해 **저자의 학위논문**(Nickerson 2023, 2장이 ECML-PKDD 논문 그대로임을 본문이 명시)의 방법·결과·향후 과제 절을 직접 읽었다. UniTTab은 arXiv v4 본문, TabDiT는 arXiv v2 본문과 부록. 검색 5회(extended 2회).

| 논문 | 무엇인가 | 한 거래 안의 생성과 금액 | 고객 이력을 **명시적으로** 참조하는 기제 | 사기 라벨 / 가맹점 | 평가 |
|---|---|---|---|---|---|
| **Banksformer** (Nickerson et al., ECML-PKDD 2022) | 은행 거래열 생성기(transformer decoder) | 필드를 정해진 순서로 생성: `p(tcode | hist) · p(date | hist, tcode) · p(amount | hist, tcode, date)`. 금액은 정규분포(평균·분산 출력). 날짜는 달력 특징(월의 날, 월말까지 남은 날, 요일, 월)과 간격 Δt의 분포를 곱해 날짜 하나를 뽑는 "date mechanism" | **없음** — 이력은 self-attention 문맥 `hist`로만 들어간다 | 둘 다 없음(거래 유형 코드 16–44종, 날짜, 금액뿐). 고정 길이 열(80 / 20건) | 단변량 W1·JSD, 3-gram JSD, (유형, 날짜) 결합 JSD, 월 현금 흐름, PCA 시각화. **TSTR 없음**("쓰임을 미리 모를 때는 가치가 덜하다"고 적음) |
| **UniTTab** (Luetto et al., MLJ 2025; arXiv 2302.06375) | **생성기가 아니다** — 마스킹 사전학습 표현 모형 + 하류 분류·회귀 | 수치는 주파수 임베딩, 예측 목표는 구간 + 이웃 구간 label smoothing | 없음(행 열 transformer의 attention뿐) | Padhi의 거래 자료에서 사기 **탐지**(F1)를 하류 과제로 씀. 생성 없음 | 하류 과제 성능만. 합성 자료 평가 없음 |
| **TabDiT** (Garuti et al., ICLR 2025; arXiv 2504.07566) — 이전 기록의 "UniTTab-AR / BankDiT"의 출판본 | 표 시계열 **확산** 생성기(행별 VAE 잠재 + DiT 잡음 제거, 부모 행 조건은 CFG). 비교용 자체 "AR Baseline" 포함 | 수치는 자릿수 표현(크기 자리 + 유효 숫자 4개). 행들을 함께 잡음 제거 | **없음** — 복사·포인터·최근성 기제 없음 | 사기 없음. 자료: Age1 / Age2 / Leaving, PKDD'99, Rossmann, Airbnb. 열 길이 상한 50–100행 | 판별 가능성(MLD-TS, LD-SR)과 부모 표 라벨 분류의 MLE-TS. REaLTabFormer보다 낫다고 보고. 본문에서 코드 주소는 찾지 못함(공개 여부 미확인) |
| Hsieh et al. (arXiv 2401.00965, 2024) | 우리와 **같은 자료원**(Altman)의 카드 거래열을 SDV의 CPAR로 생성 | 전처리 다섯 가지(금액 변환·범주 부호화)만 바꿈 | 없음 | 사기 행을 **따로** 생성(고객 3명, 사기 15,000건) | 주변 분포의 시각 비교, 합성 학습 탐지기의 FNR / FPR. 실제 학습 기준선 없음, 격차의 원인 분석 없음 |

### 겹침과 남는 것 (정정 포함)

- **후보 2의 미해결 항목은 풀렸다.** Banksformer·UniTTab·TabDiT 어디에도 "금액이나 가맹점을 그 고객의 과거 같은 가맹점·업종 행에 명시적으로 묶는" 기제는 없다. 셋 다 이력을 암묵적 문맥으로만 쓴다(RTF와 같은 부류).
  → "학습형 고객 거래열 생성기에 이력 참조 부품을 **명시적으로** 넣은 선행은 읽은 범위에 없다"고 쓸 수 있다. "없다"가 아니라 "읽은 범위에서 찾지 못했다"이다. 기제 자체(복사, 재소비, 준마르코프)의 새로움을 주장하지 않는다는 것은 그대로다.
- **새로 확인한 겹침 — 행 안의 조건부 생성 순서.** Banksformer의 두 번째 혁신이 "한 거래의 필드를 정해진 순서로 만들고 뒤 필드를 앞 필드에 조건화"하는 것이다(제거 실험 포함). 우리의 "상태 먼저"(라벨 → 간격 → 가맹점 → 금액 …)는 **형식상 같은 것**이다.
  남는 차이는 무엇을 맨 앞에 두는가(희귀한 사기 라벨)와 그 라벨로 출력층을 가르는 라우팅, 표로 둔 라벨 과정이며, 이것은 "순서를 정해 조건화한다"는 발상의 새로움이 아니라 **적용과 근거**(사다리)의 문제다. 논문에서는 Banksformer를 이 지점의 선행으로 인용해야 한다.
- **진단 주장(격차가 정상 행의 고객 이력 구조에 있다)의 선행은 여전히 찾지 못했다.** 위 넷 중 어느 것도 사기 탐지 TSTR의 격차를 사기 행 대 정상 행으로 가르지 않는다. 가장 가까운 것은 여전히 Sajja 2604.13125(행동 사기 패턴이 깨진다는 측정, 처방·위치 특정 없음)다.
  이번 검색(extended 2회)에서도 행 교체 실험을 한 논문은 나오지 않았다 — 검색 범위의 한계 안에서의 진술이다.
- **평가의 차이.** 위 생성기들은 판별 가능성·분포 거리·부모 표 라벨 분류로 평가한다. 사기 라벨이 있는 고객 거래열에서 탐지 효용(TSTR)을 상한·바닥과 함께 재고 부품별로 가르는 평가는 읽은 범위에 없다.

### 앞으로의 결정에 미치는 것

- **시간 구조 부품(다음 후보)의 가장 가까운 선행은 Banksformer의 date mechanism이다**(달력 특징의 분포를 곱해 날짜를 뽑음). 그 부품을 등록할 때 네 항목 대조의 첫 줄에 놓는다. 우리 진단(생성된 "1년"이 280–540일, 사기 진입의 달력 몰림)이 가리키는 것이 같은 기제로 풀리는지부터 본다.
- **비교군.** 심사에서 최신 생성기를 요구받으면 후보는 TabDiT다. 다만 (i) 코드 공개 여부 미확인, (ii) 그 논문의 열 길이 상한이 50–100행인데 우리 과제는 고객당 연 1,100건 안팎, (iii) 사기 라벨·가맹점 9,411종 같은 고차원 범주를 다룬 실험이 없다.
  넣을 수 있는지는 코드와 길이 제약을 확인한 뒤에 정한다 — 지금은 "확인 필요"로 둔다. Banksformer는 코드가 있으나 가맹점·사기 라벨이 없는 3필드 모형이라 그대로는 비교군이 되지 않는다.
- **아직 미해결**: Sparkov 생성기의 사기 기간 규칙(코드 미확인), TabDiT 코드 공개 여부, Garuti 학위논문 자체(출판본 TabDiT로 대신 읽음).

### 출처 (추가)

- Nickerson, A deep generative model framework for creating high quality synthetic transaction sequences (PhD thesis, Memorial University, 2023; 2장 = Banksformer 논문) — https://hdl.handle.net/20.500.14783/14762
- Banksformer 코드 — https://github.com/BigTuna08/Banksformer_ecml_2022
- Luetto et al., One Transformer for All Time Series — https://arxiv.org/abs/2302.06375
- Garuti et al., Diffusion Transformers for Tabular Data Time Series Generation (ICLR 2025) — https://arxiv.org/abs/2504.07566
- Hsieh, Wang, Cheng, Improve Fidelity and Utility of Synthetic Credit Card Transaction Time Series from Data-centric Perspective — https://arxiv.org/abs/2401.00965
