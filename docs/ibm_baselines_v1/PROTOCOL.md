# ibm_baselines_v1 — 대안 비교군: 표준적인 희귀 클래스 처방 대 등록 모듈 (등록 2026-11-09, 학습 전)

목적: 기여 C("동결 backbone 위 전이표 + 라벨 라우팅 사기 전용 head가 드문 상태 축을 복구하는 **최소·충분**
부품")의 "최소·충분"은 지금까지 **원형 대 모듈**의 비교로만 뒷받침됐다. 리뷰어가 먼저 묻는 질문은
"그냥 사기를 oversampling 하거나 손실에 가중을 주거나 라벨 조건부로 전체를 재학습하면 되지 않는가"이다.
이 단계는 그 **표준 처방들을 같은 backbone(원형 ARGN, IBM 2013)·같은 평가(1단계 지표·효용)**에 놓아
(i) 각 처방이 어느 축(구간 배분 / 상태별 출력)을 고치고 어느 표준 지표를 대가로 치르는지,
(ii) 등록 모듈이 그것들보다 낫거나, 같거나, 못한지를 **사전 판단 조건**으로 가른다. 새 모델이 아니며,
결과가 어느 쪽이든 기록한다(모듈이 지면 "최소·충분" 주장을 접고 그 사실을 기여 A·B의 결과로 적는다).

## 1. 관측 오류와 근거

- 원형 ARGN(IBM): 사기율은 2배 안이지만 사기 전부 길이 1 고립(지속률 .00–.01, 실제 .65), 사기 고객 3배,
  사기 온라인 비율 .22–.31(실제 .88), 효용 이력 AUPRC .003(상한 .61) ([ibm_tabformer_v1/RESULTS.md](../ibm_tabformer_v1/RESULTS.md)).
- 등록 모듈(전이표 + `routed_subst` head, 전체 fit 0): 구간 축 바닥 1.2배 이내(고객당 구간 수 3–5배 제외),
  사기 온라인 .84, 사기 금액 W1 2.3배, 효용 **.14** ([ibm_routed_heads_v1/RESULTS.md](../ibm_routed_heads_v1/RESULTS.md)).
- 두 번째 backbone(RTF): 전이표만으로 효용 .29→.45 ([ibm_realtabformer_v1/RESULTS.md](../ibm_realtabformer_v1/RESULTS.md)).
- 아직 없는 것: **표준 처방과의 비교**. Sparkov에서 거래별 손실 가중 재구현(`B_event_weighted`)은 구간 길이
  1.01(실제 10.05)로 무너졌으나 그것은 재구현 backbone(B)이고 원형 엔진이 아니다. 라벨-첫-열은 생성 순서
  변경(G2, 기각)으로만 시험했고 **학습 순서 고정**은 시험하지 않았다.

## 2. 가장 가까운 선행 (2026-11-09 재확인)

| 처방 | 선행 | 본 단계와의 겹침 | 남는 차이 |
|---|---|---|---|
| 소수 클래스 oversampling | SMOTE 계열; 표 생성기 비교에서 표준 기준선([CTTVAE 2602.03641](https://arxiv.org/abs/2602.03641v1), [ORD AAAI 2025](https://arxiv.org/pdf/2412.15657)) | 방법 자체는 선행 그대로 | 순차 생성기에서 **고객열 단위** 복제가 구간 배분·상태별 출력·사기율에 미치는 영향은 측정된 적 없음 |
| 비용 민감 손실 가중 | 클래스 균형 손실·focal loss; [Climb 2505.17451](https://arxiv.org/html/2505.17451v2)는 극단 불균형에서 단순 재가중이 해롭다고 보고 | 사기 행 CE 가중, 표준 그대로 | 자기회귀 순차 생성기 손실에 행 가중을 넣었을 때 생성물 효용은 보고 없음 |
| 클래스 조건부 생성 | CTGAN 조건 벡터·training-by-sampling(Xu 2019), ORD/CTabSyn의 class-conditional 학습, CTTVAE(2026) | 라벨을 다른 열의 조건으로 **학습 시 고정** | 순차 ARGN은 any-order 학습이 기본이라 조건이 희석된다는 가설을 고정 순서 재학습으로 검증 |
| 생성 시점 주변분포 재조정 | mostlyai 엔진 `rebalancing`(라벨열 우선 + 고정 확률); [2507.16419](https://arxiv.org/html/2507.16419v1)가 TabularARGN 재조정을 upsampling 기준선으로 사용 | 엔진 기능 그대로 호출 | 주변분포 제어가 **구간 배분**(상태 의존 전이)을 바꾸지 못한다는 것이 전이표와의 차이 |

읽은 범위에서 **순차 거래 생성기**에 위 처방을 적용해 희귀 상태 구간·상태별 출력·효용으로 평가한 선행은 찾지
못했다(검색 2회, 2025–2026 표 생성 불균형 문헌). 없다고 단정하지 않는다.

## 3. 가설 (사전 고정)

- **A1 (대가·trade-off).** oversampling과 손실 가중은 사기 행의 상태별 출력(온라인·금액·간격·업종)을 원형보다
  개선하지만 **사기율·사기 고객 수를 복제 배수만큼 부풀려** 표준 지표 중 하나 이상이 바닥 3배를 넘는다.
- **A2 (모듈의 충분성).** 표준 지표 대가 ≤ 3배인 비교군 중 효용(이력 AUPRC, fit 평균)이 모듈(.14) 이상인 것은
  없다. 있으면 "최소·충분" 주장을 **그 비교군에 대해 철회**하고, 모듈의 남는 장점(학습량·동결 backbone·부품별
  해석)만 적는다.
- **A3 (전이표의 결합성).** 어떤 비교군이든 생성 시점에 전이표를 얹으면(학습 0) 구간 축 지표가 비교군 단독보다
  좋아진다. 비교군 + 전이표가 모듈 이상의 효용을 내면 "두 번째 부품(라우팅 head)은 학습 변경으로 대체 가능"으로
  기록하고 그 대체의 비용(학습 파라미터 수·사기율 왜곡)을 함께 적는다.
- **A4 (진단이 처방을 예측).** fit별 teacher-forced 라벨 경로 탐침(라벨 단독 P(온라인|사기), 나이 1 사기 확률)이
  자유 생성의 상태별 출력·구간 축 복구 여부를 fit 12개(비교군 10 + 원형 2) 중 ≥ 10개에서 맞게 예측한다
  (예측 규칙: 라벨 단독 P(온라인) ≥ .6 ⇔ 사기 온라인 비율 바닥 3배 이내; P(사기|나이 1) ≥ .3 ⇔ 지속률 바닥 5배 이내).

## 4. 설계

데이터·분할·엔진·평가는 [ibm_tabformer_v1](../ibm_tabformer_v1/PROTOCOL.md) 그대로(IBM 2013, train 1,000 /
development 240 / 봉인 test 미읽음, mostlyai-engine 1.0.4 `./run`, Medium, GPU). 모든 비교군은 **원형 fit과 같은
`prepared/child` 워크스페이스에서 출발**하며 아래 한 가지만 바꾼다. fit seed 20261108·20261109, 생성 seed
20261011·20261012(기존과 동일). 스크립트 `scripts/run_ibm_baselines.py`.

| 팔 | 바꾸는 것 | 학습량 | 비고 |
|---|---|---|---|
| `rebalance` | 생성 시점 `rebalancing={column: event_is_fraud, probabilities: {'1': p_train}}`(엔진 기능) | 0 (원형 fit 재사용) | 라벨열 우선·고정 주변확률; 전이표의 "주변분포만" 대응물 |
| `oversample_x3`, `oversample_x8` | 엔진 `split` 뒤 **trn 파티션에서만** 사기 경험 고객열을 복제(×3, ×8; 새 customer_id, 문맥도 복제). val 파티션·analyze·encode는 그 위에서 원형과 동일 | 전체 | ×8은 trn 고객 수준에서 사기 고객 ≈ 정상 고객(균형). 내부 epoch cap(ceil(n/50))은 커진 n에 따라 자연히 늘어남 — 원형 규칙 그대로 |
| `weight_sqrt`, `weight_inv` | `_calculate_sample_losses`의 행 마스크를 사기 행에서 w배(격리 소스 사본, `relaxed_cap`과 같은 방식; 설치 엔진 불변). w_sqrt = round(√(n_normal/n_fraud)), w_inv = round(n_normal/n_fraud) (train 행 기준, prepare 시 계산·기록) | 전체 | 모든 열 손실에 적용(라벨열 포함). 엔진의 표본 단위 정규화(∑mask)는 그대로 두므로 사기 포함 고객열 안에서 사기 행 비중이 w배가 된다. val 손실도 같은 함수 → 조기 종료 기준도 가중 |
| `fixed_label_first` | `train(enable_flexible_generation=False)` + 목표 열 순서를 **라벨 → gap → merchant → amount → category → use_chip**으로 고정 | 전체 | any-order 희석 없이 모든 head가 매 행 라벨을 입력으로 받음. 생성 순서도 같은 고정 순서 |

- 각 전체 학습 팔: fit 2 × 생성 2(단독) + 생성 2(**+전이표**, `benchmarks/argn_generation_module.py`의 전이표
  훅, 학습 0). `rebalance`: 원형 fit 2 × 생성 2. 총 **새 fit 10**, 생성 **44**(4 + 5×2×2×2), ARGN 외 모델 없음.
- 전이표는 [ibm_generation_module_v1](../ibm_generation_module_v1/PROTOCOL.md)의 train 전이표 그대로(재추정 없음;
  oversampling 팔에도 **원래** train 표를 쓴다 — 복제된 사기율이 아니라 실제 사기율을 복원하려는 것이므로).
- 진단: 각 fit에 `scripts/probe_native_episode.py --dataset ibm`(나이별 사기 확률)과 라벨 단독 출력 탐침
  (라벨만 사기로 두고 나머지 입력을 정상 행으로 치환했을 때 P(온라인), [ibm_output_heads_v1](../ibm_output_heads_v1/PROTOCOL.md)의 분해).
- 평가: `run_rare_state_persistence.py --dataset ibm`(바닥 200회 재사용) + `run_fraud_utility.py --dataset ibm
  --detector stable` + `summarize_fraud_utility.py`. 등록표에 family `baselines`, model `ARGN_<arm>[_table]_ibm2013`.
- 비용 기록: 팔별 학습 파라미터 수(전체 모델 대 라우팅 head), fit 시간, 학습 행 수.
- 하지 않는 것: 비교군의 하이퍼파라미터 탐색(w·배수는 위 공식으로 한 번만), 비교군 위에 라우팅 head 추가,
  REaLTabFormer 비교군(이 등록 밖), 봉인 test 읽기.

## 5. 판단 조건

모든 비율은 1단계 바닥(실제 대 실제 200회 중앙값) 대비, fit 2 × 생성 2의 중앙값. 효용은 fit 평균.

1. **A1 채택**: oversampling·가중 4팔 중 ≥ 3팔에서 사기율 또는 사기 고객 수 비율 > 3. 기각: ≤ 1팔.
2. **A2 채택(모듈 충분)**: 표준 지표(가맹점 축 제외) 최대 비율 ≤ 3인 비교군 단독 팔 중 효용 ≥ .14인 것이 없다.
   **기각**: 하나라도 있으면 그 팔을 이름으로 적고 "모듈이 그 팔보다 효용에서 낫다"는 문장을 모든 문서에서
   철회한다(RESEARCH_DECISIONS에 날짜 항목).
3. **A3**: 비교군 + 전이표 ≥ 비교군 단독(지속률·사기 고객·나이 1 종료 비율 모두)이 5팔 중 ≥ 4팔이면 채택.
   비교군 + 전이표의 효용이 .14 이상인 팔이 있으면 "head 대체 가능(비용 X)"로 기록.
4. **A4**: 예측 규칙 일치 ≥ 10/12이면 채택, ≤ 8/12이면 기각(진단→처방 규칙을 주장에서 제외).
5. 비교군이 모든 축에서 모듈보다 낫고 표준 대가도 ≤ 3배이면: 기여 C를 "최소 모듈"에서 **"진단이 가리키는
   표준 처방의 선택 규칙"**으로 재정의하고, 그 전환을 명시한다.

## 6. 실행 후 갱신

fit 로그·가중치 해시·생성물·탐침·1단계/효용 표·실패를 `RESULTS.md`와 CSV로 남긴다. 복제 trn 파티션의 고객 수·
행 수·사기율, 가중치 w 값, 격리 소스 사본 diff를 함께 보존한다. 생성물·가중치는 서버
`artifacts/ibm_baselines_v1/`에만 둔다.
