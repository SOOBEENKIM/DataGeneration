# ibm_output_heads_v1 — 동결 엔진 위 라벨 조건부 출력 head 재학습 (등록 2026-11-08, 학습 전)

목적: [ibm_generation_module_v1](../ibm_generation_module_v1/RESULTS.md)에서 전이표는 구간 축을 복구했지만
사기 거래의 **상태별 출력**(온라인 비율 .88→.18–.27, 사기 금액·간격 W1 6–10배)은 생성 시점
개입으로 전혀 바뀌지 않았고 효용도 .002에 머물렀다. 원인 가설: 원형 엔진의 출력 head들이 라벨
임베딩을 쓰지 않도록 수렴했다(사기 .1%는 손실에 기여하지 않음). 이 단계는 **원형 가중치를 동결한
채 출력 head(regressor+predictor)만** 라벨-첫-열 고정 순서로 다시 학습해, 상태별 출력이 복구되는지
와 그것이 효용으로 이어지는지 본다. 3단계 모듈(기여 C)의 두 번째 부품이다.

## 1. 관측 오류와 근거

`ibm_tabformer_v1/stage1`: 원형·order_only·transition_* 모두 사기 온라인 비율 비율 17–20, 사기
금액 W1 7.6–9.7, 사기 간격 W1 5.8–8.3. 표준 지표는 1.3–3.5로 정상 출력은 괜찮다. 효용(stable)
실제 상한 .61 대 생성물 .001–.003. Sparkov 효용 재측정에서 행 단독 탐지기와 사기 금액·간격 W1의
|ρ|가 .79–.81로 가장 컸다 — 상태별 출력이 효용의 직접 통로다.

## 2. 가장 가까운 선행

- 자체 선행(Sparkov): `argn_amount_learning_v1`(natural/balanced amount head), `argn_category_amount_v1`
  — B backbone 위에서 사기 라벨 조건부 출력 head를 자연 빈도/50:50 가중으로 학습. 효용 재측정에서
  amount_balanced_routed .85, category_only .07(업종만 바꾸면 붕괴).
- 문헌: 클래스 가중/재표본 학습(CTGAN training-by-sampling §4.3), 동결 backbone + head 미세조정
  (일반적 전이학습), TabularARGN v2 §3.5–3.6 any-order 조건부 출력. 새 구조가 아니라 **동결 backbone
  위 head 재학습 + 드문 상태 가중**이라는 알려진 조합이다.

## 3. 중복과 남는 가설

head 재학습·클래스 가중 각각은 선행과 겹친다. 남는 것: (H1) 원형 엔진의 history·embedder·context를
그대로 두고 출력 head만 라벨-첫-열로 다시 학습하면 사기 거래의 온라인 비율·금액·간격·업종이
바닥 3배 이내로 돌아오는가(= 이력 표현에는 정보가 있고 head가 안 썼을 뿐인가). (H2) 자연 빈도
학습(natural)으로도 되는가, 사기 가중(balanced)이 필요한가. (H3) 정상 출력 품질 비용(표준 지표)이
+.05 이내인가. (H4) 전이표와 결합하면 효용(stable 이력 AUPRC)이 .002에서 실제 상한(.61) 쪽으로
크게 오르는가; head만 고치고 전이표를 빼면(라벨 동역학은 깨진 채) 효용이 얼마나 오르는가.

## 4. 비교 실험

- 기반: `ibm_tabformer_v1/runs/native_{20261108,20261109}` 동결 가중치. 각 fit마다 head 학습 2회
  (natural, balanced) = **새 head 학습 4회**(전체 fit 0). 엔진 내부 학습 파티션(trn 900명 / val 100명,
  원형 fit과 동일 분할)만 사용. development·test 미사용.
- 학습 대상: 출력 열 5개(gap, merchant, amount, category, use_chip)의 regressor 층과 predictor.
  embedder·column embedder·history LSTM·context compressor·attention·SLEN/SIDX/SDEC head·**라벨 head는
  동결**. 열 순서는 학습·생성 모두 SLEN/SIDX/SDEC → 라벨 → gap → merchant → amount → category → use_chip
  고정(any-order 아님).
- 손실: 엔진과 같은 sub-column 교차 엔트로피, 패딩 마스크 동일. `natural`은 행 가중 1,
  `balanced`는 사기 행 가중 = (trn 파티션 정상 사건 수 / 사기 사건 수), 가중 평균. 각 팔은 자기
  목적함수로 val 파티션을 매 epoch 평가(개발 세트로 고르지 않음). Adam 1e-3, batch 8 시퀀스, 창 100
  (엔진 BatchCollator 그대로), dropout 유지, 최대 30 epoch, patience 3, 최적 epoch head 저장. seed = 기반
  fit seed.
- 생성(development 240명 문맥, draw 20261011/12): `heads_natural_table`, `heads_balanced_table`(둘 다
  라벨 첫 열 + 전이표 + 재학습 head), `heads_balanced_no_table`(라벨 첫 열 + 재학습 head, 라벨은
  엔진 head). 12 생성. 비교군: 기존 native, transition_label_first.
- 평가: `run_rare_state_persistence.py --dataset ibm`(등록표 추가, 바닥 200회), `run_fraud_utility.py
  --dataset ibm --detector stable`. 탐침은 라벨 head가 동결이라 생략.

## 5. 판단 조건

- **H1 채택**: `heads_balanced_table` 두 fit 평균에서 사기 온라인 비율·사기 금액·사기 간격·사기 업종
  비율이 모두 바닥 3배 이내(원형 17–20, 7.6, 8.3, 2.7). 하나라도 원형의 절반 이상 남으면 "이력 표현
  자체가 드문 상태를 담지 못한다"로 기록하고 head 재학습으로는 불충분하다고 적는다.
- **H2**: natural이 balanced와 같은 수준이면 가중은 불필요; natural이 원형 수준이면 가중 필수.
- **H3**: 표준 지표 절대 악화 ≤ +.05, 정상 금액·간격 W1 비율 변화 ≤ 1.
- **H4**: `heads_balanced_table` 이력 AUPRC ≥ .2(원형 .003)이면 효용 측 지지, ≥ .4이면 상한(.61)
  방향 강한 지지. `heads_balanced_no_table`과의 차이가 전이표의 효용 기여.
- 이 결과는 "동결 backbone + head 재학습으로 복구 가능한가"의 확인이며 새 방법의 우위 주장이
  아니다. Sparkov로의 역이식과 backbone 무관성은 별도 등록이 필요하다.

## 6. 실행 후 갱신

학습 로그(epoch별 trn/val 손실), head 가중치 해시, 생성 12개의 지표·효용, 실패를 `RESULTS.md`·CSV에
남긴다. 가중치·생성물은 서버에만 둔다.

## 수정안 1 — 검증 창 (2026-11-08, 첫 학습 시도 중, 생성 전)

첫 시도(엔진 BatchCollator의 창 100 슬라이싱을 val에도 적용)에서 내부 검증 100명의 사기 사건이
epoch마다 0–28개만 창에 들어와 balanced 검증 손실이 사용 불가능할 만큼 흔들렸다(patience 3으로
epoch 4에서 멈출 수 있는 상태). val은 **전체 시퀀스**(창 없음)로 평가해 사기 사건 전부(약 130개)가
매 epoch 들어가게 하고 patience를 5로 올린다. 학습 쪽 창 100·batch 8·Adam 1e-3은 그대로다. 첫
시도의 로그는 `heads_attempt1_windowed_val/`에 보존한다.

## 수정안 2 — 가맹점 head 제외, 학습률 (2026-11-08, 수정안 1 학습 완료 후, 생성 전)

수정안 1 학습(4회, 각 6 epoch, 최적 epoch 1)에서 val 손실이 1 epoch부터 단조 상승했다. 열별 val CE
(`val_ce_by_column_20261108.csv`, 원형 대 재학습): 가맹점 head(9,411 수준, 학습 파라미터 139만 중
대부분)가 정상 행에서 6.59→6.99로 즉시 과적합하며 조기 종료를 지배했고, use_chip의 사기 행 CE는
원형 1.95 → natural 2.66 / balanced 2.42로 오히려 나빠졌다(1 epoch, 배치당 사기 1건에 가중 884가
걸리는 잡음). 업종은 정상·사기 모두 개선(1.50→1.35, 3.38→3.08), 금액·간격은 변화 없음.
수정: (1) 가맹점 head는 원형 그대로 두고 gap·amount·category·use_chip head만 재학습한다(가맹점
분포는 이 단계의 측정 대상이 아니다), (2) 학습률 1e-3 → 3e-4. 그 외 동일. 수정안 1의 head는
`heads_attempt2_all_columns/`로 보존하고 생성에는 쓰지 않는다. 판단 조건은 바꾸지 않는다.

## 수정안 3 — 구현 오류 정정과 epoch 예산 (2026-11-08, 생성 전)

- **구현 오류**: 수정안 1·2의 학습은 엔진 파라미터 이름(`regressors.regressors.regressor@…`)을 잘못 맞춰
  **predictor 선형층만** 학습했다(25,404 파라미터). regressor 층을 포함하도록 고쳤다(326,204 파라미터,
  가맹점 제외). 그 전 결과는 `heads_attempt3_predictors_only/`에 보존한다. 프로토콜의 학습 대상
  정의는 바뀌지 않았다.
- 정정 후 30 epoch(수정안 2 설정)에서 val 손실이 30 epoch까지 단조 감소해 예산에 걸렸다. 열별 val CE
  (`attempt4_30epochs/`): balanced의 사기 use_chip CE 1.95→1.50 / 2.08→1.37(원형 대비 개선, 상한 약 .43에는
  멀다), natural은 2.65/2.57로 악화, 업종은 둘 다 크게 개선(3.3→1.7, 정상도 1.5→1.0), 금액·간격 변화 미미.
- **예산**: max_epochs 30 → 100(patience 5 유지). 다른 설정은 수정안 2와 같다. 판단 조건은 바꾸지 않는다.
