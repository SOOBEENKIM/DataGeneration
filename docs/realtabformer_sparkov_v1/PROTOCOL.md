# realtabformer_sparkov_v1 — 2단계(2) 프로토콜 (등록 2026-11-07, 실행 전)

목적: 가설 (a) "표준 집계 지표는 통과해도 사기 구간 동역학은 깨진다"를 ARGN이 아닌 표준
순차 생성기에서 확인한다. REaLTabFormer(Solatorio & Dupriez 2023, relational 모드)는
Seq2Synth 등 최근 벤치마크가 공통으로 쓰는 비교군이며 이 프로젝트에서 실행된 적이 없다.

## 1. 관측 오류와 근거

`rare_state_persistence_v1/RESULTS.md`: 원형 ARGN 지속률 오차 바닥의 33배, 표준 1.5배.
CPAR·MeLD·LBDTPP는 표준 지표부터 깨져 (a)의 사례가 아니다. ARGN 외 사례가 없다.

## 2. 가장 가까운 선행 방법

REaLTabFormer 0.2.4 공식 패키지(GPT-2 encoder-decoder, 부모 행 조건 자식 시퀀스 생성).
확인한 제한: 자식 시퀀스는 디코더 1,024 위치 안이어야 하고, 한 행이 `numeric_nparts=2`에서
약 13–14 토큰이므로 부모당 최대 약 70행. 한도를 넘는 부모는 학습에서 **제외**된다(잘리지 않음).
Sparkov 고객은 평균 약 1,300건이므로 고객 전체를 한 자식 시퀀스로 둘 수 없다.

## 3. 중복과 남는 가설

REaLTabFormer 적용 자체는 선행이다. 남는 질문은 이 생성기가 창 안의 사기 구간 지속(나이 1 전이,
구간 길이)을 보존하는가뿐이다. 과제 차이를 명시한다: 고객 시퀀스를 60건 창으로 나누고 창을
독립 자식 시퀀스로 학습·생성한다(ARGN의 전체 이력·자유 길이 과제보다 쉬운 설정).

## 4. 비교 실험

- 데이터: `sparkov_argn_control_v2/prepared/train.parquet`(688명)과 `train_context.parquet`.
  자식 창 = 고객별 `event_index` 순서로 60건씩(마지막 창은 나머지). 부모 행 = 고객 정적 열 +
  `window_index`(0부터) + `window_rows`(창 길이). 자식 열 = gap(정수 초, 첫 행 0), receiver_or_mark,
  amount_or_numeric_value(소수 2자리), category, event_is_fraud.
- 학습: `model_type="relational"`, `numeric_nparts=2`, `output_max_length`는 60행이 들어가도록
  측정값으로 설정, `train_size=0.9`, `early_stopping_patience=5`, `epochs` 상한은 1 epoch 시간을
  재어 총 8시간 이내로 정하고 기록. GPU 1개(index 1). seed 20261107 하나(예산상 1 fit).
- 생성: development 147명 각각 실제 길이와 같은 창 수·창 길이를 부모 행으로 주고(`related_num=window_rows`),
  생성 seed 20261011/20261012 두 번. 창을 이어 붙여 `entity_id, event_index` 복원.
  **실제 길이를 조건으로 준다**는 점은 ARGN(자체 길이 생성)과 다른 유리한 조건이며 결과 표에 명시한다.
- 평가: `scripts/run_rare_state_persistence.py`의 등록표에 `other_generator/REaLTabFormer_w60`로 추가하고
  같은 바닥으로 비율을 계산한다. 창 경계가 실제 구간을 자르는 효과는 실제 데이터에도 같은 60건
  경계를 적용한 "실제(창 절단)" 참조 행을 함께 계산해 분리한다.

## 5. 판단 조건

- 표준 지표 비율 중앙값 ≤ 3 이면서 지속률 또는 구간 길이 비율 > 5 → 가설 (a)의 비-ARGN 사례 확보.
- 둘 다 바닥의 3배 이내 → REaLTabFormer는 창 과제에서 구간을 보존한다. ARGN 실패는 전체 이력·자유
  길이 설정의 문제일 가능성을 기록하고 3단계 모듈의 비교군으로 채택한다.
- 표준 지표도 > 5 → 학습 부족으로 분류, 예산 내 1회 재학습(epoch 증가)만 허용하고 그래도 같으면
  "충분히 학습하지 못함"으로 보고한다.

## 6. 실행 후 갱신

학습 시간, epoch 수, 제외된 창 수, 생성 행 수, 창 절단 참조값, 실패를 함께 기록한다.
