# ibm_realtabformer_v1 — 두 번째 backbone(REaLTabFormer)에 같은 두 부품 (등록 2026-11-08, 학습 전)

목적: [ibm_routed_heads_v1](../ibm_routed_heads_v1/RESULTS.md)에서 기여 C의 두 부품 — (i) 생성 시점 전이표,
(ii) 라벨 라우팅 사기 전용 출력 head — 이 동결 ARGN 위에서 성립했다. "backbone 불문" 주장을 하려면 구조가
전혀 다른 두 번째 생성기에서 **같은 두 부품을 같은 데이터·같은 평가**로 얹어야 한다. REaLTabFormer
(GPT-2 encoder-decoder, relational 모드; [realtabformer_sparkov_v1](../realtabformer_sparkov_v1/PROTOCOL.md)에서
Sparkov에 한 번 실행)를 IBM 2013 과제에 학습하고, 그 위에 두 부품을 **학습 변경 없이(전이표) / 동결
backbone 위 head 학습만으로(라우팅 head)** 올린다.

## 1. 관측 오류와 근거

- IBM 원형 ARGN([ibm_tabformer_v1](../ibm_tabformer_v1/RESULTS.md)): 사기 구간 배분(사기 고객 3배, 지속률 0)과
  상태별 출력(사기 온라인 .27, 실제 .88) 둘 다 깨짐 → 전이표(구간 축, 바닥 1.2배 이내)와 라우팅 head(출력
  축, 바닥 1.5–2.3배)로 복구, 효용 .003→.14.
- Sparkov REaLTabFormer([결과](../realtabformer_sparkov_v1/RESULTS.md)): 창 안 구간 지속은 보존했지만 사기 진입을
  **과소** 배분(사기 고객 68–75 대 114)했고 표준 지표(업종 TV 18배, 고객당 가맹점 21배)가 깨졌다. ARGN과
  반대 방향의 배분 오류이므로, 같은 전이표가 양쪽을 고쳐야 backbone 불문 주장이 선다(2026-11-08 결정).
- REaLTabFormer는 IBM에서 아직 실행된 적이 없다. 원형 측정(진단 재현)도 이 단계의 일부다.

## 2. 가장 가까운 선행

- backbone: REaLTabFormer 0.2.4 공식 패키지(Solatorio & Dupriez 2023), 열별 토큰 제약 샘플링(`prefix_allowed_tokens_fn`).
- 부품 (i)의 선행: 디코딩 시점 logits 개입(plug-and-play / constrained decoding; 표형 LM에서는 REaLTabFormer·GReaT의
  열 어휘 제약). 여기서는 라벨 열 위치의 logits를 **외부 전이표**(train 추정, 학습 0)로 바꾼다 — 열 어휘
  제약과 같은 자리에 들어가는 외부 확률 대체이며 새 기법이 아니다.
- 부품 (ii)의 선행: class-conditional / hard-routed expert output layers(long-tail decoupled heads). ARGN 단계와 같다.
- 자체 선행: `ibm_generation_module_v1`(전이표), `ibm_routed_heads_v1`(라우팅 head), `realtabformer_sparkov_v1`(창 과제).
- 기제가 등록된 ARGN 부품과 같고 backbone만 바뀌므로 새 문헌 조사는 하지 않았다. GPT-2 계열 backbone에서
  같은 부품이 성립하는지가 이 단계의 유일한 질문이다.

## 3. 중복과 남는 가설

- **T0(원형 재현)**: IBM에서 REaLTabFormer 원형의 구간 배분·상태별 출력·표준 지표를 ARGN 원형과 같은 평가기로
  측정한다(가설 없음, 기록).
- **T1(전이표)**: 전이표(+창 간 상태 이월)가 구간 축(사기 고객 수, 24h 구간/사기 고객, 지속률, 고객당 구간 수)을
  원형 REaLTabFormer 비율의 절반 이하로 줄이는가. 바닥 3배 이내면 강한 채택. 원형이 이미 3배 이내인 축은
  "악화 없음(≤ 원형 비율 + 1)"으로 판정.
- **T2(라우팅 head)**: 사기 출력 4지표(사기 온라인 비율, 사기 금액 W1, 사기 간격 W1, 사기 업종 TV)가 전이표
  팔 대비 절반 이하 또는 바닥 3배 이내인가. 라벨 단독 P(온라인) ≥ .6.
- **T3(정상 불변)**: 표준 지표(업종 TV, 금액·간격 W1, 고객당 가맹점)가 원형 REaLTabFormer 대비 ±.03 이내.
- **T4(효용)**: stable 이력 AUPRC가 원형 REaLTabFormer보다 높고 ≥ .2면 지지, ≥ .4면 강한 지지.
- **backbone 불문 판정**: T1·T2가 ARGN과 REaLTabFormer 양쪽에서 채택되면 "두 부품이 두 backbone에서 같은 두
  축을 복구한다"고 적는다(IBM 한 데이터, 시뮬레이터). 한쪽만 채택되면 부품이 backbone 특성에 의존한다고
  적고 어느 부품이 어디서 실패했는지 기록한다.

## 4. 비교 실험

### 데이터·창
- `ibm_tabformer_v1/prepared/{train,validation}.parquet` + `*_context.parquet`(ARGN과 동일 분할; test 봉인 유지).
- 자식 창: 고객별 `event_index` 순 **W건**(마지막 창은 나머지). W는 `prepare`가 잰 최장 창 토큰 수 + 16이
  디코더 한도 1,024 이하인 가장 큰 값을 {60, 50, 40}에서 고른다(규칙 사전 고정; 선택값 기록).
- 자식 열 순서: **event_is_fraud 첫 열**, gap(정수 초, 첫 행 0), receiver_or_mark, amount_or_numeric_value(소수 2자리),
  category, use_chip. 라벨 첫 열은 라우팅·전이표가 같은 행의 라벨을 참조하기 위한 조건이며, Sparkov RTF
  (라벨 마지막)와 다른 점으로 기록한다. 원형 팔도 같은 fit을 쓰므로 팔 간 비교는 영향 없다.
- 부모 행: 고객 정적 열(first_year, cards_before, events_per_year_before, home_state) + window_index + window_rows.

### 원형 fit
- `model_type="relational"`, `numeric_nparts=2`, `train_size=.9`, `early_stopping_patience=5`, batch 8, Sparkov와 동일
  설정. **fit 2회**(seed 20261108 / 20261109, GPU 1 / 2 병렬), epoch 상한은 1 epoch 시간을 재어 각 5시간 이내,
  최대 40. 저장 오류 대비 복구 경로(`--recover`) 유지.

### 부품 (i) 전이표 + 창 간 이월
- 표: `ibm_generation_module_v1/transition_table.json`(train 추정, P(사기 | 직전 라벨, 직전 구간 나이, 과거 사기)) 그대로.
- 생성: 라벨 열 위치에서 모델 logits를 표의 log p로 교체하는 `LogitsProcessor`. 상태(직전 라벨·구간 나이·과거
  사기)는 같은 창에서 생성된 라벨 토큰으로 계산하고, **창 경계는 이전 창의 마지막 상태를 다음 창의 초기
  상태로 이월**한다(고객별 창을 window_index 순으로 라운드 생성). 이월은 전이표의 "과거 사기" 조건이 창
  독립 생성에서는 정의되지 않기 때문에 필요한 최소 장치이며, 원형 팔에는 적용하지 않는다(Sparkov 설계
  유지). 전이표 외의 열은 모델 그대로.

### 부품 (ii) 라벨 라우팅 사기 전용 head
- 구조: 디코더 마지막 블록의 `ln_2 + mlp`, `ln_f`, 그리고 `lm_head`의 **라우팅 열 토큰 행만 잘라낸 복사본**
  (gap·amount·category·use_chip의 모든 sub-column 토큰; 가맹점·라벨 제외 — ARGN과 같은 열 집합). 원형
  가중치로 초기화, 원형 파라미터 전부 동결·불변.
- 라우팅: 토큰 위치 → (행 k, sub-column j)는 결정적이다(`[EOS][BOS]([BMEM] c0..c_{S-1} [EMEM])*`). 행 k의
  라벨 토큰(첫 열, 이미 생성/입력됨)이 사기면 그 행의 라우팅 열 위치의 logits를 사기 head로, 아니면
  원형으로. 학습은 teacher-forced 동일 규칙. 열 어휘 제약(허용 토큰 외 −∞)은 양쪽에 똑같이 적용.
- 손실: 사기 행의 라우팅 열 토큰만, 열 어휘 안의 CE, 가중 없음. 정상 행은 손실에 들어가지 않는다.
- 팔: `routed`(실제 문맥) / `routed_subst`(사기 행의 비라벨 **필드**를 같은 창의 정상 행 필드로 치환한 입력,
  목표는 원래 토큰; p=1). 숫자 열은 필드 단위로 치환하므로 한 행 안의 부분 토큰은 치환 행끼리 일관된다.
- 최적화: Adam 1e-3, batch 8창, 최대 100 epoch, 조기 종료(patience 5)는 **원형 fit 자신의 eval 분할**(패키지가
  `train_size=.9`, seed=fit seed로 나눈 10% 창 — 원형이 학습하지 않은 창)의 사기 행 라우팅 토큰 평균 CE로
  하고 열별 CE는 기록만 한다. 분할은 같은 데이터셋에 같은 seed로 `train_test_split`을 다시 적용해 복원하고,
  fit 시점에 기록한 eval 창 수·앞 예시 해시와 일치함을 확인한다(development·test는 어디에도 쓰지 않는다).
- 실행 전 검사(`check`): 미학습 복사본의 라우팅 모델 = 원형 logits(최대 차 < 1e-4), 사기 head 교란 시 사기
  행·라우팅 열 위치만 변화. Sparkov fit으로 기제를 먼저 검사하고 IBM fit이 끝나면 다시 검사한다.

### 생성·팔
- 팔 4개 × fit 2 × draw 2(seed 20261011 / 20261012) = **16회**, development 240명, 실제 창 수·창 길이 조건
  (`related_num=window_rows`, Sparkov와 동일한 유리한 조건으로 기록):
  `rtf_native`(원형, 창 독립), `rtf_table`(전이표+이월), `rtf_table_routed`, `rtf_table_routed_subst`.
- 비교군: ARGN 원형, transition_label_first, routed_table, routed_subst_table(모두 기존 생성물).

### 평가
- 1단계 평가기(`--dataset ibm`, 등록표에 `realtabformer` 가족 추가, 바닥 200회), 효용 stable, Spearman 재계산,
  라벨 경로 탐침(정상 행에서 라벨 입력만 1로 바꿔 P(온라인)), teacher-forced 열별 CE(holdout 100명).

### 예산
- 원형 fit 2(각 ≤ 5h), head 학습 4(각 ≤ 30분), 생성 16(각 ≤ 1h), ARGN 전체 fit 0.

## 5. 판단 조건

§3의 T1–T4와 backbone 불문 판정. 채택되더라도 "현행 방법 대비 우위"가 아니라 "두 부품이 두 backbone에서
같은 축을 복구한다"의 확인이다. IBM 하나(시뮬레이터)에서의 확인이며 실제 데이터 검증은 남는다.

## 6. 실행 후 갱신

W 선택값·토큰 수, 제외 창 수, fit 시간·epoch, head 파라미터 수·학습 로그, 생성 16개 지표·효용, 열별 CE,
라벨 경로 탐침, 실패를 `RESULTS.md`·CSV·JSON에 남긴다.
