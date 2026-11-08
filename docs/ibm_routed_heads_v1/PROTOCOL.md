# ibm_routed_heads_v1 — 라벨 라우팅 출력 head(사기 행 전용 head) (등록 2026-11-08, 학습 전)

목적: [ibm_output_heads_v1](../ibm_output_heads_v1/RESULTS.md)·[ibm_masked_heads_v1](../ibm_masked_heads_v1/RESULTS.md)
에서 동결 backbone 위 **공유** head 재학습은 치환 학습까지 더해도 라벨 단독 P(온라인)이 .25–.27(목표
≥ .6)에서 멈추고 자유 생성 사기 온라인 비율 .34–.35(실제 .88)에 그쳤다. 라벨 임베딩 경로는 살아
있으므로(거리 1.2, regressor 직접 입력) 남는 설명은 정상·사기 거래가 은닉 16유닛을 공유해 생기는
**용량·간섭**이다. 이 단계는 간섭을 구조적으로 없앤다: 정상 행은 원형 head 그대로, 사기 행은
**별도 head**로 라우팅한다. head 수준에서 상태별 출력 축을 복구하는 마지막 시도다.

## 1. 관측 오류와 근거

- `ibm_masked_heads_v1/LABEL_PATH_PROBE_*.csv`: 정상 행에서 라벨 입력만 1 → P(온라인) .25/.27(mask100);
  사기 가중 884·치환 p=1에서도 epoch당 +.02.
- Sparkov `argn_amount_learning_v1`: `amount_balanced_shared` .78 < `amount_balanced_routed` .85(같은 방향).
- 공유 head의 손실은 정상(가중 합 50%)과 사기(50%)의 타협이고, 가중을 더 키우면 표준 지표가 깨진다
  (이미 경계).

## 2. 가장 가까운 선행

- 혼합 전문가/조건부 라우팅: 입력 조건(여기서는 라벨)에 따라 별도 파라미터를 쓰는 출력층. 게이트를
  학습하지 않고 **라벨로 고정 라우팅**하므로 hard-routed expert heads(class-conditional output layers)
  에 가깝다. 희귀 클래스에 전용 파라미터를 두는 것은 long-tail 분류의 decoupled/expert heads와 같은
  계열이다.
- 자체 선행: Sparkov `amount_balanced_routed`(라벨 조건 금액 head, 전체 학습), IBM 공유 head 재학습 2건.
- 새 모델 구조가 아니라 **동결 backbone 위 상태별 출력 파라미터 분리**이며, 기여 C의 "상태별 출력
  부품"의 최소 형태 후보다.

## 3. 중복과 남는 가설

라우팅·전문가 head 자체는 선행과 겹친다. 남는 것: (R1) 사기 전용 head가 자유 생성에서 사기 온라인
비율·사기 업종·사기 금액·간격을 원형의 절반 이하로 줄이고(바닥 3배 이내면 강한 채택) 라벨 단독
P(온라인) ≥ .6에 이르는가. (R2) 사기 head를 실제 사기 문맥으로 학습한 것(`routed`)과 치환 문맥
(p=1, `routed_subst`)으로 학습한 것 중 어느 쪽이 생성에 옮겨가는가 — 간섭이 없을 때도 노출 편향이
남는지를 가른다. (R3) 정상 출력은 구조상 원형과 동일해야 한다(표준 지표 변화 ≤ ±.03 — 사기 행
수·이력만 다름). (R4) 효용(stable 이력 AUPRC)이 .009에서 .2 이상으로 오르는가.

## 4. 비교 실험

- 기반: 동결 원형 fit 2개(`ibm_tabformer_v1/runs/native_{20261108,20261109}`), 라벨-첫-열 고정 순서,
  생성은 라벨 첫 열 + 전이표(`ibm_generation_module_v1`). 전체 fit 0.
- **구조**(`benchmarks/argn_routed_heads.py: attach_routed_heads`): gap·amount·category·use_chip의 모든
  sub-column에 대해 regressor(ModuleList)+predictor(Linear)의 **복사본**(원형 가중치로 초기화)을
  `fraud_regressors`/`fraud_predictors`로 추가한다. 라벨 sub-column 임베더의 pre-hook이 행의 라벨
  코드에서 라우팅 마스크를 만들고(학습: 실제 라벨 (B,T,1); 생성: 그 행에서 방금 뽑은 라벨 (B,1,1)),
  regressor·predictor의 forward hook이 마스크로 `where(사기, 사기 head, 원형 head)`를 돌려준다.
  원형 파라미터는 전부 동결·불변(해시 확인). 가맹점 head는 제외(이전 등록과 동일).
- **손실**: 사기 행만(라벨=1, padding 제외) 열별 CE, **가중 없음**(사기 행 안에서 natural). 정상 행은
  손실에 들어가지 않는다(원형 head가 그대로 담당). 학습 사기 행 1,303건(train 1,000명).
- **팔**: `routed` (실제 문맥), `routed_subst` (ibm_masked_heads_v1의 `substitute_inputs`, p=1: 사기
  행의 비라벨 입력 토큰을 같은 창의 정상 행 토큰으로 치환, 목표는 원래 값) × fit 2 = head 학습 4회.
- 최적화: Adam 1e-3, 창 100, batch 8, 최대 100 epoch, 열별 조기 종료(patience 5) — 기준은 내부 val
  100명(사기 127건)의 사기 행 열별 CE, 각 팔은 자기 학습 조건의 문맥(실제/치환)으로 판정하고 둘 다
  기록. development·test 고객은 어디에도 쓰지 않는다.
- 생성: 2팔 × fit 2 × draw 2 = **8회**(validation 240명 문맥, seed 20261011/20261012). 비교군: 원형,
  transition_label_first, heads_balanced_table, mask100_table.
- 평가: 1단계(`--dataset ibm`, 등록표 추가), 효용(stable), teacher-forced 열별 CE(라우팅 모델), 라벨
  경로 탐침(정상 행 라벨→1은 구조상 사기 head로 가므로 "정상 문맥에서의 사기 head 출력"을 직접
  잰다).

## 5. 판단 조건

- **R1 채택**: 한 팔의 두 fit 평균에서 사기 온라인 비율 비율이 원형(17)의 절반 이하이고 라벨 단독
  P(온라인) ≥ .6. 바닥 3배 이내면 강한 채택. 둘 다 아니면 기각 → **head 수준 복구 불가능**으로
  결론하고 backbone 학습 변경(라벨 조건부 손실)으로 넘어간다.
- **R2**: 두 팔의 생성 측 사기 출력 지표 비교로 기록.
- **R3**: 표준 지표 절대 변화 ≤ ±.03(위반 시 라우팅 누수를 의심하고 구현을 점검).
- **R4**: 이력 AUPRC ≥ .2 지지, ≥ .4 강한 지지.
- 채택되더라도 "현행 방법 대비 우위"가 아니라 "동결 backbone 위 최소 상태별 출력 부품이 성립한다"
  의 확인이다. 두 번째 backbone 없이는 backbone 불문 주장을 하지 않는다.

## 6. 실행 후 갱신

학습 로그(열별 두 val CE), head 해시, 생성 8개 지표·효용, 열별 CE, 라벨 경로 탐침, 실패를
`RESULTS.md`·CSV에 남긴다.
