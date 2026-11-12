# ibm_generation_module_v1 — 원형 ARGN에 생성 시점 전이 모듈 (등록 2026-11-08, 생성 전)

목적: [ibm_tabformer_v1](../ibm_tabformer_v1/RESULTS.md)에서 확인한 원형 ARGN의 구간 배분 실패
(지속률 .00, 사기 고객 3배, 사기 출력이 정상화)를 **학습 없이 생성 시점에서** 고치는 최소
모듈이 작동하는지 본다. 3단계 모듈(기여 C) 후보의 첫 IBM 적용이며, Sparkov의 전이 head
가족(`argn_gap_episode_v1`)과 달리 **원형 엔진의 동결 가중치**에 그대로 건다.

## 1. 관측 오류와 근거

원형 fit 2개(가중치 b0b1f46a…, c5f0089b…)는 teacher-forced에서도 P(사기|직전 사기)=.003(관측 .74).
cap 완화 대조군이 가중치까지 동일했으므로 학습량으로는 고칠 수 없고, 라벨 head가 이력 속
라벨 상태를 쓰지 못하는 것이 원인이다. 또 라벨이 마지막 열이라 같은 행의 use_chip·금액·간격이
라벨을 보지 못해 사기 거래가 정상처럼 생성된다(온라인 비율 .88→.22–.31).

## 2. 가장 가까운 선행

- 자체 선행: Sparkov `transition_only`/`episode_only`(2026-09-27) — optimization 고객에서 추정한
  직전 라벨×구간 나이(×과거 사기 경험) 전이표로 라벨을 샘플링. 단 그것은 재구현 backbone(B,
  past-state 특징·digit 코덱)의 생성 훅이었다.
- Sparkov `order_category_*` 대조(2026-09-28): 유연 생성(any-order 학습)을 이용한 생성 시점 열 순서
  변경. 재학습 없음.
- 문헌: 명시적 지속 시간 상태 모델(HSMM; REDSLDS AISTATS 2025), CTGAN 조건부 샘플링, TabularARGN
  v2 §3.5–3.6의 any-order 생성. 전이표 자체는 알려진 통계 대조이며 새 모델이 아니다.

## 3. 중복과 남는 가설

전이표·열 순서 각각은 선행과 겹친다. 남는 것: (G1) 동결된 원형 엔진에 **생성 시점** 두 개입만으로
구간 배분이 바닥 3배 이내로 돌아오는가(backbone·학습 무관성의 첫 증거), (G2) 라벨을 첫 열로 두는
것만으로 상태별 출력(온라인 비율·사기 금액·간격)이 따라오는가, 아니면 전이표가 함께 있어야 하는가,
(G3) 두 개입이 정상 거래 품질(표준 지표)을 해치지 않는가, (G4) 효용(stable 탐지기)이 .003에서
실제 상한(.61) 쪽으로 움직이는가.

## 4. 비교 실험

- 가중치: `ibm_tabformer_v1/runs/native_{20261108,20261109}` 동결. **새 fit 0.**
- 전이표(`fit_table`): train 1,000명(봉인 test·development 미사용)에서 직전 라벨 ∈{없음,0,1} ×
  직전 사건의 구간 나이(1–20, 21+ pooled) × 과거 사기 경험(0/1) 칸마다 P(사기)를 Jeffreys .5
  smoothing으로 추정. 첫 사건은 train 첫 사건 사기율(0건 → .5/(n+1)). 미관측 칸은 같은 직전 라벨의
  나이 pooled 값으로 대체. 길이 조건(joint)은 쓰지 않는다. 사기 prior를 바꾸지 않는다.
- 생성 훅(`benchmarks/argn_generation_module.py`): 엔진 1.0.4 `SequentialModel`을 생성 모드에서만
  감싸 (i) 고객별 메모리(직전 라벨, 구간 나이, 과거 사기)를 LSTM 상태 튜플 뒤에 붙여 continue_mask
  필터와 함께 흐르게 하고, (ii) 라벨 sub-column의 predictor 출력을 전이표 log 확률로 교체한다.
  다른 열은 엔진이 그대로 샘플링한다. 출력 후 재라벨·비율 강제·복사 없음.
- 열 순서(`label_first`): `_resolve_gen_column_order` 결과에서 라벨 열을 SLEN/SIDX/SDEC 바로 뒤
  (첫 실제 열)로 옮긴다. 학습은 any-order였으므로 재학습 없음.
- 팔 4개 × fit 2 × draw 2(seed 20261011/12, 원형과 동일) = 생성 16회, development 240명 문맥:
  `order_only`(라벨 첫 열, 엔진 라벨 head), `transition_only`(원래 순서, 전이표), `transition_label_first`
  (둘 다), 그리고 `native`(기존 4개 재사용, 생성 안 함).
- 평가: `run_rare_state_persistence.py --dataset ibm`(등록표에 세 팔 추가), `probe_native_episode.py`
  는 가중치가 같으므로 다시 돌리지 않고 생성물 필드 비교만, `run_fraud_utility.py --dataset ibm
  --detector stable`.

## 5. 판단 조건

- **G1 채택**: `transition_*` 두 fit 평균에서 지속률·구간 길이·나이 1 종료·고객당 구간 수·사기 고객 수
  비율이 모두 바닥 3배 이내. 하나라도 5배 이상이면 기각(전이표 구조가 IBM 묶음을 못 잡는 것으로
  기록).
- **G2**: `order_only`에서 사기 온라인 비율·사기 금액·사기 간격 비율이 원형의 절반 이하로 줄면 "열
  순서만으로 상태별 출력 회복". `transition_only`에서는 줄지 않고 `transition_label_first`에서만 줄면
  "전이+순서 결합 필요"로 기록. 둘 다 안 줄면 상태별 출력은 학습 문제로 남긴다.
- **G3**: 표준 지표 절대 악화 ≤ +.05(Sparkov screen), 사기율 바닥 2배 이내.
- **G4**: stable 탐지기 이력 AUPRC가 원형 .003에서 .2 이상으로 오르면 효용 측 지지.
- 이 결과로 "새 방법의 우위"를 주장하지 않는다. 확인하는 것은 실패가 생성 시점 개입으로 복구
  가능한 종류인가이다.

## 6. 실행 후 갱신

전이표, 생성 16개 지표, 탐침 필드, 효용, 실패를 `RESULTS.md`·CSV에 남긴다. 생성물은 서버에만 둔다.
