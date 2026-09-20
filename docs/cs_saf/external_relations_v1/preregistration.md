# Berka / Sparkov 외부 관계 확인 v1 — 2026-09-21

## 범위

기존 원본·canonical 자료·entity split을 고정한 **탐색적 관계/전처리 진단**이다. 자체 DGP 오류를 통과해야 외부 자료를 볼 수 있다는 조건을 두지 않는다. 신경망·생성 모델·보정 후보를 학습하지 않는다. 관측 전이/수치 평균의 단순 예측표는 관계의 유용성을 확인하는 진단 대조군이며, 자유 생성 성능이나 새 방법의 기여를 입증하지 않는다.

실행 전 이미 본 정보: 기존 보고의 파일 해시·행/개체 수·schema·split 크기·Berka 전체 zero-gap 수, 원본 경로, 전처리 코드. 행동별/시간별 관계 수치는 아직 보지 않았다. 기존 구형 Sparkov 모델 결과는 과거에 평가되었으므로 이번 validation도 연구 전체의 완전히 새 확증 자료라고 부르지 않는다.

## 데이터와 정보 제한

- 원본 경로: 상위 폴더의 `cof-seqgen-0707-2119-Version3-complete/data/cof_seqgen_saf/`.
- Berka full: 기존 train 3,150계좌, validation 675계좌. Sparkov fraudTrain: train 688카드, validation 147카드. 기존 test 개체와 Sparkov fraudTest의 결과는 분석하지 않는다. 전체 원본 파일 byte hashing은 허용하되 test outcome을 분석하는 것과 구분한다.
- parquet에서 train/validation ID 필터와 필요한 열만 읽는다. 원본 CSV/ASC의 독립 검산은 chunk를 읽고 해당 ID에 한해 분석한다. 반환되는 분석 frame에는 test 행이 없어야 한다.
- 사기 라벨은 **결과/기술적 집계용**이다. Sparkov `entity_any_fraud`와 현재/미래 fraud 라벨을 단순 행동·금액 예측표의 입력으로 쓰지 않는다. 기존 label-aware 층화 split은 유지하고 새 결과를 보고 재분할하지 않는다.
- train 개체를 SHA256(dataset, entity_id, seed=20260921) 순서로 80% fit / 20% internal check로 나눈다. gap 양수20/40/60/80% 분위 경계·amount 95% 분위·vocabulary는 fit에서만 정한다. validation을 보고 경계/설정을 바꾸지 않는다. fit→internal-check와 전체 train→validation의 방향을 같이 보고한다.

## 시각·행동·순서

- Berka: 일 단위 날짜, mark=`operation`(결측 시 `type`), 별도 `transaction_type`, 금액·잔액. 현재 schema에서 mark는 수취인이 아니다.
- Sparkov: 문자열 시각에서 계산한 초 단위 timestamp, mark=merchant, 별도 category, amount, fraud 결과. `unix_time`과의 차이가 일정한 이동인지/변동인지 확인하되 기존 문자열 기준을 유지한다.
- 첫 거래에는 이전 행동/반복 결과를 만들지 않는다. entity 경계를 넘지 않는다. 마지막 거래열 종료를 반복 중단으로 만들지 않는다.
- 주 전이 분석: 현재 날짜/시각과 직전 날짜/시각에 각각 거래가 하나뿐인 **순서가 명확한 인접 쌍**. 제외 비율을 공개한다. 반복 길이는 이러한 명확한 연결이 끊기면 재시작하는 보수적인 길이로 정의한다.
- 민감도: 모든 인접 쌍을 포함한 기존 정렬, 같은 시각 내 event ID 역순, 고정 난수101/102/103의 같은 시각 내부 재정렬. gap·이전 행동·반복 길이를 매번 다시 계산한다. 이 민감도로 참 순서를 알아냈다고 해석하지 않는다.

## 사전에 고정한 집계

1. split별 개체/거래/전이 수, 길이 분위수, 같은 시각 사건 비율, 0 gap 비율, 누락·음수·미등록 mark, 첫 gap 규칙, split 분리, 원본/canonical 동일성.
2. gap bin(0 별도 + 양수 분위5개)별 mark 반복률, category/type 반복률, log1p(amount) 평균; 이전 반복 길이1/2–3/4–7/8+별 지속률. gap×run 교차표를 모두 공개한다. 최소500전이/30개체 미만은 해석 부족으로 표시한다.
3. 정규화한 전이 관계: M0 전체 mark 빈도; M1 이전 mark별; M2 이전 mark×현재 관측 gap bin별 예측. M0에 Jeffreys .5 smoothing, M1/M2에 상위 분포 pseudo-count20 고정. OOV는 별도 unknown 범주로 평가하고 마스킹하지 않는다. 내부 check/validation NLL과 M1−M0, M2−M1의 paired 차이를 보고한다. validation에서 보정하거나 계수를 찾지 않는다.
4. 금액: log1p(amount)의 A0 전체 평균, A1 현재 mark별, A2 현재 mark×gap bin별 shrinkage 평균(pseudo-count20)의 MAE. 첫 거래 제외. 현재 mark/gap을 관측한 조건부 관계 진단이며, 미래 gap까지 미리 안다는 예측 성능으로 해석하지 않는다.
5. Sparkov: 사기 빈도/개체 수와 gap·category·학습자료 기준 amount 상위5%에 따른 fraud 비율. 탐지 모델·TSTR 미실행. 희소 집단 결과는 수와 불확실성을 함께 공개한다. Berka 잔액은 거래 유형별 금액/잔액변화 기술 집계만 수행하며 단순 회계식이 참이라고 가정하지 않는다.

불확실성: 개체 단위 paired bootstrap500회(seed20260922), event-weighted mean/rate의 기술적95% percentile 구간. 개체 균등 평균도 보조 보고한다. 여러 관계를 보는 탐색 결과로서 다중비교 보정된 확증 p-value를 주장하지 않는다. 전이 NLL의 추가 개선이 내부/validation 둘 다 양수인지, validation 구간이0 아래인지, 상대개선1% 이상인지 구분해 보고하며 한 기준만으로 새 모듈 필요성을 선언하지 않는다. train/validation 차이는 독립 개체 분할 사이 분포 이동도 포함한다.

## 산출물과 종료

표·그림·검산·입력/실행 manifest, 관계가 안정적인 부분과 불안정/순서 의존적인 부분, 다음 비교의 데이터별 목표를 저장한다. 같은 데이터의 정답 조건부 확률을 아는 것처럼 oracle TV를 계산하지 않는다. 실제 데이터 두 표본 차이는 모델오차의 이론적 하한이 아니다.

이 진단에서 자동으로 새 구조를 만들거나 튜닝하지 않는다. 다음 비교를 구체화할 때 현 U/G의 controlled-only schema(2개 context, 보조 필드 없음, 길이32 제한)를 외부 자료에 그대로 적용할 수 있는지도 코드로 점검한다. 외부 schema를 조용히 누락하거나 새로운 구조를 기존 U/G라고 부르는 방식은 금지한다. 비교 구현이 필요한 경우 변경·대조군·학습 예산을 별도 고정하고 CPU 검증을 거친다.
