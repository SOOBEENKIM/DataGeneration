# 외부 U/G 이식 및 첫 비교 파일럿 — 실행 전 고정

2026-09-21. 앞선 관측 관계 진단 후의 **탐색적 외부 파일럿**이다. 실행 전 본 정보는 기존 외부 관계 결과와 이번 무학습 CPU/처리량 검사다. 지금 새 외부 checkpoint의 성능은 아직 보지 않았다. 외부 전체 데이터의 예측/생성을 처음 비교해 중요한 잔여 오류를 확인한다. 논문 채택용 우수성 확증이 아니다.

## 공통 데이터

- 기존 train/validation 개체 분할 유지. train 내부 SHA256 80/20 fit/check는 이전 외부 관계 진단과 동일. Berka fit 2,520계좌/587,527거래, check 630/150,319; Sparkov fit 550카드/715,729거래, check 138/200,838. 가중치 학습은 fit, 선택은 check. outer validation 675계좌/147카드, test 사용 없음. 모든 fit 거래를 사용하며 길이/개체 수를 잘라 적은 데이터 결과로 바꾸지 않는다.
- Berka: gap, operation(type fallback), amount, transaction_type; static account_frequency, district, city, region, account_open_day. balance/k_symbol 제외: 회계·잔액 생성 과제가 아님.
- Sparkov: gap, merchant, amount, category; static gender, state, birth_year, city_population. 전체 기간 사기 여부·거래 사기 라벨·가맹점 위경도 제외. 사기 탐지·공간 관계 효용 과제가 아님.
- 간격/수치/어휘는 U/G와 단순 대조군에서 fit-only. 미등록 범주와 결측은 구분하며 미등록 정답을 마스킹하지 않는다. 양수 gap이 zero atom으로 인코딩되지 않도록 외부용 support의 첫 경계를 정확히0으로 둔다. 옛 helper/결과는 바꾸지 않는다.
- 공식 ARGN은 public split callback으로 같은 fit/check 개체를 지정한다. 공식 분석은 내부 check도 포함한 **outer train**으로 인코딩 통계를 구성하는 기본 동작을 유지하고 명시한다. 이는 U/G의 fit-only 전처리보다 넓은 정보이며 완전히 동일한 내부 전처리 조건이라고 하지 않는다. outer validation/test는 어떤 모델의 fit/인코딩에도 제공하지 않는다. 공식 baseline 내부를 제안 모델에 맞춰 임의 교체하지 않는 예외다.

## U_ext / G_ext의 변경과 보존

기존 두 context의 rank16 bank 각각을 **공유 rank32 경로 한 개**로 바꾸고, 외부 정적 정보의 공통8차원 projection을 추가한다. 이 변화는 외부 입력 일반화를 위한 이식이며 기존 모델을 그대로 동결한 실험이 아니다. 이전 가중치를 재사용하지 않는다. 과거 category/type을 인코더에 반영하고 현재 보조 범주는 gap→mark→amount 다음에 생성한다. 현재 gap에 따라 비반복 후보 사이의 상대확률을 바꾸는 경로나 추가 규제/생성 손실은 넣지 않는다.

U는 copy+fresh 혼합, G는 관측 반복+비반복 분해라는 차이를 유지한다. 같은 데이터별 파라미터 수(Berka 125,461; Sparkov 265,438), 초기 tensor, epoch별 순서를 제공한다. 각 target은 최근31거래를 보고, 긴 거래열 전체의 모든 target을 한 번씩 학습한다. 추론도 같은 최근31거래를 재계산한다. 첫 거래에 gap 손실 없음. 나머지는 gap/mark/amount/보조 범주 NLL의 성분별 평균 합이다. 균형/rollout 손실 없음.

훈련 seed20260925 하나, Adam lr .001, batch512, gradient clip1, 최대30epoch/20분, check base loss 최소 checkpoint, patience5. 최종 epoch도 보관한다. 한 seed이므로 학습 재현성/유의한 우수성을 주장하지 않는다.

선택 checkpoint마다 raw와 동일한 gap별 반복확률 보정도 둔다. fit에서 실제 이력을 넣은 관측 반복확률의 logit에 gap-bin 상수6개(0gap+양수5구간)를 더한다. 단일 pooled 보정이며 가짜 두 집단을 만들지 않는다. 평균 Bernoulli NLL + .001×계수제곱합, 경계[-5,5], 최대100회 L-BFGS-B. 모델 가중치는 동결한다. 비교 지표를 보고 계수/구간을 추가하지 않는다. 첫 거래는 보정하지 않고 비반복 mark 상대확률은 유지한다. 보정된 생성 행동을 다음 이력에 실제 반영한다. Berka의 원래 정렬에 fit되는 보정이라는 순서 한계도 공개한다.

## 외부/단순 대조군

- ARGN: 공식 mostlyai-engine2.4.0, Medium, 최대64epoch/20분, batch64, sequence_window32, value_protection=False, flexible_generation=False, 공식 check 선택. 설치된 공식 API 그대로 split/analyze/encode/train/generate.
- CPAR: SDV1.38.0/DeepEcho0.8.1, 64epoch, sample_size1, 공식 segment_size32로 모든 거래를 segment에 포함한다. 단일 full-batch API 특성의 메모리 비용을 기록한다. 30분 시간 상한. 이전에 검증한 pinned source의 **수치적으로 동등한 vectorized loss**만 실행 가속에 사용한다. 공식 continuous-target alignment/normalization을 조용히 수정하지 않는다. 기본 최종 epoch 선택이며 U/G와 같은 validation checkpoint API라고 하지 않는다. 자원 오류는 실패 기록으로 남기며 모델 성능으로 순위를 매기지 않는다.
- 단순 모델 두 개: (1) 주변 root 빈도, (2) 직전 root×gap의 전이 빈도. Sparkov root=category, Berka root=operation. 그 다음 merchant/type은 root 조건 빈도, amount는 root/merchant(type) 관측값에서 재표본한다. 전체 거래열 복사는 하지 않는다. 평활화20/200/2000 중 fit→check의 공동 범주 NLL로 하나만 선택. root 전이·emission을 통한 단순 정보 공유가 가맹점별 희소 표의 약점을 얼마나 해결하는지 본다.
- 외부 모델끼리 epoch 숫자를 동일한 최적화 기회라고 주장하지 않는다. 실제 epoch/update/시간/loss 추이/장치/파라미터를 공개하고 부족하면 우수성 주장에 쓰지 않는다. 결과 후 자동 연장·재학습 없음.

## 생성과 평가

fit에서 뽑은 공통128개 context 계획, 생성 난수20261021/20261022. U/G/단순/CPAR는 같은 fit 기반 길이 계획으로 **전체 길이** 생성. U/G의 길이 정책은 경험적 길이 모듈이며 길이를 예측 입력으로 주지는 않는다. CPAR는 공식 length 인자를 이용한다. ARGN은 같은 context를 주되 공식 자연 길이 생성 유지. 길이를 잘라 맞추거나 관계/금액을 사후 수리하지 않는다. **ARGN까지 실현 길이를 완전히 맞춘 비교는 아니며** 길이 지표/실현 거래 수를 함께 공개한다. 그 차이를 숨긴 모델 구조의 인과 기여를 주장하지 않는다.

평가 변환은 fit만으로 고정한다. primary 2개씩:

- Berka: 이전 비어 있지 않은 날짜와의 간격×현재 거래 종류의 공동분포 TV(날짜 내 순서 불변), 거래 종류×금액 구간 공동분포 TV. 일별 거래 수 관계도 보조로 보고한다.
- Sparkov: gap bin×직전 category×현재 category 공동분포 TV, category×금액 구간 공동분포 TV. merchant×금액은 보조.

gap은0별도+fit 양수20/40/60/80분위5개, amount는fit log1p 10분위. TV는 reference/generated 경험분포의 union support에서 .5×절대차합이며 oracle TV가 아니다. 기본 gap/mark/category TV, amount KS, 길이 KS, 미등록/음수/비정수 값 비율을 함께 공개한다. Berka 일별 집계는 calendar floor이며 비정수 생성 gap을 별도 보고한다. 이를 조용히 유효한 일 단위 생성으로 처리하지 않는다. 전이 평가의 명확한 순서 subset coverage도 공개한다.

실제 이력 예측은 U/G에 대해 관측 gap의 mark NLL, 반복 Brier, log-amount MAE, 조건부 category/type NLL을 보고한다. 인코딩/분해가 다른 외부 모델의 raw loss를 직접 순위화하지 않는다. 생성 난수2회 평균과 범위를 보고하며 이를 학습 seed2개로 부르지 않는다. 전체 주/보조 지표를 공개하고 개선 항목만 골라 성공 선언하지 않는다. 이 파일럿에는 모델 채택 문턱이나 단일 승자 규칙을 만들지 않는다.

## 종료와 보관

구현·CPU/실제자료 검증·고정 파일럿·결과/제약 보고로 종료한다. 새 구조/계수 탐색은 자동 실행하지 않는다. 분포와 관계가 함께 좋아진 후보가 있다면 이후 독립 시드의 확증 설계를 정한다. 실패해도 원인 위치와 확인되지 않은 가설을 구분한다. 모든 기존 실패는 유지하고 기존 branch에 코드·등록·집계·그림을 저장한다. 원본/ID/checkpoint는 작업공간에만 보관한다.
