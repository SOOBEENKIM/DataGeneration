# TabularARGN 기반 관계 경로의 제한된 파일럿

기준 commit `5f0bfc251288333f8988306799fa0277887ca21d`. 새 branch
`research/argn-relation-pilot-v1`. CS-SAF 소스·과거 결과·공식 설치 파일은 수정하지 않는다.
이 문서와 설정 및 CPU 검증을 commit한 뒤 과학적 학습을 시작한다.

## 질문과 근거

이전 외부 비교에서 Sparkov의 간격–업종 전이 TV는 공식 ARGN .1962,
관측 전이 대조군 .0774였다. 업종 반복률이 간격에 따라 달라지는 관측 관계도
fit/check/validation에서 확인했다. 이것은 이 설정에서 남은 오류이며
ARGN 계열의 불가피한 한계나 현재 신경망 오류의 유일한 원인이라는 뜻은 아니다.

질문은 **공식 생성기에 현재 간격×직전 업종의 직접 경로를 넣으면 일반 경로
추가보다 생성 관계를 더 개선하는가**다. Sparkov 한 자료, 학습 seed 하나,
세 모델 fit과 모델별 생성 seed 두 개로 끝낸다. 새 데이터 독립 확인은 아니다.
이득이 없어도 순위에 맞춰 rank·손실·계수·열 순서·epoch·seed를 추가하지 않는다.

## 세 조건

- A: 공식 mostlyai-engine 2.4.0 Medium 순차 모델의 새 fit.
- G: A의 업종 logit에 `W tanh(e_previous+e_gap)+b`를 추가한 일반 MLP 경로.
- R: A의 업종 logit에 `W(tanh(e_previous)*tanh(e_gap))+b`를 추가한 곱 상호작용 경로.

e는 각각 직전 업종과 현재 native gap code의 rank16 embedding이다. G는
두 one-hot 입력을 연결한 bias 없는 선형층+Tanh와 동등하다. G/R은 같은
정보·동일 추가 파라미터 수·동일 추가 초기 tensor를 가진다. output W,b는 0으로
시작해 세 조건의 초기 예측/기본 가중치가 같다. 첫 거래는 추가 경로를 0으로
둔다. 직전 업종은 관측/생성된 실제 category이며 merchant에서 강제 복원하지 않는다.

현재 gap은 native 103개 code, category는 rare token을 포함한15개다.
추가 파라미터는 `(103+15)*16+16*15+15=2,143`개다.
나머지 모든 모델 가중치를 함께 학습한다. 인코더까지 고정한 사후 보정이 아니다.
기존 CS-SAF 규제, 균형 보조 손실, rollout 손실, 정답 생성 법칙을 넣지 않는다.
원래 필드 순서 gap→merchant→amount→category와 모든 출력 codec을 유지한다.
생성된 category는 native recurrent state와 다음 직접 경로에 모두 입력한다.

이것은 저랭크 상호작용 자체의 신규성 주장도, 희소 집단 선택성을 확증하는 실험도
아니다. 원래 CS-SAF의 active/null 정답은 외부 자료에 없다. 확인할 수 있는 범위는
이 관계 경로의 추가 가치와 부작용이다. 전체 모델 새 이름/논문 기여 확정은 보류한다.

## 동일 조건과 데이터 경계

기존 external_port_v1 Sparkov의 fit550/check138/validation147개체와 128개체
생성 고객 계획을 사용한다. 기존 공식 ARGN의 OriginalData를 **복사**하여 세 조건에
동일 codec/부호화 train/check를 제공한다. 기존 checkpoint는 시작점으로 사용하지
않고 모두 같은 seed로 처음부터 학습한다. 과거 실행과 직접 차감해 추가 경로
효과라고 하지 않고 이번 A/G/R끼리 비교한다.

공식 분석 통계는 fit+check에서 추정한 기존 동작이다. 이를 fit-only codec이라고
부르지 않는다. 파라미터 학습은 fit, checkpoint 선택은 native check 전체 손실,
최종 평가는 기존 외부 validation이다. test/fraudTest/사기 라벨을 읽지 않는다.
새 fit seed로 새 데이터 독립 검증을 했다는 주장도 하지 않는다.

학습 예산은 64epoch/20분, batch64/window32, 공식 optimizer·학습률·early-stop·
최저 check-loss 선택을 그대로 둔다. 모든 조건에 같은 예산을 주며 실제 update
수/마지막 epoch/선택 epoch/곡선을 공개한다. 상한 도달은 수렴 증거가 아니다.
사용 가능한 GPU를 각 실행 직전에 확인하고 다른 작업은 종료하지 않는다.

## 실행 전 판정 기준

R의 가능성 판정은 다음을 모두 만족해야 한다. 완벽한 동시 개선을 요구하는 것은
아니며 아래 허용치는 제한된 엔지니어링 screen이지 금융 실용성의 입증이 아니다.

1. 주 지표 간격×직전업종×현재업종 joint TV의 생성2회 평균이 A와 G 각각보다
   5% 이상 작고, 각 생성 seed에서도 두 모델보다 작아야 한다.
2. gap/업종/merchant 주변 TV, 업종–금액/merchant–금액/merchant–업종 joint TV,
   amount KS, length KS의 평균 악화가 A 대비 각각 +.01 이내여야 한다.
   특히 merchant와 맞지 않는 category를 생성해 주 지표만 개선하는 것을 감시한다.
3. 실제 validation 이력의 업종/merchant/금액 native code NLL이 A 대비 각각
   1% 이내의 비용이어야 한다. 동일 codec 사이의 CE 비교이며 연속 금액 밀도나
   raw 금액 MAE로 설명하지 않는다. 첫/종료/padding은 별도로 mask한다.
4. unknown 행동/업종, invalid 금액/간격 비율은 0이어야 한다.

G만 좋아지면 일반 경로의 효용으로 해석하고 R의 추가 기여를 주장하지 않는다.
R이 A만 이기면 구조의 추가 기여 screen은 실패다. 모두 실패하면 이 후보를 종료한다.
통과해도 seed 반복·독립자료·동일 보정 기회·다른 전이 법칙·탐지 효용은 미검증이다.
CS-SAF 이전 실패는 이번 결과와 무관하게 보존한다.

## 검증과 산출물

CPU에서 무경로/native 동등성, 초기 기본 tensor 동등성, 현재·미래 정답 비누출,
G/R 동일 파라미터/정보, reload, 학습/생성 경로 일치와 생성 category 피드백을 검사한다.
실제 생성은 공식 길이 모델을 유지하고 각 조건의 실현 길이도 보고한다.
같은 seed여도 가변 길이 때문에 이벤트별 공통 난수 coupling은 보장하지 않는다.
최종 생성 지표는 별도 histogram 계산으로 검산하고 원 checkpoint/소스 checksum을
비교한다. Git에는 코드·등록·환경·집계·그림만 넣고 개체별 원자료나 checkpoint를
올리지 않는다. main과 기존 CS-SAF branch에는 병합하지 않는다.
