# 학습 후 현재 조건과 생성 이력 차이의 분리

2026-09-27, balanced_shared 두 fit과 해당 자유 생성 일부를 본 뒤 등록한
후속 탐색 진단이다. 최초 학습의 사전등록 결과로 소급하지 않는다.
실제 조건의 개발 사기 금액 중앙값340–353과 첫 자유 생성97–157의 차이가 근거다.
현재 네 팔의 학습·생성 프로토콜은 그대로 유지한다.

가장 단순한 balanced_shared 두 checkpoint를 고정한다. 추가 학습은 없다.
같은 development1146사기+고정 추출1146정상 위치, 실제 과거와 실제 label을
항상 제공하고 현재 거래의 조건을 아래 여섯 경우로 바꿔 amount를 생성한다.

1. category/gap/merchant 모두 실제 값.
2. 세 필드 모두 native ARGN으로 생성.
3. category만 실제 값.
4. gap만 실제 값.
5. merchant만 실제 값.
6. category와 merchant는 실제 값, gap은 생성.

미래 실제 거래는 주지 않는다. 다만 기존 진단처럼 실제 길이/position을 준다.
고객 context256차원과 실제 과거 history832차원은 이미 검증한 cache에서 복원한다.
현재 amount column은 cache의 공통 입력에서 mask되어 있다. 첫 digit logits를
case1에서 캐시 출력부와 native 생성 간 대조한다. 다음 LSTM state는 버린다.
다단계 자유 생성의 대체 평가가 아니다.

sampling seed4개는 최초 학습 대조와 같고 batch768로 고정한다. 매 case/seed에서
RNG를 재설정하지만 샘플링하는 필드 수가 다르므로 동일 noise 결합을 주장하지 않는다.
label별 amount median/log-W1과 관측 category/merchant의 일치율을 기술한다.
일치율 자체는 생성기가 원본 거래를 복사해야 한다는 목표가 아니다.

case1→2 차이는 실제 이력에서도 현재 조건 생성이 만드는 분포 이동의 근거다.
case2와 완전 자유 생성 차이를 이력만의 순수 인과 효과로 계산하지 않는다
(길이, 위치, 사기 발생 고객/시점의 분포도 다르다). 일부 필드를 실제 값으로
고정한 조합은 실제 데이터 joint-support를 벗어날 수 있으며, 완전 인과식별이나
원인별 기여율로 해석하지 않는다. 변수별 복원 효과를 다음 출력 학습 선택의
진단으로만 사용한다. final test 미접근, 기존 head/부모/cache hash 유지.
