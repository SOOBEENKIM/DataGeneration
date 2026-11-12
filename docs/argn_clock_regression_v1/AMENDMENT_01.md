# 검증 fixture의 정밀도 수정 — 생성 실행 전

첫 독립 밀도 비교는 double head에 float32로 만들어진0.3/0.7 값을 대입하면서,
SciPy에는 double0.3/0.7을 전달했다. 1e-8 허용오차에서 최대3.87e-8의 차이로
실패했다. 조건부 식을 바꾸거나 허용오차를 넓히지 않고 fixture의 가중치 생성만
명시적으로 float64로 수정한다.

CPU GMM 적합 manifest가 먼저 기록된 상태였으므로 원래 manifest/모델/소스 및
원래 test 내용을 보존한다. AMENDMENT_01.json은 원래/수정 test hash와 보존 위치,
그 한 파일만 예외로 허용하는 별도 launcher hash를 기록한다. 모델·학습 자료·통계
적합 결과는 변경하지 않는다. 수정 시험을 통과한 뒤에만 GPU 생성으로 진행한다.
