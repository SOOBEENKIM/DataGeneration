# 이번 구현의 정확한 범위

초기 시간 출력은 570개 학습 고객의 normal-stay 초기 간격 2,838개로 적합한
3성분 log1p-Gaussian mixture다. index1–5라는 경계는 기존 MIN_HISTORY=5에서 왔다.
K=3은 기존 GMR 성분 수와 같은 단순 대조를 위해 고정했다. 보정 기저의 6/50/200 경계는
직전 진단의 유효 이력 시작, 기존 초반/중반 평가 구간에 맞춘 설계 선택이며 선행논문이
최적값이라고 증명한 설정은 아니다. 별도 데이터 탐색이나 개발 결과를 통한 경계 튜닝은 없다.

초기 mixture는 개별 고객의 미래를 조건으로 삼지 않는다. 이 단순 초기 marginal 모델로
초기 clock 형성 문제를 해결할 수 있는지 먼저 구별한다. 고객 내 초기 간격의 상관이나
고객별 초기 속도 다양성이 여전히 틀리면 그것이 후속 조건부/공유 초기 상태 모델의 근거가
될 수 있지만, 이 비교를 마치기 전에 새 latent state나 diffusion을 덧붙이지 않는다.

handoff 보정 두 계수는 새 정상 시간 출력에만 작용한다. 다른 상태의 조건부 수식을
고정해도 과거 간격이 달라지면 사기 구간에 들어갈 때의 GMR 입력과 이후 native 거래
출력이 달라질 수 있다. 사기·금액 관계를 full generation에서 다시 검사하는 이유다.
200 이후 추가 보정이0인 것도 후반 생성 결과의 불변성을 뜻하지 않는다.

학습은 초기 mixture NLL 적합 후 이를 동결한 상태의 2계수 전체 재생 최적화다.
따라서 여기서 결합 팔이 좋아져도 '초기 분포와 전환을 end-to-end 공동 학습했다'고
쓰면 안 된다. 순차적으로 적합한 initial output / downstream calibration의 연결 대조다.
현재 데이터에서 필요한 상호작용인지와 논문 수준 신규성은 별개다.

수식으로는 정상 유지에서 n≤5이면 z~GMM_initial, 이후 z~GMR(clock_generated)+
clip(delta_current(position,clock_generated)+b(position)^T theta,±.35)다.
b는 [6,200)에서만 지원을 갖는다. 같은 u/normal noise로 모든 최적화 평가를 비교한다.
초기 혼합분포의 실제 학습 모수는 INITIAL_MIXTURE.json, 모수 최적화 경로는 각
learning_curve.csv에 저장한다. 개발/최종 test 결과로 이 모수를 선택하지 않는다.

선행 대응과 한계는 실행 전 고정한 PROTOCOL.md에 있다. 이후 기록을 추가하더라도
원래 source/protocol/manifest/실패 결과를 덮어쓰지 않는다.
