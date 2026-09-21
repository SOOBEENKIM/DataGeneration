# 금액 출력의 수식과 비교 범위

현재 이력 표현 h, 생성할 현재 gap g와 mark m은 기존 D와 같다. 금액 head의
입력은 `x=[h,value_gap(g),value_mark(m)]`이다. 현재 auxiliary는 입력하지 않는다.

```text
q_k(x) = softmax(Wx+b)_k
F_k = fit 거래 중 금액이 구간 k에 속하는 사건들의 원금액 목록
P(A=a | x) = Σ_k q_k(x) · count(a in F_k) / |F_k|
L_amount = -log q_{bin(A_observed)}(x)
E[log(1+A) | x] = Σ_k q_k(x) · mean_{a in F_k} log(1+a)
P(A>c | x) = Σ_k q_k(x) · count(a>c in F_k) / |F_k|
```

pool은 고정되며 학습 파라미터가 아니다. 같은 금액이 여러 거래에서 관측되면 그
빈도만큼 pool에 남긴다. zero bin이 있으면 해당 pool은 전부 0이다. 실제 생성은
q로 bin을 뽑고, pool의 사건 인덱스를 균등하게 뽑는다. 원금액을 저장하고 기존
float32 log1p 표준화 좌표로 변환한 값을 다음 GRU 이력에 넣는다.

학습 loss의 금액 항은 **관측 구간의 확률**을 학습한다. fit 값의 raw 경험 질량과
비교하면 pool 내부 빈도항은 파라미터와 무관한 상수지만, fit에서 한 번도 관측하지
않은 정확한 금액에는 이 생성기가 질량을 주지 못한다. 따라서 check/validation의
구간 CE를 raw 금액의 likelihood나 기존 lognormal NLL과 동일시하면 안 된다.
fit 범위 밖 check 값도 경계 구간으로 부호화되어 bin CE에는 들어가지만 실제로
그 원금액을 생성할 수 있게 되는 것은 아니다. 지원 범위 한계를 별도로 기록한다.

표본추출 pool·경계·기대 log 금액은 fit만으로 계산하며 check는 epoch 선택,
validation은 최종 평가에만 사용한다. 테스트 split의 outcome은 읽지 않는다.
금액의 직접 조건에는 과거와 현재 gap/mark만 들어가고, Sparkov category 등 현재
보조 필드의 정답을 미리 사용하지 않는다.

이번 변경은 head 출력 수, 확률 분포, 생성 방식, 금액 손실 표현을 함께 바꾸는
일반 대조군이다. 모델 전체를 같은 공통 초기값에서 공동 학습하므로 다른 head의
학습 결과가 달라질 수 있다. 용량 효과나 손실 gradient 크기 효과를 개별적으로
식별한 실험으로 해석하지 않는다. 새 규제/보정/생성 손실은 없다.

주 지표는 기존과 같은 행동–금액 TV 및 자료별 시간–행동 TV이며 작은 비용을
허용한다. 상위 금액 보존을 위해 fit q99/q99.9 초과율과 root별 초과 질량의 L1
오차를 추가로 등록했다. root별 L1은 전체 거래수로 나눈 `P(root,A>cut)`를
사용한다. tail 내부에서 다시 정규화한 조건부 TV와 다르다.

한 학습 seed와 두 생성 난수로 실행했다. 같은 난수 seed라도 모델의 sampling
연산 수가 달라 개별 필드 난수까지 대응되는 것은 아니다. 반복되는 학습이나
독립 데이터 일반화 검증은 수행하지 않았다. 이번 판단은 고정한 탐색 기준이며
통계적 유의성 또는 논문 채택 가능성을 판정하는 기준이 아니다.
