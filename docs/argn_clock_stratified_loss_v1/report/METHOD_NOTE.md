# 손실 세분화의 의미와 한계

200+ 부모 구간에서 고정된 원래 행 가중치를 w_i, 총합을 M이라 하자.
하위 구간 g의 질량은 M_g=sum_{i in g}w_i, alpha_g=M_g/M다.
실제/생성 조건부 empirical distribution을 P_g,Q_g로 두면
P=sum_g alpha_g P_g, Q=sum_g alpha_g Q_g다. 여기서 P와 Q는
실제 과거와 같은 위치·고객·라벨 경로를 제공한 **학습 시뮬레이션**의 분포다.

하위 구간 안의 최적 coupling을 alpha_g로 혼합하면 원래 pooled 문제에서 가능한
coupling 하나를 얻는다. 따라서 W1(P,Q) <= sum_g alpha_g W1(P_g,Q_g).
왼쪽은 위치를 넘나드는 최적 짝짓기를 허용하고 오른쪽은 같은 구간 안에서만 허용한다.
이것은 [조건부 Wasserstein, Proposition1](https://arxiv.org/html/2403.18705v3)의
이산 조건에 대한 적용이며 새 정리가 아니다.

각 행의 최종 계수는 alpha_g*(w_i/M_g)=w_i/M로 유지된다. 고객 균등 팔에서
각 하위 구간마다1/N_customer(g)를 새로 계산하면 이 등식이 깨진다. 구현은 원래
200+의1/N_customer(200+)를 상속하여 그 변경을 배제한다. 초기 두 구간과
200+ 전체의 바깥 계수도 그대로 유지한다.

전체 학습 손실에서 바뀐 항의 바깥 계수는 .5/3=1/6이다. 같은 theta에서는
L_refined-L_coarse=(conditional_200plus_W1-pooled_200plus_W1)/6가 정확히 성립한다.
IMPLEMENTATION_CHECK.json의 양 가중치×네 기존 head 대조가 이를 검사한다.

이 부등식은 더 세분화된 목적이 더 작은 값을 갖는다는 뜻이 아니다. 같은 모델에서
새 손실이 더 큰 것이 자연스럽다. 서로 다른 loss 종류의 절대 목적값을 그대로 비교하여
학습 악화를 선언하지 않는다. 두 계수 집합을 **각각 같은 coarse/refined 목적**으로 재평가한다.

세분화하면 구간을 넘는 상쇄를 제한하지만, sparse tail의 추정 잡음, head의 표현 제약,
개인 상태의 생성 중 분포 이동, 최적화의 국소해는 남을 수 있다. 실제 반복 생성에서는
위치·고객·라벨의 지원/혼합 질량도 학습 조건과 다르므로 위 부등식만으로 일반화·반복 생성
개선을 보장하지 않는다. 별도 full generation 평가가 필요한 이유다.
