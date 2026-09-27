# 시간 출력에서 바뀐 부분과 선행 중복

현재 객체는 ARGN 기반 거래 생성기의 시간 출력이다. customer/count/label 및
category/amount 출력의 가중치는 고정하고, 실제 생성 전체에서 생기는 간접 영향은
동일 고객·label 경로의 matched 평가로 측정한다.

원래 GMR은 전환 상태 s와 strict-past 개인 clock c에서
`z=log(1+gap)`의 조건부 혼합분포를 계산한다. 위치 보정 팔은

\[
q(z\mid c,n,s)=\sum_k w_k(c,s)\,
\mathcal N\bigl(z;\mu_k(c,s)+\mathbf1_{s=0}\delta(n),\sigma_k(c,s)^2\bigr),
\qquad \delta(n)=a+b\exp(-n/50)
\]

를 사용한다. n은 지금 생성할 거래의 index, s=0은 정상→정상이다. 이후 기존
codec 한계에서 clip하고 정수 초로 양자화한다. 혼합 가중치, 개인 clock에 대한
기존 회귀, 성분 분산은 원래 GMR과 같다. 보정이 작아도 이후의 clock이 바뀌므로
개인 사기 관계·사기 시간·금액까지 좋아진다고 가정할 수 없다.

`clock_prefix`와 `clock_rollout`의 구조와 파라미터 수는 같다. 둘 다 실제 학습
label 경로 아래의 시간 전용 시뮬레이션에서 주변/개인 관계 목적을 최소화한다.
prefix는 실제 과거에서 clock을, rollout은 생성 과거에서 clock을 계산한다.
최종 전체 거래 생성에서는 둘 다 생성 과거를 쓴다. 초기50거래 정도의 보정과
장기 수준 보정이 정확한 물리 과정이라는 가정은 하지 않는다. 제한된 두 기저의
효용과 비용을 확인하는 실험이다.

## 가까운 선행과 구별 범위

| 선행 | 읽은 범위와 겹치는 부분 | 여기서의 구별/제한 |
|---|---|---|
| [Calinon2016](https://publications.idiap.ch/downloads/papers/2016/Calinon_JIST_2015.pdf) | §5.1 식13–16: Gaussian mixture의 조건부 가중치/평균/분산 | GMR 자체, 진행 시점 조건 자체는 선행이다. 기존 GMR을 강한 기준으로 둔다. |
| [Lin et al.2026 v1](https://arxiv.org/html/2604.27182v1) | §III-B Proposition2, §IV Algorithm1, §V 설정: 조건 입력 분포 변화와 동역학 보정 | 동일 조건분포여도 입력 분포에 따라 주변분포가 달라질 수 있다는 관점은 선행이다. Algorithm1의 실제 s_t를 쓰지 않고 생성 상태에서 보정 계수를 학습한다. MCMC/그 정상성 주장을 재현하거나 새로 증명하지 않았다. |
| [MoTPP2026 v1](https://arxiv.org/html/2609.21382v1) | §4–5: history 조건부 시간 mixture/과제 평가 | 일반 조건부 시간 밀도 자체는 선행이다. Sparkov 전체 필드 직접 재현이라고 하지 않는다. 기존 history_only 출력과 비용을 함께 비교한다. |
| [이전 normalization 독해](../argn_time_density_v1/PROTOCOL.md) | 원래 clock/statistics를 조건으로 유지하는 재표현은 선행 | 이번에는 실패한 clock-relative 위치 재표현을 다시 채택하지 않았다. 기존 조건부 법칙을 보존한 작은 변경의 효과를 구분한다. |

학습 목적의 분포 맞춤, simulator calibration, 위치 보정 자체도 일반적 방법이다.
여기서 확인할 것은 같은 작은 출력에서 실제 과거와 생성 과거 학습의 추가 효과와
비용이다. reference가 같은 금융 표의 같은 희귀 조건을 평가하지 않았다는 이유로
그 모델의 실패라고 주장하지 않는다. 모든 2026 논문을 빠짐없이 검토했다거나
최신 full generator 전체보다 우수하다고 주장하지 않는다.

판정은 PROTOCOL.md와 screen.csv를 따른다. 목표 정상 오류가 줄어도 GMR의
개인 onset/normal 관계와 fraud 오류를 훼손하면 채택하지 않는다. 반대로 단순
prefix 보정이 더 낫다면 rollout 학습의 추가 기여 가설을 지지하지 않는다.
