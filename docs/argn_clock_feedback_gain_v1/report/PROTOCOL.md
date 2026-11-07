# 초기 출력을 보존하는 GMR 반응 강도/보정 강도 대조 — 사전 등록

2026-09-28 사용자 승인 후 등록. 기준은 초기 수정 init_only다. 활성 clock_prefix와 직전 personal_six, init_handoff도 저장 결과로 비교한다. 기존 가중치/출력은 보존하고 최종 test를 열지 않는다. 개선 없는 결과를 push하거나 자동 교체하지 않는다.

## 관측 근거와 선행

직전 중반 진단에서 GMR clock만 실제 과거로 바꾸면 정상 유지100–199 W1 .152997→.054459, 보정 clock만 교체하면 .168459였다. 보정 제거는 저장 prefix에서 .172234→.055980이지만 반복 재생에서는 .298989로 악화했다. 입력에 따라 필요한 보정이 달라진다. oracle은 생성 방법으로 사용하지 않는다.

[F-LLM2026 v1 §4.2–4.3/5.1–5.2](https://arxiv.org/html/2602.12756v1)를 재확인했다. 출력의 입력 민감도와 feedback gain, 보정된 출력을 다시 입력하며 학습하는 발상은 선행한다. 이 논문은 frozen LLM의 예측/추정 residual 제어이며, 우리는 거래 간격의 확률분포를 평가한다. 그 observer나 Lipschitz 제한을 구현하는 것이 아니고, 수축/안정성 정리의 가정도 확인하지 않았으므로 보장을 가져오지 않는다.

[Calinon2016 §5.1](https://publications.idiap.ch/downloads/papers/2016/Calinon_JIST_2015.pdf)의 GMR 조건부 Gaussian 평균/공분산 식을 사용한다. GMR와 회귀계수 조절은 알려진 방법이다. 이번은 선행 개념을 현재 오류에 적용한 작고 검증 가능한 대조이며 신규 방법 확정이 아니다. [Lin2026 §III-B](https://arxiv.org/html/2604.27182v1)의 입력 분포/조건부 법칙 구분도 유지한다.

## 변경 수식과 비교

현재 GMR 성분의 평균은 mu_y,k + beta_k(c−mu_x,k) + delta(c,n)이며, c는 생성된 strict-past20-gap median의 log1p다. 초기 normal-stay index1–5 mixture는 그대로다.

새 평균에 추가하는 값은

    clip(w(n) * [-a * beta_k(c−mu_x,k) + (g−1)*delta(c,n) + b], −.35, .35)

이다. 정상 유지에만 적용한다. w는 log1p 위치의50/125/350에 값0/1/0을 갖는 hat이다. n≤50 및n≥350은0. 위치들은 기존 진단 구간/보정의 중간 위치에 근거한 설계 선택이며 선행이 최적값이라고 정한 값이 아니다. 초기6–49는 같은 조건부 출력이 정확히 유지된다. 변경 범위 밖의 생성 과거는 바뀔 수 있다.

GMR의 mixture 확률과 조건부 분산은 같은 입력에서 유지하며, **성분 평균의 clock 회귀 항에 대한 반응만** 줄인다. 입력 clock 자체나 과거 계산은 바꾸지 않는다. 따라서 실제 clock 교체의 모든 효과를 구현한 것도 아니고 전체 mixture의 수축을 보장한 것도 아니다. 모든 팔의 추가 평균 변화는같이±.35로 제한한다.

| 팔 | 학습 계수 | 고정 |
|---|---|---|
| correction_gain | g∈[0,1.5] | a=0,b=0 |
| regression_gain | a∈[0,1] | g=1,b=0 |
| joint_gain | a,g | b=0 |
| output_gain_shift | g,b∈[−.35,.35] | a=0 |

마지막 팔의 g 범위는[0,1.5], b만[−.35,.35]다. 같은2계수의 추가 출력 유연성만으로 joint 효과가 설명되는지 구분한다. 두1계수 팔은 joint의 제거 대조다. equal-parameter control이 동일한 함수 집합이라는 뜻은 아니다.

## 학습 조건과 지표

4팔 모두 zero-change 초기값(a=0,g=1,b=0), Powell, 최대192목적 평가(초기 포함),xtol/ftol.001, 같은noise20261201, 동일 optimization619고객의 전체 길이. 실제 label/length를 고정하지만 **시간 과거는 전부 생성한 값**이다. 선택은 훈련 목적 최저값. 내부69고객×2noise(20261202/03)는 진단에만 사용하며 선택/튜닝하지 않는다.

원래 고객균등 시간·clock·관계 보존 목적을 공통 사용한다. 앞선 상쇄 진단에 따라 50–199 loss를50–99/100–199로 나누되 기존 부모 그룹의 고객별 row weights와 질량을 그대로 상속한다. 기존200+ 세분화는 유지한다. 추가로100–199 정상 유지의 raw 개인 gap W1이 init_only reference+.005를 넘으면 기존 관계 항과 같은4배 hinge를 더한다. 그 외 기존 개인/onset/사기 hinge는 변경하지 않는다. 이는 네 팔에 공통인 학습 목적 변경이며, 과거 후보 대비 결과를 순수 구조 효과라고 하지 않는다.

분포 loss로 적합하며 실제 다음 gap을 생성 입력에 주거나 oracle clock을 쓰지 않는다. GMR 가중치/평균/공분산, 기존6계수/초기 mixture, native ARGN/count/label/category/amount는 동결한다. 새 native ARGN 학습0회, 작은 coefficient fit4회다. 두 기존 부모에 공유하는 새 head이므로2개의 새 head 학습 seed 반복이라고 부르지 않는다.

## 검증과 생성

- NumPy/Torch double 독립 모수 일치, zero-change identity, 확률/분산 보존, 비대상 phase/위치 불변, 추가평균 bound, scalar 제거 대조의 joint 포함관계, 과거 원본 SHA256.
- 공통 loss의 child measure가 원래50–199 row weights와 정확히 합쳐지는지 확인한다. raw 개인 gap 분모를 사용한다.
- 4팔×2부모×4기존draw =32새 전체 거래 생성. 원래 native router/strict-past/reordering 검사를각 worker에서 수행한다. 원본 labels/lengths, head/native 가중치와 출력 hashes를 검증한다.
- GPU0–3 중 외부 compute작업이 없고 memory<500MiB/util≤5인 장치를 두 번 확인해 사용한다. 우리 작업만 장치당 최대2개. CPU계수 적합과 GPU전체 생성을 구분한다.

각 부모에서 init_only 대비100–199와전체 정상 gap이 각각 최소10%/5% 감소하고4draw중3개 이상 개선하는 것을 일차 성공 조건으로 둔다. early6–49는+.005,50–99/200–499/500–1023는+.03,1024+는+.02,고객균등 정상+.01,정상 개인+.005,사기 onset개인+.01,전체사기gap/개인사기금액/정상금액/가맹점TV+.01,가맹점 다양성95% 이상을 함께 확인한다. 목표구간 개인 gap도+.005 이내여야 한다. 다른 기존 active/current 기준screen은 별도로보고한다.

joint가단독/같은2계수 출력대조보다낫지 않으면 joint 필요성을 주장하지 않는다. 개선이 나와도 최신 생성기 직접 비교·다른데이터·독립 새 head 학습seed·최종test는별도과제로남는다. 결과를보고 경계/가중치/범위/평가기준을변경하지 않는다.
