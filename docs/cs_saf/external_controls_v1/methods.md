# 정확한 확률식과 비교 범위

이 문서는 등록된 출력 변경의 수식을 설명한다. 새 모델의 우수성이나 학습 성공을 전제하지 않는다. 간단한 설명은 [한 장 설명](model_explanation.md), 판단 기준은 [실행 전 등록](preregistration.md)에 있다.

## 행동 분포

`c`는 GRU128차원의 strict-past 요약과 정적 정보8차원이다. `e(g)`는 fit-only support로 부호화한 현재 간격의32차원 embedding이다. 과거는 최근31거래이며, 생성한 거래를 다음 창에 실제로 넣는다.

기존 fresh 분포는 `q(m)=softmax(W_c c+b)_m`다. 행동 수정에서는 다음으로 바꾼다.

```text
q(m | c,g) = softmax(W_c c + W_g e(g) + b)_m
r(c,g) = b_repeat(c) + Σ_k w_k tanh(Ac)_k tanh(Be(g))_k / √32
s(c,g) = sigmoid(r(c,g))

U: P(m | c,g,previous) = s 1[m=previous] + (1-s) q(m | c,g)
G: P(previous | c,g,previous) = s
   P(m≠previous | c,g,previous) = (1-s) q(m | c,g)/(1-q(previous | c,g))
```

첫 거래에는 U/G 모두 q만 사용한다. U의 s는 숨은 복사확률이고 실제 반복확률은 `s+(1-s)q(previous)`다. G의 s는 관측 반복확률 자체다. gap 보정도 숨은 s가 아니라 각 모델의 **관측 반복확률** logit에 적용한다.

D는 `[c,e(g),previous_embedding,has_previous]`를 MLP→softmax에 입력한다. previous embedding은 첫 거래에서0이다. 반복 전용 gate나 G의 후보 제외 연산은 없다. PAD만 출력에서 제외하고 UNK/MISSING은 scoring과 sampling에 모두 남긴다. 일반 D의 행동 head 폭은 G_both 행동 head와 파라미터 수가 가깝도록 사전 공식으로 계산한다: Berka34, Sparkov136. 결과를 보고 폭을 탐색하지 않는다.

## 금액 분포

양수 원금액을 a, 기존 history 좌표를 `z=(log(1+a)-μ_codec)/s_codec`라 한다. 새로운 양수 분포는 fit에서 정한 `x=(log(a)-μ_log)/s_log`에 Gaussian 세 개의 혼합을 둔다. 혼합 가중치와 각 성분 평균·표준편차는 이력·현재 간격·현재 행동의 함수다. fit에0이 있으면 추가로 `p0=P(a=0)`를 학습한다.

```text
P(a=0) = p0
f_a(a>0) = (1-p0)/(a s_log) × Σ_k π_k Normal(x; μ_k, σ_k²)
f_z(z) = f_a(a) × s_codec × (1+a),  a>0
```

`p0=0`인 Sparkov에서는 양수 부분만 사용한다. Berka는 fit의0을 별도 원자로 학습한다. `π=softmax(logits)`, `σ=clamp(softplus(raw_scale),.03,3)`이다. 평균은 제한하지 않는다. 표준편차 범위는 표준화한 log(a) 좌표의 값이며 원금액의 상한을 보장하지 않는다.

위 Jacobian을 loss에 포함해 양수 부분의 밀도를 기존 z좌표에서 계산한다. 0에 대해서는 질량 `-log(p0)`를 사용하므로 이전 연속 Gaussian의 밀도와 동일한 기준측도는 아니다. 각 모델 안에서 check loss로 checkpoint를 고르되, 이 native loss를 모델 간 우열 지표로 사용하지 않는다.

생성에서는 성분과 정규 표본을 뽑아 float64의 양수 a를 만들고, 0 원자가 선택되면 정확한0을 만든다. 이 raw 표본을 그대로 저장한다. 다음 이력에 들어가는 z만 float32로 만든다. 이중 표현은 음수 clipping이 아니라 codec 왕복의0 반올림 오차를 피하기 위한 것이다. 큰 금액을 자르거나 resample하지 않는다.

금액 log-MAE용 점예측은 혼합의 `E[log(1+a)]`를16점 Gauss–Hermite로 근사한 값이다. 원금액의 평균, 중앙값, 예측 밀도 점수와 구분한다. 이전 Gaussian에서는 그 z좌표의 평균을 사용한다.

## 학습과 해석

일반 likelihood 성분별 평균의 합을 유지한다. 새 보조 손실·규제·oracle 목표는 추가하지 않는다. 첫 거래에 없는 gap target은 gap 평균에서 제외한다. amount head만 교체해도 공동 이력 인코더로 흐르는 기울기가 바뀌므로, 다른 필드의 예측과 자유 생성이 달라질 수 있다. 따라서 금액 변경의 효과를 출력에서 음수만 없앤 효과로 설명하지 않는다.

현재 gap을 fresh head에 추가하면 정보 경로와 용량이 함께 바뀐다. D는 같은 입력 정보와 비슷한 전체 용량을 갖춘 일반 구조 대조군이지만, 같은 함수 집합·최적화 난이도·초기 출력 분포를 보장하지는 않는다. 공통 초기 tensor만 정확히 일치시켰다.

두 변경의 네 조합은 효과를 분리하는 탐색 비교다. 사후 보고의 `factorial_effects.csv`는 평균 차이와 조합 차이를 기술하며 모집단 인과효과나 유의성 검정이 아니다. 동일 생성 seed도 모델별 sampling 경로가 달라 필드별 난수까지 동일하게 정렬됐음을 뜻하지 않는다. 학습 seed1개·생성2개의 평균으로 통계적 일반화를 주장하지 않는다.
