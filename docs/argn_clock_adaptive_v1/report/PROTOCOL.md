# 위치·개인 속도 보정: 실행 전 고정 조건 (2026-09-28)

사용자가 승인한 세 대조를 수행한다. 직전 clock_prefix, HEAD와 실패 기록은 보존한다.
모든 새 파일은 이 ignored artifact 안에만 둔다. 최종 test는 읽지 않는다.

## 근거와 선행 중복

직전 전체길이 대조에서 1024 이후 정상 W1 .33332→.15276, 전체 정상 .08397→.11088였다.
같은 고객 cohort에서는 중간 개선도 있어 전체 고객의 보편적 trade-off가 아니다.
두 계수 delta=a+b exp(-n/50)는 n=500 이후 사실상 상수다. 기존 GMR은 이미 개인
strict-past clock을 쓰지만 추가 delta는 쓰지 않는다. 단순 더 긴 최적화는85회 수렴,
목적값0.142% 개선만 있었다. 이 근거는 ../argn_clock_recheck_v1에 보존되어 있다.

가장 가까운 기반은 Calinon2016 GMR §5.1이다. 보정 spline, customer balancing,
조건부 평균 이동은 알려진 방법이며 신규 발명으로 주장하지 않는다.
[Zheng/Kottas/Sansó JASA2026](https://doi.org/10.1080/01621459.2026.2657051)의 방법은
[공개v2 §2.1–2.3](https://arxiv.org/html/2407.03774v2)에서 확인했다. duration marginal과
lag 의존성을 함께 설계한다는 목표는 겹친다. stationary marginal 보장은 초기분포와
성분 marginal 조건을 필요로 하며 우리20-median/phase/nonstationary 설정에는 전용되지 않는다.
[OMPB UAI2026 §4.1–4.3](https://proceedings.mlr.press/v337/huang26b.html)의 frozen backbone에
작은 residual을 얹는 원칙도 알려져 있다. 여기서는 실제 target feedback/온라인 Bayesian update,
PAC-Bayes 보장을 사용하지 않는다. 이번은 특정 데이터의 원인 대조와 단일 시간 출력 개발이다.
최신 생성기 전체 대비 우위, 보장, 기여 확정은 이 네 대조만으로 주장하지 않는다.

## 변경과 네 팔

원래3성분/4phase GMR buffers, ARGN checkpoints, category/amount/label/count 모두 고정.
NN에만 location delta를 더하고 공통으로 [-.35,.35] log-gap에 제한한다. 그 외 phase의
직접 출력은 동일하며, 생성된 과거 변화에 의한 간접 변화는 별도 평가한다. 첫 gap도 기존 그대로.
위치는 과거 생성 거래 수 n, 속도 c는 직전20개(최소5개)의 gap median log1p다.
미래 거래열 길이, 고객ID별 전용 계수, 실제 미래 gap을 생성 조건으로 쓰지 않는다.

|팔|학습 measure|보정식/자유 계수|
|---|---|---|
|event_two|거래 균등|clip(a+b exp(-n/50)), 2개|
|customer_two|고객 균등|동일식,2개|
|position_six|고객 균등|log1p(n)의6 knot 선형 spline,6개|
|personal_six|고객 균등|log1p(n)의3 knot × c의2 endpoint tensor interpolation,6개|

고객 균등은 각 손실 marginal/위치집합 안의 고객별 행 가중치1/N_customer_in_group를
실제·생성 양쪽에 적용한 정확한 weighted W1이다. 단순히 고객별 W1 평균이 아니다.
빈 고객은 해당 marginal에 포함하지 않는다. event_two/customer_two는 가중치만 다르다.
마지막 두 팔은 동일6계수, 같은손실/학습지원/예산/출력제한이다. position knots는
25,125,350,750,1500,2500, personal progress knots는25,350,2500으로 고정한다.
clock endpoints는 optimization NN valid-clock의 고객균등10/90% 분위수만으로 결정.
clock이 유효하지 않으면 중간 endpoint 혼합(.5), 범위 밖은 경계값으로 제한한다.
**6개 계수를 위치 해상도에 쓸지 위치×속도에 나눌지의 비교**다. 동일한 위치함수에
clock 한 개만 삭제한 완전 nested ablation이 아니므로 결과를 순수 입력 추가 인과효과로 확대하지 않는다.

이전 보정과 달리 공통 절대 clipping이 추가된다. event_two는 동일 전체길이 최적화
85평가의 선택값/목적값도 대조해 clipping의 실질 영향을 기록한다. 그동안 학습된 후보와
같은 위치 보정 family라는 뜻이며, 이전48회 결과와의 차이를 학습시간 단독 효과라 부르지 않는다.

## 학습과 예산

optimization619고객의 전체815664 nonfirst events, 최대위치3105. actual-prefix 조건,
기존20261201 noise의 첫1024 및 tail 확장 방식을 그대로 사용한다.
목적은 기존과 동일 .5*(전체NN W1 + 1–49/50–199/200+ W1평균),
plus 기존 GMR 대비 personal .01/onset .01/fraud .02 허용 후4배 hinge.
actual-prefix에서 clock 변화0이며 onset/fraud 직접 출력은 고정이다. 별도 conditional loss,
rollout 학습, 개발세트 tuning, 여러 모델 혼합은 이번에 추가하지 않는다.
두계수 bounds a±.35,b±1; spline 계수±.35. 전부 zero 초기점, Powell xtol/ftol .001,
최대1536 function evaluations. 중간 최선점을 선택하고 수렴/상한종료를 구별한다.
0번째 포함 all curves 저장. 동일 함수 반복은 memoize하되 평가횟수 기록한다.
internal69고객×2noise는 진단만 하고 계수/구조 선택에 사용하지 않는다.

## 검증 및 실행

weighted W1을 scipy와 대조, 고객가중치 동일 질량 확인, NumPy/Torch basis와 output 대조,
zero 보정 GMR 동등성, NN 외 불변, history 미래누출/재정렬/종료 검증,
원래buffers/HEAD/모든 입력source SHA 검증 후2기존부모×4같은draw×4팔=32개 자유생성.
2부모는 기존 native checkpoints이며32독립학습이 아니다. 새 시간보정 fit4회.
비어있는 GPU만 사용한다. 생성 label/length는 같은draw의 기존실행과 완전히 일치해야 한다.
샘플의 원시W1 재계산과 금액/merchant/다양성/개인관계도 함께 평가한다.

## 실행 전에 정한 해석/보존 기준

주목표: 기존 clock_prefix 대비 전체 정상 gap과1024+ 정상 gap을 함께 줄이는가.
각 부모에서 전체정상 W1 5%이상 감소와4draw중3이상 개선, late W1 악화없음을 요구한다.
고객균등 전체 정상 W1도 부모별 악화없음. 사기onset 개인 +.01, 정상 개인 +.005,
사기gap/개인사기금액/정상금액/정상merchantTV 각각+.01, merchant다양성95%이상을
비교 비용으로 둔다. matched 6–49/50–199/200–499/500–1023 오류는 +.03 이내.
이는 이번 자동교체 보수 기준이며 논문 작성 가능성의 필요충분조건이 아니다.
personal_six의 추가효과는 position_six/customer_two보다 주목표가 두 부모에서 좋아지고
같은 비용 범위를 유지할 때만 후보로 해석한다. 어느 한 점수의 승리를 기여로 확정하지 않는다.
실패/혼합결과면 직전 모델과HEAD 유지, commit/push 없음. 결과 및 실패 증거만 로컬 보존.
