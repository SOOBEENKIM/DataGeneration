# 개인 속도 진행과 반복 생성 보정 — 2026-09-28

## 근거와 구별할 가설

고정 GMR은 개인 onset gap .1168 / 전체 사기 gap .0998을 얻지만 정상 gap
.1373으로 phase GMM .0743, 일반 history .0930보다 나쁘다. 같은 실제 prefix와
labels의 clock 개입에서 GMR gap .0822→.2883으로 반복 입력 비용을 확인했다.
장기 고객의 초반/후반 속도 변화도 원본과 달랐다. 출력 하나만 변경한다.

최근접 방법은 Calinon2016 §5.1 식13–16의 Gaussian-mixture regression이다.
진행 시점 입력도 일반 조건부 회귀의 적용이며 신규 방법으로 주장하지 않는다.
2026 Lin et al. `Preserving Temporal Dynamics in Time Series Generation`
v1 §III-B Proposition2는 조건부 출력이 같아도 입력 분포가 바뀌면 주변 분포가
달라지는 현상을 다룬다. §IV의 Algorithm1은 실제 시점 값 s_t를 이용하므로
우리 자유 생성에 그대로 적용하지 않는다. 이 논문의 MH 보정, 정상성 보장을
구현/검증했다고 하지 않는다. §III-B Proposition1의 일반적인 W1 하한도
본 연구의 근거로 사용하지 않는다.

- https://publications.idiap.ch/downloads/papers/2016/Calinon_JIST_2015.pdf
- https://arxiv.org/html/2604.27182v1
- https://arxiv.org/html/2609.21382v1 (MoTPP §4–5: 일반 조건부 시간 mixture)

가설: 진행 시점 추가만으로 부족하다면, 실제 clock을 입력한 1-step 보정과
생성 gap을 clock에 다시 넣는 보정의 차이가 남는가? 보정/분포 맞춤 자체는
알려진 방법이다. 검증할 추가 효과는 같은 출력/두 파라미터에서 입력 이력의
출처만 바꾼 학습이 자유 생성의 주변-개인 관계 충돌을 줄이는지다.

## 고정 비교와 새 세 팔

기존 onset_fit, phase GMM, history_only, clock_joint(GMR)은 그대로 보존한다.
새 팔은 다음과 같다. 원본 ARGN encoder/label/count/category/amount는 동결한다.

1. `progress_joint`: (clock20, log(1+event_index), log(1+gap))의 전환별 3성분
   full-covariance GMM. 기존 GMR에 진행 시점만 추가한 단순 대조.
2. `progress_prefix`: 같은 GMM의 정상→정상 log-gap 위치에
   delta(n)=a+b*exp(-n/50)를 더한다. 실제 strict-past clock으로 보정 계수를 학습.
3. `progress_rollout`: 같은 두 계수와 목적함수로, 생성 clock의 반복 입력까지
   포함해 학습한다. 자유 생성에서는 모든 팔이 생성 과거만 사용한다.

출력은 하나의 조건부 mixture이며 혼합 전문가/사후 원본 행 복사/미래 clock은
없다. 사기 구간 출력에는 보정을 직접 적용하지 않지만 clock을 통해 간접 영향은
가능하므로 사기/개인 관계를 반드시 함께 평가한다. 첫 gap은 native 값을 유지한다.

GMM 설정은 기존과 동일: 3성분, n_init3, reg_covar1e-4, random_state20261115,
전환당 optimization 최대50000점. 미래 길이·종료 시점은 새 입력에 넣지 않는다.
GMM 4개를 한 번 적합하며 부모별 독립 재학습으로 세지 않는다.

## 보정 학습과 데이터 격리

optimization619고객 각각 처음 최대1024거래만 사용한다. 첫 gap은 무시하고 실제
label 경로를 외생 조건으로 고정한 시간 전용 시뮬레이션이다. 전체 거래열 자유
생성 성능으로 해석하지 않는다. 원본/생성 모두 같은 고객·길이·label 지원집합이다.
노이즈 seed20261201의 common random numbers를 모든 계수에 재사용한다.
최종 test는 읽지 않는다. internal_validation은 적합 이후 진단에만 사용하고
development 출력/점수는 계수 선택에 사용하지 않는다.

최소화 목적: 0.5*(전체 정상 log-gap W1 + 정상 진행 구간별 W1 평균)
+ 0.25*정상 log-clock W1
+ 4*max(정상 개인 gap W1 - 원래 GMR 같은 조건 W1 - .01,0)
+ 4*max(onset 개인 gap W1 - 원래 GMR 같은 조건 W1 - .01,0)
+ 4*max(전체 사기 gap W1 - 원래 GMR 같은 조건 W1 - .02,0).
진행 구간은 [1,50),[50,200),[200,1024). 개인 비율은 실제 평가와 같이
log1p(gap/past_median), 최소5개 과거 gap 이후만 포함한다. 개인 비율 평가의
median은 원래처럼 cap이 없다. 모델 입력 clock은 기존20개 median/7일 cap/fallback과
같다. 실제 prefix 팔의 사용 clock 분포항은 상수다.

Powell, initial(0,0), a∈[-.35,.35], b∈[-1,1], maxfev48, xtol=ftol=.001.
평가한 점과 목적값을 전부 보존하고 optimization 목적 최저점(0 포함)을 선택한다.
한 점이 잘 나올 때까지 예산/범위/목적을 변경하지 않는다. full rollout internal
진단은 별도 seed20261202,20261203. 학습 성공과 최종 채택을 구분한다.

## 자유 생성과 판정

2 native 부모(20260930,20261001)×4draw(20261011,12,21,22)×3팔=24새 전체 생성.
기존 matched GPU 결과와 비교. 4개 GPU 중 실제 유휴 장치를 두 번 확인해 사용.
동일 부모/draw의 고객 길이와 모든 label이 고정 후보와 정확히 같아야 한다.

두 부모 각각에서 정상 gap이 GMR보다 개선되고 history_only보다 낮아야 한다.
GMR 대비 개인 onset/정상 개인 gap 증가는 각각≤.01, 전체 사기 gap≤.02,
개인 사기 금액/정상 금액/정상 merchant TV≤.02, merchant 다양성 감소≤5%.
기존 phase GMM 대비 정상/사기 gap 증가≤.03도 유지한다.
rollout 방법 추가 효과는 progress_joint와 progress_prefix보다 정상 gap을 줄이고
위 보존 조건을 모두 만족할 때만 지지한다. 통계적 유의성/비열등성 검정은 아니다.
실패하면 비용을 기록하고 기존 후보를 유지한다. 최신 전체 생성기 우위나 신규성은
이 실험 하나로 확정하지 않는다.

## 실행 검증

다변량 조건부 밀도를 독립 SciPy joint/marginal 식과 대조한다. 벡터 feature와
재귀 clock의 완전 일치, 조건부 NumPy 시뮬레이터와 torch 밀도 일치,
native 첫 gap/엄격한 checkpoint 로딩/실제 배치 축소·재정렬/routing을 검사한다.
raw 생성 지표 독립 재계산, labels/lengths/weights/hash 보존을 확인한다.
source/protocol/선행 manifest는 적합 전에 동결하고 실패 실행도 삭제하지 않는다.
