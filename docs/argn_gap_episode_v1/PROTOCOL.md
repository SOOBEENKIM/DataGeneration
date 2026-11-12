# ARGN 간격·사기 전환 후속 대조 — 2026-09-27

사용자가 승인한 업종+금액 기준 / +간격 / +전환 / +둘의 비교다.
기존 protocol과 출력은 변경하지 않는다. 최종 test event는 열지 않는다.

## 오류와 가까운 방법

기준은 category_and_amount의 두 부모 fit×두 생성 draw다. 사기 금액 log-W1
평균 .2143, 개인 대비 .1371로 회복했지만 사기 간격 .9394, 사기 경험 고객
평균71/147(개발 실제114/147), 사기 전용 고객0(실제12)다. 현재 duration
전환은 사기 과거 경험, 시작 경계, 초기 상태–길이 의존성을 생략했다.

ARGN v2 §3.5–3.6 https://arxiv.org/abs/2501.12012v2 의 기존 필드 조건부
출력과 CTGAN §4.3의 조건별 표본 배분이 gap 학습의 직접 선행이다.
이번 gap 변경은 새 의존성 구조가 아니라 native gap head의 추가 조건부 학습이다.
REDSLDS(AISTATS2025) https://proceedings.mlr.press/v258/slupinski25a.html 는
상태 전환과 explicit duration을 연결한다. OmegaSDS(2026)
https://arxiv.org/abs/2605.06315 는 recurrent switching dynamics를 학습한다.
이번 전환은 latent-state 신모델이나 이들 구현 재현이 아니라 관측 label의
빈도/기간/과거 경험/초기 상태를 분리하는 알려진 통계적 대조다.
LBDTPP(2026) https://arxiv.org/abs/2606.24982 는 block event generation으로
긴 rollout을 다루지만, gap이 실제 이력에서도 틀리면 그 구조를 먼저 도입할
근거가 없다. 2026-09-27 원문 버전 페이지 재확인; 세부 내용은 기존 재점검
독해 범위에 따른다. 이 실험으로 최신 생성기 전체에 대한 우위를 주장하지 않는다.

## 고정과 gap 원인 분리

기존619/69/147고객 optimization/internal-validation/development split,
codec, 전체 과거 encoder, 두 부모, category+amount head를 고정한다.
현재 gap 이후 merchant/amount는 gap 입력에서 가려진다. 먼저 native head를
실제 이력·실제 현재 label/category에 조건화해 gap을 샘플링한다. 최초 거래는
gap 없음이므로 학습과 진단 모두에서 제외한다. native/cache 모든 token logit
일치 검증 후 natural_gap, balanced_gap을 학습한다. 각2개 seed, native head
복제만 최적화한다. sample indices와 optimizer/예산/validation criterion은 동일.

두 목적은 각각 실제 label 빈도 가중 NLL과 label별 각1/2 NLL이다. 표본은
정상1024+사기1024 replacement, Adam .001, dropout .25, clip1, 최대2000step,
100step마다 전체 내부 validation macro NLL, patience5/min_delta1e-5, step0 포함.
개발 set으로 checkpoint/팔을 고르지 않는다. 네 factorial에서는 사전 지정한
balanced_gap을 사용한다. natural_gap은 추가 학습 대조이며 조건부 결과 전부 공개.
사기 prior 자체는 바꾸지 않는다.

optimization/development의 사기 전체+같은 수의 고정 정상 event,4draw에서
실제 이력·실제 label/category의 gap W1/중앙값과 NLL을 기록한다.
development에서는 동일 실제 이력·label에 category만 학습된 기준 모델로
샘플링한 경우도 대조한다. 이후 자유 생성과 차이를 보며, 차이를 모두 과거
누적 오차라고 부르지 않는다(고객·사기 위치·생성 길이 구성도 다름).

## 강한 단순 전환 대조

1. duration: 기존 train-only previous label×run age(21 이상 pooled).
2. episode: duration에 strict-past ever-fraud와 최초 관측 label을 추가한다.
   최초 label은 첫 거래를 생성한 후에만 저장한다. 정상에서 다시 발생하는
   위험과 시작 경계의 사기 지속을 분리한다. 각 cell Jeffreys .5 smoothing,
   미관측 cell은 기존 duration으로 fallback. 강제 1episode나 강제 종료 없음.
3. joint: episode와 동일하되 첫 label은 ARGN이 먼저 샘플링한 planned length에
   조건화한다. optimization 고객 첫 거래에서 log1p(length) 1변수 depth≤2,
   min_leaf10 log-loss tree로 분할을 학습하고 leaf rate에 .5 smoothing을 한다.
   100행 threshold를 규칙으로 넣지 않는다. 길이 head는 학습·수정하지 않는다.

길이를 먼저 샘플링하는 전체 거래열 생성 p(L|context)p(y1|L)… 분해다.
실제 개발 고객 미래 길이를 생성에 제공하지 않는다. 실제 prefix teacher 진단의
관측 길이는 별도 oracle planned-length 진단으로 명시하고 online 다음 거래
예측 성능으로 부르지 않는다. 모든 rate는 optimization의 실제 노출 event로
계산하고 50:50 label 학습은 없다. 마지막 event 뒤를 종료로 세지 않고 censoring을
보존한다. 전환 조건 선택/튜닝에 개발 출력은 사용하지 않는다.

## 자유 생성 비교와 검증

기준4개 생성 재사용 + 새4팔×2fit×2draw=16개 생성.
핵심4cell: 고정category+amount / +balanced gap / +joint transition / +둘.
episode_only4draw는 joint의 초기 길이 조건을 제거한 추가 대조다.
같은 learned gap head를 두 gap 팔에서 사용한다. 같은 전환 파라미터를 두 joint
팔에서 사용한다. context, seed, original checkpoint 유지. 출력 후 relabel/비율
강제/거래 복사 없음. 동일 난수 소비로 gap-only의 label과 전체 길이 일치를 검사한다.
전환 변경에서도 길이 sampled sequence 일치를 검사한다.

전환 단독 합성 label rollout은 고정 기준이 샘플링한 길이에서 별도 seed로 많은
draw를 평가해 고객 구성 오차의 sampling noise와 남은 길이 분포 오차를 구분한다.
이는 거래 생성기 전체의 반복 학습 수를 늘리는 것이 아니다.

모든 기존 지표 + 정상/사기 gap W1, 고객 사기 경험/전용 구성, 반복 episode,
시작/지속/종료, censoring별 기간, 금액/개인 대비/고차 관계, unique merchant 및
정상 품질을 기록한다. 성공 practical screen: 두 fit평균에서 사기gap≥20% 감소,
고객 경험 및 전용 구성 절대 count오차 감소; 기존 fraud amount/relative W1 및
정상 amount/gap/merchant TV 비용은 절대+.05 이내인지 각각 표시한다.
모든 지표/실패/시드별 결과 공개, 두fit를 유의성 검정으로 포장하지 않는다.
단순 통계 모델로 해결되는 부분은 신규 기여에서 제외한다. 전체4cell이 좋아져도
개인 행동 신모듈이나 CPAR/최신 모델보다 우수함이 증명된 것은 아니다.
