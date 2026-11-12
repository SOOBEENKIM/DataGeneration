# 현재 refined_personal의 초반 오류 원인 진단 — 실행 전 고정

사용자가 승인한 대상은 최신 stratified_loss_v1/personal_six다. 새 학습·튜닝·모델
교체·push를 하지 않는다. 기존 개선본과 현재 후보의 source/head/full outputs를 고정한다.
전체 정상 .052838,1024+ .079196 개선은 기존 결과로 보존하며 이번에 수정하지 않는다.
초반 event_index6–49 .331035가 보존 clock_prefix .236051보다 큰 원인을 구분한다.

## 문헌과 진단의 근거

[Lin et al.2026 §III-B/IV, Algorithm1](https://arxiv.org/html/2604.27182v1)는
조건부 출력과 생성 중 조건 분포 변화를 구분한다. 이 구분을 진단에 사용한다.
논문의 실제 s_t를 참조하는 보정 알고리즘이나 일반 Wasserstein 하한을 우리 자유 생성에
도입하지 않는다. 해당 정상성/분포 가정이 현재 거래열에서 성립한다고 하지 않는다.
[He et al., EMNLP2021](https://aclanthology.org/2021.emnlp-main.415/)의 실제 prefix를
제공한 뒤 이어 생성하는 대조도 참고한다. 언어 모델의 회복 결과를 거래 모델에 전용하지 않고,
실제 과거 제공이 얼마나 개선하는지 직접 측정한다. 초기 오류를 무조건 누적오차라 가정하지 않는다.
새 모델 구조 제안이 아닌 원인 대조다. 최신 참조와 알려진 진단 방법의 역할을 구분한다.

## 코드에서 확인한 경계와 대조

현재 시간 출력은 frozen GMR(phase, strict-past20-gap median)에 보정을 더한다.
GMR 구현은 native ARGN base vector를 시간 출력 계산에 사용하지 않는다. 다른 필드는
native/기존 출력 모델을 유지한다. gap의 첫 행은 간격이 아니므로 ring에 넣지 않는다.
event_index1–5에서는 이전 유효 간격이5개 미만이어서 fallback16535초를 사용한다.
6–20에서는5–19개,21부터20개가 있다. 6–49를6–20과21–49로 나누고,
1–5,50–99,100–199도 함께 확인한다. 이 분할은 손실 변경이나 추가 학습에 쓰지 않는다.

주요 모델: clock_prefix(보존), coarse_personal(같은 모델 수정 전), refined_personal(현재).
진단 대조: clock_joint(보정 없는 동일 GMR), refined_position(현재 같은6계수 위치 모델).
모든 checkpoint의 GMR buffers가 동일한지 확인한다.

고정 label/length 경로에서 같은 uniform/normal noise로 다음 입력만 바꾼다.

|모드|GMR의 개인 속도 입력|보정의 개인 속도 입력/개입|
|---|---|---|
|RR|실제 strict-past|실제 strict-past|
|GG|자체 생성 strict-past|자체 생성 strict-past|
|RG|실제 strict-past|자체 생성 strict-past|
|GR|자체 생성 strict-past|실제 strict-past|
|seed5|첫5간격만 실제 값으로 memory 구성 후 자체 생성|그 memory를 공통 사용|
|reset21|21번 직전에 실제 과거20간격으로 memory를 한 번 교체|이후 자체 생성|
|prefix_delta_first5|GG 유지|첫1–5에서만 보존 clock_prefix의 delta를 쓰고 이후 현재 delta|

마지막 모드는 초기 보정 차이의 영향만 분리하는 대조이며 혼합 모델 채택이나 새 구조 학습이
아니다. seed5에서 실제 값을 준 구간은 성능 개선 평가에서 제외한다. reset21 이전 출력은 GG와
정확히 같아야 한다. oracle 입력은 진단에만 사용하며 자유 생성 성공 점수로 보고하지 않는다.
개입 결과를 현실 금융 인과관계의 증명으로 확대하지 않는다.

## 자료·난수·평가

- optimization619고객:2noise(20261202/03); internal69와 development147:4noise(20261202–05).
- 실제 label 경로에서 주요3모델×7모드, 대조2모델×RR/GG. 시간만 event_index1–199 재생.
- 기존2부모×4draw의 생성 label/length에서도 대조한다. current는7모드, prefix/coarse는RR/GG,
  각2noise. 실제와 생성의 공통 entity/position 지원만 사용한다. 정상 조건이 양쪽에서 일치하는
  행을 비교해 label 구성 혼동을 줄이고, 그 표본 수를 기록한다.
- 시간 head는 phase/clock/position만 쓰므로 독립 NumPy로 정확한 동일 조건부 법칙을 계산한다.
  native engine의 전체 RNG stream을 복제한 것으로 주장하지 않는다. 저장된 실제 GPU 생성
  결과의 오류를 별도로 재계산해 통제 재생과 같은 방향인지 확인한다.
- RR/GG는 같은 실제 경로·noise를 사용하므로 조건 분포의 영향을 구분할 수 있다. 원래
  전체 자료의1024-prefix noise 생성 순서를 유지한다. 모델별·모드별 randomness를 바꾸지 않는다.
- W1(log1p gap), 고객 균등 marginal W1, 부호 있는 log-gap/clock bias, 분위수, 개인 속도
  분산과 유효 이력/endpoint clip 비율, 개인 gap 관계를 기록한다. W1 차이를 가산적 인과
  기여율로 해석하지 않는다. NN과전체정상을 분리한다.
- 원래 6–49 개발 오류(같은 지원집합)를 재현한다. 주요 비교는 같은 고객에서 이루어지며,
  반복 noise를 새 독립 학습 seed라 부르지 않는다. 최종 test는 열지 않는다.

## 구현 검증과 판단

NumPy/Torch double 조건부 모수 parity, strict-past clock(6/21 경계 포함), 미래값 변경
불변, 첫 event gap 무시, 재정렬, RR의 기존 actual-prefix 출력 parity를 먼저 확인한다.
prefix/position은 correction의 clock 입력이 없으므로 RG=RR,GR=GG가 정확히 성립해야 한다.
seed5의6번째 입력과 reset21의21번째 입력은 실제 clock과 같아야 한다.

RR에서도 큰 초반 오류가 남으면 조건부 출력/초기 fallback의 부적합을 먼저 본다.
RR은 맞고GG에서 커지며 실제 과거 교체가 회복시키면 생성된 초기 상태가 중요한 원인이다.
RG/GR로 원래 GMR과새 보정의 clock 경로를 구분한다. 첫5 보정 대조가 회복시키는지도 본다.
어떤 대조도 해결하지 못하면 그 사실을 기록한다. 결과에 맞춰 새로운 학습·경계·보정식을
이번 진단에 추가하지 않는다. 다음 수정은 확인한 원인과 가까운 선행을 연결한 뒤 설계한다.
