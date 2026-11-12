# 조건부 clock 반복 입력의 분리 진단 — 2026-09-27 UTC

relative의 최초 완성 draw에서 전체 정상 gap W1이 약2.7이고 generated past-gap
중앙값이 실제17364.5초 대비 약2000초다. 상대 개인 gap은 나쁘지 않다.
원인을 단순 상관관계로 확정하지 않기 위해 다음 조건부 실험을 등록한다.

development에서525개 이상의 거래가 있는 고객만 사용한다. 실제 첫25거래로
clock을 초기화한 후 다음500거래의 **실제 라벨 순서**를 고정한다. 일반 자유
생성 점수/생성 모델의 실용 성능으로 보고하지 않는 oracle-conditioned 진단이다.
샘플 seed20261111–14를 공통 적용하고 상대 출력과 알려진 clock GMR을 각각 확인한다.

- actual_clock: 매step 실제 strict-past clock으로 다음 gap을 생성한다.
- recursive_clock: 같은 시작25거래 후 생성한 gap으로 clock을 갱신한다.

동일 label·고객·초기 이력·난수에서 clock 입력 출처만 다르다. relative 선택 head는
NN 마지막층0이며 두 부모의 출력 prior가 동일한지 먼저 확인하므로 고정 부모 하나만
사용한다. GMR도 동일 통계 적합을 사용한다. 다른 거래 필드는 이 두 선택 head에
영향하지 않는다. 따라서 이 조건부 실험 내 변화는 clock 반복 입력에 의한 효과다.
원본 전체 자유 생성 오류의 유일한 원인/ARGN 일반의 구조적 한계라고 확대하지 않는다.

누적20/100/500step에서 전체 gap log-W1, 실제 clock 대비 평균 절대 log오차,
generated/real clock 중앙값을 보고한다. 최종 test 미사용, 새 fitting 없음.
