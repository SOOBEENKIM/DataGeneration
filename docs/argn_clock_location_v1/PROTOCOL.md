# 기존 GMR 관계를 유지하는 위치 보정 대조 — 2026-09-28

argn_clock_evolution_v1의 내부 검증만 보고 추가한 대조다. 기존 GPU 생성은
진행 중이며 새 자유 생성 development 결과는 아직 비교/열람하지 않았다.
당시 internal의 실제 clock 정상 개인 gap W1은 기존 GMR .01570, 진행 GMR
.03326이다. 진행 조건을 넣으며 mixture 전체를 다시 적합한 변경 자체에도
비용이 있다. 따라서 기존 GMR의 성분 확률/조건 회귀/분산을 그대로 유지하고
같은 두 위치 계수만 학습하는 두 팔을 구분한다. 기존3팔은 그대로 끝까지 평가한다.

가장 가까운 방법과 입력 분포 이동 근거는 이전 프로토콜의 Calinon2016,
Lin et al.2026, MoTPP2026 그대로다. 새 이론/새 GMR을 주장하지 않는다.
추가 검증은 '진행 GMM 전체 재적합 없이 생성 과거를 고려한 위치 학습만으로
기존 개인 관계를 유지하며 전체 시간 비용을 줄일 수 있는가'이다.

- `clock_prefix`: 기존 GMR + 정상→정상 delta(n)=a+b*exp(-n/50), 실제 과거로 학습.
- `clock_rollout`: 같은 기존 GMR/두 계수를 생성 clock 반복 입력에서 학습.

argn_clock_evolution_v1/PROTOCOL.md의 optimization 고객/처음1024지원집합,
목적함수/정상·사기·개인 관계 보존 penalty, noise seed, bounds, Powell48평가,
internal 진단/heldout 격리/생성seed/비용판정을 모두 동일하게 유지한다.
새 GMM 적합0, 두 계수 보정2개,2부모×4draw×2팔=16새 전체 생성이다.
실제 과거와 생성 과거 팔 모두 원래 GMR을 같은 지원집합에서 비교한다.
기존 GMR 파일은 변경하지 않고 correction buffer만 별도 checkpoint에 추가한다.

normal gap이 두 부모 모두 기존 GMR과 history_only보다 낮아야 하고 GMR 대비
onset 개인/정상 개인≤+.01, 사기 gap/금액/정상 금액/merchant TV≤+.02,
diversity≥95%, phase GMM 대비 정상/사기 gap≤+.03 조건을 유지한다.
추가 rollout 효과는 clock_prefix보다도 정상 gap이 작고 보존 조건을 통과할 때
지지한다. 단순 prefix 보정으로 더 잘 해결되면 알려진 보정의 효과로 해석하며
rollout 학습의 새 기여라고 하지 않는다. 두 데이터 적합 공유는 독립 부모 재학습이 아니다.

출력 routing/clock 재정렬/첫 gap/labels/길이/hash/독립 raw 지표 검증도 동일하다.
아래 문서는 변경하지 않고 참조한다.

- [원래 등록](../argn_clock_evolution_v1/PROTOCOL.md)
- [추가 근거의 내부 수치](../argn_clock_evolution_v1/internal_conditional_diagnostics.csv)
