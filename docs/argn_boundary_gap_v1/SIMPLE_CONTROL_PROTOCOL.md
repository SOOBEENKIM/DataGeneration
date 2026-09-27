# 부가 진단: 과거를 쓰지 않는 조건부 시간 혼합분포

2026-09-27, 새로운 전체 생성 결과의 평가 전에 등록한다. GPU gap 대조의 학습,
선정, 자유 생성, 채택 screen은 변경하지 않는다.

복잡한 gap 전문가가 단순한 transition 조건 분포보다 필요한지 알아보기 위해
optimization의 두 전환별 `log1p(gap)` Gaussian mixture(K=1,2,3)를 학습한다.
K는 internal-validation의 해당 전환 NLL로 고른다. 직접 간격 밀도 학습의 근거는
[CIKM2022 §3.2 LNM](https://arxiv.org/html/2210.15294v2)이며, 여기서는 역사 encoder와
개인 조건을 제거한 더 단순한 통계 대조다. 원 논문 신경 TPP 재현이라고 하지 않는다.

실제 development 전환 위치 각각에서4개 공통 seed로 간격을 샘플링하고
전환별 W1, optimization에서 고정한 과거 간격 사분위별 W1을 비교한다.
양의 간격으로 복원하며 native gap codec 범위로 clipping한다. 모든 clipping 수를 기록한다.
이 대조에는 실제 phase가 제공되므로 자유 생성 결과가 아니며, 고객 구성/금액을
동시에 보존했다는 증거가 될 수 없다. 작은 셀의 고객·행 수를 명시한다.
조건부 시점에서 간단한 모형이 비슷하거나 더 좋으면 head 복잡성/과거 활용의
독립 기여를 보류하고, 전체 자유 생성에 붙인 대조가 다음 우선 실행이다.
