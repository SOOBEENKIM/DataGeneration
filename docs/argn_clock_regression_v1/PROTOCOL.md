# 추가 강한 단순 대조: clock 조건부 GMR — 2026-09-27 UTC

새3팔의 학습 곡선에서 희귀 전환 내부 검증 손실이100step 이후 악화했고,
relative는0step을 선택했다. 일반 이력 출력 첫8생성은 onset 개선과 정상 개인
gap 비용을 함께 보였다. 상대 출력의 자유 생성 결과를 보기 전에 이 대조를 등록한다.
큰 신경망의 조기 과적합과 낮은 차원의 조건부 관계 학습을 구분하기 위한
**알려진 통계 비교군**이며 새로운 연구 기여가 아니다.

[Calinon2016 §5.1 식13–16](https://publications.idiap.ch/downloads/papers/2016/Calinon_JIST_2015.pdf)의
GMR은 결합 Gaussian mixture에서 관측 입력의 조건부 분포를 정확히 계산한다.
여기서는 (strict-past clock, log1p(gap))의2변수 결합을 4전환별로 학습한다.
현재 clock에 따라 성분 확률·평균이 바뀌는 단일 혼합분포로 샘플링한다.
로봇 trajectory/TP-GMM 프레임 적응 전체를 재현하는 것이 아니다.

[2026 normalization 분석 §4.2](https://arxiv.org/html/2603.11869v1)의 조건부 변화
문제에 대응하는 저차원 대조다. 최신 생성기를 대신하거나 신규성을 주장하지 않는다.
일반 이력/mark LNM과의 중복은 원래 시간 출력 프로토콜에 기재했다.

clock/fallback/codec/first-event 제외/private RNG/나머지 모델은 등록된
argn_time_density_v1과 정확히 같다. GMM 3성분, full covariance, n_init3,
reg_covar1e-4, seed20261115, 전환당 최대50000점을 optimization에서만 선택한다.
해당 고정 설정으로 전환4개를 한 번 적합하며 개발 성능으로 수를 선택하지 않는다.
macro 실제-prefix NLL 및 gap/개인 gap 진단은 기존3split/4sampling seed 그대로다.

같은2부모×4draw로 전체8개 거래열을 추가한다. 기존 GPU0/1 worker를 건드리지
않고 빈 GPU를 두 번 확인한 뒤2/3 중 가용 장치에서 실행한다. 최종 test 미사용.
공식 Gaussian 조건부 식의 독립 SciPy 밀도 대조, native routing/clock 검증,
labels/lengths 완전 일치와 raw 지표 재계산을 실행한다. 기존3팔/시드/선택 결과는
변경하지 않으며 별도 manifest를 사용한다.

채택 비용은 time_density_v1의 screen 그대로 적용한다. 이 단순 대조로 목표를
충족하면 먼저 강한 기준으로 보존하고, 그 효과를 새 neural architecture 기여로
부르지 않는다. 실패하면 비용과 actual-prefix/free-generation 차이를 보존한다.
