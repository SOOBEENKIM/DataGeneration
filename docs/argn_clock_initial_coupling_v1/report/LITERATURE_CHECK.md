# 원문 재확인과 이번 변경의 위치 — 2026-09-28

실행 전 근거는 PROTOCOL.md에 고정했다. 아래는 원문 확인 범위를 보충하는 기록이며
진행 중인 모델·손실·평가 조건을 변경하지 않는다.

| 원문과 이번에 확인한 범위 | 이미 알려진 내용 | 이번 연구에 적용하는 판단 |
|---|---|---|
| [Intensity-Free Learning of Temporal Point Processes, ICLR2020 §3.2–3.3](https://arxiv.org/pdf/1909.12127) | log-normal mixture로 간격 밀도를 표현하고, 이력·metadata·sequence embedding으로 조건부 모수를 정한다. | 초기 단순 mixture는 알려진 밀도 모델이다. 고객 정보를 넣는 것 자체도 차별점으로 삼을 수 없다. 이번에는 초기 marginal과 후속 보정 연결의 효과를 분리한다. |
| [LBDTPP2026 Algorithm2, §IV-C/E, Assumption1/Theorem1](https://arxiv.org/html/2606.24982v1) | 비어 있는 이력에서 블록 생성을 시작하고 생성 prefix를 후속 조건으로 사용한다. 오차 경계는 local approximation과 prefix 안정성 가정에 의존한다. | 초기 생성과 후속 조건 분포의 관계는 선행 목표다. 현재 작은 GMR 보정의 안정성이나 block diffusion 우월성을 그 정리에서 곧바로 도출하지 않는다. 여기서는 단계별 개입과 실제 전체 생성으로 검증한다. |
| [MoTPP, 2026-09-18 §4.2/5.1/5.2](https://arxiv.org/html/2609.21382v1) | 기존 THP encoder에 zero-inflated LogNormal mixture 출력을 붙이고 우도로 학습한다. 활동/시간은 이력 조건부 독립으로 분해하고, 남은 시간 rollout에 실제 남은 사건 수를 사용한다. | 기존 encoder를 유지한 시간 출력 개선도 이미 선행된다. 실제 길이를 주는 조건부 학습/진단과 우리 전체 자유 생성 결과를 구분한다. 이 모델이 Sparkov에서 실패했다고 주장하거나 이번 head를 최신 전체 생성기보다 우수하다고 쓰지 않는다. |

이번 실험은 이 문헌들의 전체 모델 재현이 아니다. 알려진 단순 초기 분포와 생성 이력에서
학습한 후속 보정의 2×2 대조다. 직접 비교가 남은 최신 생성기와 다른 데이터까지 평가해야
최종 논문 기여의 범위를 판단할 수 있다. 단순 초기 mixture만으로 같은 효과가 나오면
더 복잡한 결합 방식의 필요성을 주장하지 않는다.
