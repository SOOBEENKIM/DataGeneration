# 초기 수정 모델의 100–199 과대 간격 진단 — 실행 전 고정

2026-09-28 사용자 승인 범위는 원인 분리다. `argn_clock_initial_coupling_v1/init_only/time_head.pt`를 고정한다. 새 적합, 계수 탐색, ARGN 학습, 전체 거래 생성, 모델 교체, push를 하지 않는다. 기존 결과의 초반 0.068592/전체 0.069124/100–199 0.162779/후반 0.079196을 보존한다. 최종 test를 열지 않는다.

## 선행과 해석

[Lin et al.2026 §III-B Proposition2 및 §IV](https://arxiv.org/html/2604.27182v1)의 조건부 법칙과 입력 분포 구분을 원문에서 재확인했다. 이 개념을 진단에 사용하며, 해당 GAN/MCMC 방법을 구현하거나 그 정리를 우리 모델의 오차 보장으로 사용하지 않는다. [He et al., EMNLP2021 §4–5](https://aclanthology.org/2021.emnlp-main.415.pdf)의 실제/생성 prefix 교체와 이후 회복 대조를 참고한다. 언어 생성에서의 결론을 금융 거래에 전용하지 않는다. 개인 이력 입력, prefix 교체, 시간 출력 보정은 알려진 방법이며 이번은 새 기여 확정이 아니다.

## 고정 대조

개입은 **100≤event_index<200의 normal→normal 출력에만** 적용한다. 초기 mixture와 1–99 생성은 동일하다. 200부터 기존 출력을 복구하고 1023까지 영향이 이어지는지 관찰한다. G/R은 각각 자체 생성/실제 strict-past 개인 속도다.

| 모드 | GMR 속도 | 보정 속도 | 기존 보정 |
|---|---|---|---|
| GG_on | G | G | 유지 |
| RG_on | R | G | 유지 |
| GR_on | G | R | 유지 |
| RR_on | R | R | 유지 |
| GG_off | G | 무관 | 대상 구간에서 0 |
| RR_off | R | 무관 | 대상 구간에서 0 |

모든 실제 속도는 현재 거래 이전의 gap만 사용한다. 실제 clock 제공 결과는 원인 진단이며 자유 생성 성능이 아니다. W1 차이를 가산적 원인 기여율로 해석하지 않는다.

1. **반복 재생**: 실제 label/length 경로(optimization/internal/development) 및 기존 8개 생성 label/length 경로에서, 초기 수정 head가 스스로 만든 시간 과거를 반복 입력한다. 같은 noise에서 6모드를 비교한다. 1–99 출력은 정확히 같아야 한다. 다른 상태의 수식은 그대로지만 변화한 과거 때문에 나중 출력이 바뀔 수 있다.
2. **저장된 과거 고정 대조**: 기존 init_only 전체 생성 8개의 과거 간격으로 각 시점의 입력 clock을 고정한다. 100–199 다음 간격만 6모드로 다시 뽑고 출력은 과거에 되먹이지 않는다. 직접 조건부 출력 효과와 반복 입력 효과를 구분한다. 이 대조도 완성 거래 데이터 생성 성능이 아니다.

optimization 619고객×noise2, internal69×noise4, development147×noise4; 생성 경로2부모×4draw×noise2. noise는 직전 진단과 같은 20261202–05이며 optimization/생성 경로는 첫2개다. 총156개 1024위치 반복 재생,96개 100위치 고정-prefix 재표본 대조다. 독립 학습 seed 반복으로 부르지 않는다. 최대 CPU worker4개.

평가 구간은6–49/50–99/100–149/150–199/100–199/200–499/500–1023. 실제·생성 라벨 경로가 모두 정상인 동일 entity/position을 비교하고 NN과전체정상을 분리한다. 원래 GPU 전체 생성의 비조건부 지표는 별도로 그대로 재현한다. label 불일치 고객의 행을 같은 행으로 취급하지 않는다. 고객 ID와 encoded record ID는 이전 검증된 명시적 매핑을 재사용하고 실제 metadata와 원본 label/간격을 대조한다.

지표: log1p gap W1, 고객 균등 marginal W1, signed mean log gap/clock bias, raw strict-past 개인 gap W1, GMR의 보정 전 평균 log 출력, 보정량 평균/분위수/clip 비율. 개인 관계의 분모는 oracle이 아닌 해당 생성 경로의 raw strict-past median이다. W1 외에 부호를 함께 보아 보정이 늘리는지 줄이는지 확인한다.

## 실행 전 검증과 판단

- 고정 head의 NumPy/Torch double 모수 일치, 무개입 재생이 직전 Simulation과 정확히 일치, 초기 mixture와 원래 GMR 모수 보존.
- clock은 gap를 개별적으로 cap한 뒤 strict-past median을 계산하는 기존 구현 그대로. encoded record/customer ID 매핑 검증. 첫 gap 제외, 재정렬/미래값 불변과 기존 경계 검증은 이전 검증과 함께 보존한다.
- 모든 국소 개입의 1–99 출력 동일성; off가 대상 phase 이외를 바꾸지 않는 같은-input 식 검사; 고정-prefix에서 보정만 변경하면 mixture 확률/분산 동일성 확인.
- 실제 clock으로 회복하고 off가 악화하면 속도 입력 쪽을 우선한다. 같은 clock에서 off가 회복하면 보정 부적합 가능성을 우선한다. RR_on/RR_off에도 큰 오류가 남으면 GMR 자체의 해당 위치 조건부 적합을 추가 원인으로 남긴다. 혼합 결과는 혼합으로 보고한다.
- 학습·internal·development 및 저장된 생성 prefix의 일치 여부와 표본 수를 함께 보고한다. 결과에 따라 모드·구간·noise를 늘리거나 새 학습을 섞지 않는다. 후속 수정은 이 진단 결과를 근거로 별도 설계한다.
