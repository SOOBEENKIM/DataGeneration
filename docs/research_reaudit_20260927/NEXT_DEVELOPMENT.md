# ARGN에서 이어갈 최소 개발 명세

2026-09-27 재점검 이후 후보. 아래 신경망 학습은 아직 실행하지 않았다.
기존 단순 전환 생성 대조의 결과를 근거로 둔다. 실패한 S의 규모만 늘리는 변경이 아니다.

## 첫 질문은 실제 진단으로 답했다

**사기 label을 알고 있을 때, 실제 과거에서 다음 거래 금액을 제대로 생성하는가?**
저장된 label-first ARGN에서 실제 과거+실제 label/category/gap/merchant를
고정하고 amount digits만 순차 sampling했다. optimization/development와 정상/사기를
분리하고 label만 바꾸는 민감도도 확인했다. 실제 사기 금액408.43/361.07에
대해 약51을 생성했다. 정상 금액은 실제46.00/47.44와 비슷한 규모였다.
[진단 결과](AMOUNT_PROBE_RESULTS.md). 현재는 episode age별 금액 sampling
집계까지 완료한 것은 아니다. 이 조건부 결과를 무조건 생성 성능으로 보고하지 않는다.

그 결과에 따라 다음 최소 학습은 희귀 상태별 출력 학습을 우선한다.
우선 충분한 단순 head를 만든 뒤 개인/episode 정보의 추가 효과를 분리한다.
다음은 그 후보 구성 요소이며 세 변경을 한꺼번에 넣는 명세가 아니다.

**현재 ARGN도 이미 label을 조건으로 금액을 생성한다.** label 입력을 처음
추가한다는 제안이 아니다. 바꿀 대상은 상태별 파라미터 공유/출력 경로와 학습
표본·손실 배분이며, 기존 경로가 해당 조건부분포를 충분히 학습하지 못한 원인을
대조해야 한다. 먼저 같은 head의 상태별 학습 대조로 설명되는 이득인지 확인한다.
새 adapter가 필요하다면 동일 학습 목적의 일반 head와 용량을 맞춰 비교한다.

1. **전환:** 첫 거래 prior와 이후 onset/continuation/return 확률을 분리한다.
   일반 Bernoulli MLP가 ARGN 이력·정적 context와 strict-past 상태를 받게 한다.
   이전 label, 현재 run age/time, 과거 사기 경험/종료 이후 시간, 정상 이력의 금액
   요약을 후보 입력으로 두며 미래 사기 여부·정답 고객 길이는 입력하지 않는다.
   Markov·duration에 없는 개인/episode 정보를 제공하는 일반 head부터 비교한다.
2. **거래 출력:** 실제 이력의 rare-state 출력도 나쁘면 label 조건부 출력 학습을
   별도 대조한다. 기본 shared ARGN, 일반 class-conditioned head, rare-state
   residual adapter를 같은 정보·유사 용량으로 비교한다. 상태별 학습 표본을
   재배치해도 label prior를 임의로 50:50으로 바꾸지 않는다. 공유 파라미터가
   있는 경우 oversampling의 비용과 확률 보정은 실제 대조해야 한다.
3. **금액 표현:** conditional emission이 여전히 금액 규모를 놓칠 때만 개인
   기준 대비 log residual 또는 조건부 연속 분포를 검토한다. 첫/짧은 이력의
   fallback, 생성 중 기준 갱신, 실금액으로의 역변환을 같은 구현에서 검증한다.
   TabCascade와 기존 단순 조건부 분포가 가까운 선행이므로 변환 자체는 기여 아님.

## 학습 전 고정할 비교

- 공식 설정에 가까운 window100 ARGN과 현재 full-history 설정을 분리한다.
  같은 학습 목적에서 window 차이만 보는 대조 없이 장기 이력의 효과를 주장하지 않는다.
- 이미 실행한 raw / Markov / duration을 보존한다.
- 학습할 경우 변경 없는 ARGN / 전환 head만 / 거래 출력 변경만 / 둘 모두를
  비교한다. 개인·episode 특징 제거와 일반 MLP 대조로 정보·용량 효과를 분리한다.
- 학습 seed와 생성 seed는 다르게 반복하고 모든 결과를 보고한다. seed 추가로
  사전 실패를 성공으로 바꾸지 않는다. train/validation label별 loss와 위험 집합
  calibration도 남겨 희귀 전환을 못 배운 checkpoint를 총 loss만으로 해석하지 않는다.
- 정상 품질·merchant/category·다양성·길이·사기 고객 구성과 episode 수를 함께
  본다. 고차 희귀 지표는 셀/고객 support와 real-real 변동을 같이 제시한다.
- CPAR은 충분한 학습/설정 검토 후 비교한다. TabDiT는 공식 학습 코드 미공개
  상태여서 독립 구현의 범위/동등성 점검을 따로 기록해야 한다. 같은 과제의
  실행 가능한 순차 모델과 새 문헌을 후보 확정 시 다시 대조한다.

## 어떤 결과가 논문 기여 후보가 되는가

전환만 회복하면 알려진 상태 모델 적용 결과다. 거래 출력만 개선하면 label 조건부
학습/금액 표현 개선으로 설명될 수 있다. **개인별 episode 배치와 그 안의 거래
속성을 동시에 보존하는 추가 효과**가 각 단독 변경·일반 head·단순 기간 모델을
넘어 남고, 정상 품질 비용이 제한적이며 다른 고객/자료에서도 재현되어야 한다.

가까운 선행은 REDSDS/REDSLDS/OmegaSDS(상태·기간), SAGE(값에 따른 의존성),
TabCascade(조건부 연속 출력), Imb-FinDiff(희귀 클래스), Seq2Synth/TabStruct(평가)다.
이 방법들이 하지 않은 특정 금융 평가를 추가했다는 사실만으로 방법론 신규성을
확정하지 않는다. 메커니즘의 차이와 실증 이득이 모두 필요하다.
