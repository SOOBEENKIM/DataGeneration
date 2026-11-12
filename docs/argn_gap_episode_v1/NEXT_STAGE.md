# 이번 대조 이후의 개발 판단

## 현재 확인한 것

- 이전 금액처럼 gap도 실제 이력·label·업종 조건에서 틀렸다. frozen encoder에
  native gap head만 충분히 조건별 학습하자 큰 폭으로 회복됐다. 이것은 현재
  설정의 희귀 조건 학습 실패를 지지하며, ARGN 전체 구조의 표현 불가능성을
  뒷받침하지 않는다. 동일 추가 학습의 natural/balanced 차이도 함께 본다.
- 기존 duration 모델은 이미 사기를 겪은 고객에게 다시 사기를 배치하는 경향이
  컸다. strict-past episode 상태와 초기 상태–길이를 연결한 단순 대조만으로
  이 오류가 크게 감소했다. 상태 추가 자체는 신규 기여 후보에서 제외한다.
- 관측과 생성 길이의 차이가 남았다. optimization에서 첫 거래부터 사기인 고객의
  최대 길이는15, 정상으로 시작하는 고객의 최소 길이는471이다. generated length
  draw4개에는 그 사이 길이가8–10고객씩 있다. 실제 development에도 길이16인
  사기 고객1명이 있으므로 이 영역 전체를 invalid라고 단정해서는 안 된다.
  이 결과는 새로운 길이 강제 cutoff가 아니라, 길이/초기 상태를 독립적으로
  정돈해서는 충분하지 않을 수 있다는 후속 가설이다.

## 다음에 분리할 것

1. 이번 strongest simple reference의 성능·비용을 기준으로 보존한다. gap+transition을
   무조건 최종 모델로 고르지 않고, 개인 대비 금액 및 정상 지표의 fit별 비용을 확인한다.
2. **표본추출의 변동, 길이 생성의 분포, 조건부 전환 오차를 분리**한다. label-only
   반복은768draw로 sampling range를 이미 점검했다. 새 실험은 optimization만으로
   p(length, initial state | static context)를 함께 모델링하는 단순 조건부 count/mixture
   대조와 현재 L-first 대조를 비교한다. 길이는 합성 모델에서 뽑고 실제 고객 미래
   길이를 공급하지 않는다. 실제 사례 복사나 사기 고객 수 강제로 해결하지 않는다.
3. 고객별 사기 시작은 현재도 run age≥21를 pooled한 hazard에 의존한다. 긴 정상
   이력에서 위험이 일정한지 optimization/internal-validation의 노출량으로 확인한다.
   시간/고객군/이전 episode 특성별 calibration을 본 뒤, 일반 조건부 hazard 회귀/MLP와
   개인 행동 입력 제거를 대조한다. 가중 classifier 점수를 그대로 자연 발생률로 쓰지 않는다.
4. 개인 대비 금액이 나빠지면 구간 나이/첫 사기 여부/정상 이력 길이의 구성 변화와
   같은 조건 내 출력 오차를 분리한다. 이미 age-standardized ratio 지표가 있으며,
   남는 비용을 확인한 뒤 전환과 출력에 동일한 strict-past 개인 상태를 연결하는 후보를
   그 일반 MLP 및 이번 head 보강 대조보다 나은지 검증한다.
5. Sparkov에서만 학습되는 단일 episode·짧은 사기 전용 시퀀스의 특수성으로 끝내지
   않는다. 다른 조건/자료와 충분한 seed, 적정 학습 CPAR 및 실행 가능한 최신 직접
   비교군을 마련한 뒤에 논문 기여/일반화/우위를 판단한다. 최신 문헌 중복 점검은
   새 mechanism을 정할 때 갱신한다.

이 문서는 **후속 모델 학습 완료를 의미하지 않는다.** 이번 요청 범위의4cell과
제거 대조는 RESULTS/INTERPRETATION/completion_evidence를 따른다. 다음 단계는
숫자를 좋게 보이게 만드는 후처리가 아니라 남은 생성 분해의 원인 대조다.
