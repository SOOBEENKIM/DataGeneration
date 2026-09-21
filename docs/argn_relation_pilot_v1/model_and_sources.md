# 이번 파일럿의 구조와 출처

기반 모델은 MOSTLY AI의 공식 `mostlyai-engine==2.4.0` TabularARGN이다.
[논문](https://arxiv.org/abs/2501.12012),
[공식 저장소](https://github.com/mostly-ai/mostlyai-engine).
공식 코드는 Apache-2.0이며 이번 저장소는 네트워크를 처음부터 만든 것처럼
표현하지 않는다. 논문/최신 소스와 과거 실험의 설치 버전을 구분한다.

모든 조건이 사용하는 기본 구조는 native mixed-type encoder → 순차 LSTM →
필드별 regressor/predictor다. 이 파일럿에서는 기존 열 순서
gap→merchant→amount→category, window32 학습, native 길이 생성을 유지한다.

추가 경로는 현재 업종의 logit에만 더한다. 현재 gap의 native code g와 직전
관측/생성 업종 c에서 embedding u(c),v(g)를 얻는다.

```
A: logits = official_logits
G: logits = official_logits + W tanh(u(c) + v(g)) + b
R: logits = official_logits + W (tanh(u(c)) * tanh(v(g))) + b
```

각 embedding은16차원, 마지막 W는16→15다. G/R 추가 파라미터는 모두2,143개다.
G는 같은 입력의 일반 한 은닉층 MLP, R은 직접 곱 상호작용 경로다. output W,b를
0으로 초기화하므로 시작 예측은 같고, 기본 tensor와 추가 tensor hash도 확인한다.
학습 손실은 native categorical CE이며 모든 파라미터를 함께 학습한다.
정답 법칙·평가 목표·생성 후 표 수정·사후 반복률 보정은 없다.

구현은 설치된 SequentialModel.forward의 category predictor 직후에만 경로를
삽입한다. generation loop에서는 native가 정렬/필터한 직전 출력의 category를
전달한다. 공식 패키지 파일을 덮어쓰지 않으며 context 종료 시 클래스 binding도
복원한다. 실행마다 원 forward/generation의 SHA256을 보존한다.

## 해석의 경계

- 현재 merchant와 amount는 category보다 먼저 생성된다. 추가 경로가 **현재**
  merchant/amount를 직접 수정하지는 않는다. 현재 category는 다음 거래의 LSTM
  이력과 다음 직접 경로에 반영된다. 공동 학습을 통해 기존 출력부도 변할 수 있다.
- 따라서 relation 점수만 개선하고 merchant–category의 정합성을 잃는 경우를
  별도 guard로 막는다. 이 위치에서 효과가 없다고 ARGN 기반 확장 전체가
  불가능하다고 결론 내릴 수 없다.
- 모델 이름 R은 이 실험의 식별자다. 과거 CS-SAF C/R/ER과 같은 모델이 아니다.
  G도 과거 반복/비반복 구조 G와 다른, 이 파일럿의 일반 MLP 대조군이다.
- bilinear interaction, residual path, MLP를 새로 발명했다는 주장이 아니다.
  통과해도 이 구성의 독창성·희소집단 선택성·독립 재현성은 추가로 입증해야 한다.
- native codec 통계가 fit+check를 사용한 기존 설정을 유지했다. 개체 test를
  보지 않았지만, 이 파일럿 validation은 이미 여러 번 사용한 개발 자료다.
- 이 공식 native 길이 비교의 지표와 길이를 강제로 맞춘 이전 U/G 실험의
  차이를 순수 구조 효과라고 직접 차감해서 해석하지 않는다.
