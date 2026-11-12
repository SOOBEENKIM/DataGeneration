# 실행 전 검증

- amount control3개를 포함한 관련 테스트16개 통과.
- 실제 부모 checkpoint와 실제 정상/사기 고객 prefix16위치에서112개 digit
  logits를 캐시로 재생해 원래 teacher logits와 일치 확인.
- 복제 head의 생성 token이 같은 RNG에서 부모와 정확히 일치.
- native checkpoint의 strict load, 현재 label로의 분기, nonzero gradient 확인.
- routed와 mixture의 정확한 파라미터 수 일치, 한 label의 전문가 변경이 다른
  label 출력에 영향을 주지 않는지, natural objective가 원래 빈도를 유지하는지 검증.

본 학습 runner는 세 split의 모든 digit 위치에서 replay 일치를 재검증하고
검사 수와 cache hash를 기록한다. 아래 결과는 완료 후 append한다.

## 완료 검증

- 8학습/16자유 생성, 총3,159,744행 완료. 두 dispatcher worker 정상 종료.
- 16생성 모두 동일 부모·생성 seed의 duration 기준과 고객/길이/label 순서가 정확히 같다.
  사후 label 보정이 아니라 같은 전환 head·RNG를 유지한 결과다.
- 16생성의 정상/사기 금액 log-W1을 별도 원시 parquet+SciPy 계산으로 검산해
  최대 차이2.22e-16. `independent_amount_checks.json`에 전체 기록.
- 두 모델 각각 optimization816,283/internal-validation100,226/development177,997
  모든 위치의7개 amount digit을 replay해 native logits와 일치했다.
  합계15,323,084 digit 위치다. 기존 parent checkpoint hash 유지.
- 후속 현재 조건 probe에서4,584위치의 첫 digit native/cache 일치,6조건의
  sampling 완료, 두 worker 정상 종료. 첫 학습과 별도 등록한 탐색 진단이다.
