# 외부 파일럿 실행 기록

## 고정한 소스와 변경 이력

- 이전 외부 관계 진단 기준: `04a8918`.
- 외부 이식 코드·전체 자료 CPU 검증 후의 사전등록: `676fe6a` (2026-09-21 00:46:41 KST). 설정 `configs/cs_saf_external_port_v1.json`, SHA256 `9037bd62a2c08caf72fcf83f45c0d02cd420728a0c998b660752bb5ad782a5b7`.
- CPAR 연결 정정: `ebad17d` (00:51:22 KST). 네트워크·목적함수·학습 예산·시드를 바꾸지 않고 tail 보존과 설치된 sampling API 사용을 수정했다. 별도 amendment 설정의 SHA256도 각 corrected run에 남겼다. 기존 U/G/ARGN 설정 파일은 수정하지 않았다.
- 저장 모델/생성물의 검산·금액 support 분석과 보고서 작성은 결과 후 작업이다. 추가 학습이나 생성, 모델 선택 기준 변경은 없다.

각 실행의 source commit, 입력 preflight 해시, 설정 해시, 환경 버전, GPU 번호, 출력 해시는 [results.json](results.json)에 남겼다. 원 모델의 분기와 결과를 덮어쓰지 않았다.

## 장치와 실행 범위

`finx-System-Product-Name` 워크스테이션의 RTX 3090 4개를 확인했다. 기본 제한 환경에서 NVIDIA 장치가 보이지 않던 문제와 실제 GPU 점유를 구분했다. 장치 접근 권한으로 확인했을 때 사용률 5% 미만·메모리 512MiB 미만인 GPU만 각 작업 시작 전에 선택했다. 다른 사용자의 프로세스를 종료하지 않았다.

- GPU 0: Berka U → G.
- GPU 1: Sparkov U → G.
- GPU 2: Berka ARGN → Sparkov ARGN.
- GPU 3: CPAR 및 아래의 정정 실행.
- 단순 관측 대조군과 독립 histogram/ECDF 집계는 CPU. checkpoint 검산은 U/G 종료 후 빈 GPU 0에서 읽기만 수행했다.

유효 비교는 U/G 4개, ARGN 2개, CPAR-tail 2개의 신경 모델 fit이다. 기존 checkpoint의 재사용 성능처럼 설명하면 안 된다. 선택한 U/G 4개에 같은 6계수 보정을 fit했고, 두 자료에서 Marginal/Transition을 구성했다. 총 비교는 자료별 8변형×생성 난수 2회 = **32개 생성 데이터셋**이다. 모든 모델은 정해진 개체 계획을 끝까지 생성한다. ARGN의 실현 길이는 native 결과로 따로 기록한다.

## 제외한 원 CPAR 시도 2개

1. 원 Berka CPAR는 64epoch 후 잘못 호출한 `_set_random_state`에서 생성 전에 실패했다. 이 시도의 생성 품질 점수는 없다.
2. DeepEcho native `segment_by_size`가 마지막 1–31개 거래를 버리는 사실을 확인했다. 그 시점에 진행 중인 원 Sparkov CPAR와 그 dispatcher의 명령/PID를 확인한 뒤 **이번 작업의 해당 프로세스만** 중단했다. 그 시도의 생성 점수도 없다.
3. 두 원 시도는 전체 거래 보존 조건에 어긋나므로 비교에서 제외했다. `CPAR/`의 원 파일과 실패/중단 기록을 보존했다. 성공한 원 실행으로 바꾸지 않았다.
4. tail 보존 어댑터와 실제 설치 버전의 sampling reset을 검증하고 정정 커밋을 만든 뒤 `CPAR_tail/`이라는 새 폴더에 같은 seed로 재학습했다. 설정을 늘려 좋은 성능을 고른 것이 아니다.

유효 fit 8개 외에 이 **제외된 fit 시도 2개**가 있었다. 엔지니어링 검증용 소형 CPAR 1epoch와 gradient 처리량 검사는 과학 실험 fit에 포함하지 않는다. 처리량 검사의 optimizer update는 0이다. [원 기록](excluded_attempts.json), [정정 전문](cpar_amendment.md).

## 실행 전·후 검사

- CPU 통합 검사 39 pass / GPU 미노출 1 skip.
- GPU CPAR loss 동등성 검사 7 pass. 이 중 6개는 CPU 검사의 재확인이며 새 테스트 7개가 추가됐다는 뜻이 아니다.
- CPAR tail/API 추가 검사 2 pass. 한 검사는 짧은 CPU fit과 같은 seed의 재생성까지 수행한다.
- 총 42개 고유 테스트가 통과했다. 모든 가능한 코드 오류가 없다는 수학적 증명으로 표현하지 않는다.
- 1,990,071개 개발 거래의 target 개수·입력 해시를 확인했다. 각 자료/분할의 첫·창 경계·마지막 위치를 포함한 13,879개 target에서 U/G forward를 확인했다. 정규화 오차 최대는 input verification에 남겼다.
- 저장된 네 U/G의 초기 tensor 일치, 내부 check 최저 checkpoint, optimizer update 수를 확인했다. 실제 validation의 8개 raw/보정 조건에 대해 별도 double-precision 점수 집계와 원 점수의 최대 차이는 약 `2.45e-8`이다.
- 공식 ARGN의 저장 checkpoint를 native CPU loader로 다시 읽어 weights 일치와 실제 파라미터 수를 확인했다: Berka 2,345,953, Sparkov 3,599,240. 새 학습·생성은 없다.
- 모든 생성 결과의 독립 재계산 상세는 [independent_verification.json](independent_verification.json), 원 점수는 [generation_metrics.csv](generation_metrics.csv)에 있다. 평가 구현을 import하지 않고 NumPy histogram과 empirical CDF로 재계산했다.

테스트 코드는 다음 파일에 있다.

```text
tests/test_cs_saf_external_port.py
tests/test_cs_saf_external_metrics.py
tests/test_cs_saf_v2.py
tests/test_cs_saf_cpar_loss.py
tests/test_cs_saf_cpar_tail.py
```

## 해석에서 지킬 경계

이번 모델과 지표는 결과를 본 뒤 설정을 바꾸지 않은 제한된 파일럿이다. 하지만 자료는 이미 탐색한 개발 자료이고 학습 seed도 하나다. 따라서 독립 확인이나 논문 최종 성능 비교라고 부르지 않는다. CPAR의 학습 충분성, ARGN의 길이·전처리 차이, 단순 모델의 관측값 재표본은 [methods.md](methods.md)에 공개했다. 실패한 과거 후보를 이 결과로 성공으로 바꾸지 않는다.

금액 음수 확률과 invalid-only 수리 하한은 사후의 원인 점검이다. 원 생성 거래를 clipping하거나 점수를 교체하지 않았다. 새 금액 분포나 gap 조건 행동 head는 구현·학습하지 않았다. 다음 후보를 실행하려면 이 결과를 바탕으로 비교와 판단 기준을 별도로 고정해야 한다.
