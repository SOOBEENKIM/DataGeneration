# 구조 비교 실행 기록

- 첫 CPU gate는 검사 코드의 2차원 boolean mask와 차원 추가를 한 인덱싱 표현에 넣어 중단됐다. `previous[mask][:,None]`로 검사 인덱싱만 수정했다. 모델/설정은 그대로이며 tiny training 및 과학적 학습 이전이었다. 최초 실패 로그는 `artifacts/cs_saf/structure_v1/technical_logs/cpu_gate_attempt01.log`에 보존한다.
- 두 번째 gate는 tiny U의 validation이 epoch0을 선택하여 best의 train loss 감소가 7%에 그친 데서 중단됐다. 실제 학습 loss는 25epoch 동안 약8.6→4.3으로 감소했으나 작은 validation에는 과적합했다. CPU 학습 동작 검사를 validation-best의 train loss로 판단한 것이 부적절했다. 같은 데이터·시드·25epoch·학습률을 유지하고, 최종 가중치 train NLL의 10% 감소와 validation-best 선택의 정확성을 별도로 검사하도록 고쳤다. 원래 tiny 실행/로그는 `smoke_attempt02_best_train_gate_failed`와 `technical_logs/cpu_gate_attempt02.log`에 보존한다. 과학적 학습 설정/판정 기준은 변경하지 않았으며 아직 과학적 학습은 시작하지 않았다.
- 세 번째 CPU gate는 모든 수학/생성/학습/재현 검사를 통과했다. 첫 GPU gate에서 CPU/GPU 최대 확률 차이가 1e-5를 넘었다. 같은 가중치로 cuDNN TF32만 비교한 결과 U/G/C 차이는 각각 3.44e-5/3.26e-5/3.23e-5였고, TF32를 끄면 모두 1.19e-7이었다. 검증 한도를 완화하지 않고 세 구조 모두 명시적 FP32(cuDNN 및 matmul TF32 off)로 고정했다. 첫 GPU 실패 로그와 진단 JSON, 기존 CPU PASS와 tiny 출력은 보존했다. 새 수치 설정을 포함한 동일 코드로 CPU/GPU gate를 다시 확인한 후에만 본 학습을 시작한다. 이 변경은 실행 정밀도를 명시한 것이며 구조·목적함수·학습 예산·과학적 판단 기준은 유지한다.

- 명시적 FP32 설정 후 CPU/GPU gate 모두 PASS. CPU tiny best 가중치는 이전 PASS와 세 모델 모두 정확히 같았다. GPU/CPU 확률 최대 차이는 세 모델 모두 1.19e-7이며 GPU reserved memory 최대 456MiB였다. 18회 본 학습 전에 두 gate 결과와 코드를 저장했다.

- 과학적 실행은 `6afb05b4ec06b255d0cd9b74a4f9df8276ecbd9e`에서 18회 학습·18개 부모 평가·162개 생성 데이터셋을 모두 완료했다. 본 실행의 실패·중단·재시도는 0회다. 사용 가능한 GPU를 각 학습 전에 확인했고 한 번에 한 학습만 수행했다. 모든 모델은 실제 13–16epoch에서 공통 중단 규칙에 따라 종료됐고, paired U/G/C의 선택 epoch는 각 조건·시드에서 같았다. 학습 합계 421초, 본 학습 최대 reserved memory 412MiB다.
- 과학적 판정은 FAIL이다. C의 활성 생성 L1과 Brier 비용이 세 시드 모두 두 대조군보다 나빴다. 고정된 허용선·시드·epoch를 바꾸지 않았으며 외부 확대·독립 데이터 학습·생성 손실 추가는 실행하지 않았다. 기타 분포 비용 및 비활성 생성 곡선 기준은 통과했다. 기술 성공과 방법 성공을 구분한다.
- 원본/hash/초기화/학습 순서/보정 최적해 검증을 통과했다. 별도 구현으로 저장된 feature 및 원시 생성 거래를 집계해 예측 값 324개와 생성 관계·모양 값 972개를 재계산했고, 최대 차이는 각각 4.91e-7과 1.00e-16이었다. 이 추가 검사는 재학습·새 생성·모델 선택 변경이 아니다. 최종 보고·표·그림에 전체 결과를 포함한다.
