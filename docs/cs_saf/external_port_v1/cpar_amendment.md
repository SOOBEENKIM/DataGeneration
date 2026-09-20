# CPAR 전체 거래 보존을 위한 실행 정정

원 등록/첫 코드 `676fe6a` 이후, CPAR 생성 품질을 보기 전에 발견한 두 연결 문제를 정정한다. U/G/ARGN/단순 모델의 설정·학습·평가는 변경하지 않는다.

1. DeepEcho0.8.1 `segment_by_size`는 길이32에 못 미치는 마지막 구간을 버린다. 원 등록에서 공식 `segment_size32`가 모든 거래를 포함한다고 설명한 것은 틀렸다. Berka의 원 CPAR64epoch 가중치는 생성 직전 실패로 남았고, Sparkov 원 CPAR는 제가 시작한 작업만 중단했다. 두 실행은 전체 거래 보존 조건을 위반하므로 비교 결과에서 제외하고 파일을 보존한다.
2. PARSynthesizer는 single-table synthesizer의 `_set_random_state` API를 갖지 않는다. 샘플링 전에 해당 호출에서 오류가 났다. 이 모델의 난수인 Python/NumPy/Torch를 고정하고 data processor의 `reset_sampling()`을 호출한다.

별도 `CPAR_tail` 실행은 마지막1–31개 거래도 독립 segment로 유지하는 **명시적 입력 어댑터**를 사용한다. 원본·실현 순서·gap·mark·amount를 바꾸지 않고, fit에 제공된 사건 수와 분할 후 사건 수를 assert한다. PAR 네트워크·손실·최적화·64epoch·시드·생성 계획·평가 지표는 그대로다. 이 baseline은 ‘공식 CPAR 모델 + tail 보존 입력 어댑터’라고 표시하며 전처리까지 원본 그대로라고 하지 않는다.

원래 CPAR의 generated 평가 결과는 없고, 어떤 성능 개선을 보고 선택한 변경도 아니다. 이 정정/테스트를 저장한 뒤 새 폴더에 같은 seed로 실행한다. 이전 오류를 삭제하거나 성공한 원 실행으로 표시하지 않는다. 나머지 진행 중인 모델의 base config hash는 바꾸지 않는다.
