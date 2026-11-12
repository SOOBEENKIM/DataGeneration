# 2026 직접 비교 후보의 공식 구현 점검 — 2026-09-27

LBDTPP 공식 소스17개를 commit
`0ca4817bde2333ce0dcbf156496f7fed9fd48050`으로 고정해 읽었다.
로컬 보존 위치는 `artifacts/reference_sources/lbdtpp_20260927`이며
SOURCE_MANIFEST.json에 각 파일의 hash를 기록했다. 데이터 zip은 받지 않았고
공식 코드의 학습/평가 entry point는 실행하지 않았다.

원문 [§IV-A 식23–25, §IV-C, §V-A](https://arxiv.org/html/2606.24982v1)와
[공식 코드](https://github.com/Zh-Shuai/LBDTPP/tree/0ca4817bde2333ce0dcbf156496f7fed9fd48050)를
대조했다. 다음은 코드에서 확인한 사실과 우리 비교에 필요한 결정이다.

| 확인한 동작 | 같은 Sparkov 비교에서의 처리 |
|---|---|
| 학습 입력은 gap와 하나의 categorical mark다. 별도 고객 context·금액·merchant 출력은 없다. | `mark=(label,category)` 등의 정의를 고정하면 시간/상태 부분 비교는 가능하다. 이를 전체 혼합형 거래열 생성의 완전 동등 비교로 부르면 안 된다. |
| loader는 모든 sequence에서 첫 event를 `[1:]`로 제거한다. | 원본 첫 거래를 조용히 잃으면 초기 fraud/짧은 고객 평가가 달라진다. 입력 시작 sentinel 정책 또는 첫 거래 평가 범위를 명시하고 roundtrip 확인이 필요하다. |
| unconditional time scale은 학습에서의 최대 마지막 시간으로 정규화한다. | development/test를 scale fitting에 쓰지 않는 원칙은 유지한다. |
| evaluation은 각 실제 sequence의 `normed_final_time`을 stop_time으로 공급한다. | 현재 ARGN의 sampled length 자유 생성과 조건이 다르다. 관측 horizon을 주는 별도 공통 과제를 등록하거나 train-only horizon 생성을 붙인 적응임을 공개해야 한다. |
| get_data는 validation 모드에서도 train/dev/test 파일을 모두 읽는다. main은 train 뒤 eval을 호출한다. | 이번 프로젝트의 최종 test 미개봉 원칙을 지키려면 별도 허용 split loader/runner가 필요하다. 원본 entry point를 그대로 실행하지 않는다. |
| 공식 unconditional 설정은 block8·50epoch, 대부분 diffusion100/sample50이다. 생성은 최대500block이며 기본 cache length256이다. | Sparkov의 더 긴 고객열에 대한 memory/종료·시간단위 점검이 필요하다. 짧은 window로 잘라놓고 전체 고객열 비교라고 하지 않는다. |

따라서 가까운 최신 **시간·상태 생성** 비교 후보로 확보했다. 공식 학습 경로가
확인됐다는 점에서 TabDiT의 현재 공개 평가 코드와 다르다. 다만 label/gap 부분
비교와 전체 거래 필드 비교를 구분해 설계해야 한다. 위 차이는 LBDTPP의
성능 실패 증거가 아니다.

현재 경계 gap 실험은 직접 비교 결과를 대신하지 않는다. 다음 비교 준비에서
새 head를 더 늘리기 전에, 단순 조건부 시간 분포·충분한 CPAR와 이 최신 후보가
같은 경계 조건을 어떻게 보존하는지 확인한다. 모델 기여는 그 비교에서 남는
차이로 좁힌다.

기존 CPAR 실행 기록도 다시 확인했다. 공식 모델의128회 full-batch-equivalent
업데이트와688명 학습은 완료됐지만 단일 fit이며 수렴을 입증한 예산 탐색은
아니다. 개발 context의 RI/DE/HI4명은 native encoder에서 KeyError여서
143명 공통 지지에서 생성했다. 다음 CPAR 비교는 동일143명에서 다른 모델도
평가하고, 전체147명 결과와 분리해야 한다. 이 지지 문제를 나쁜 생성 품질로
계산하거나 제외한4명을 조용히 다른 모델 비교에 섞지 않는다.
[실제 CPAR 완료 기록](../sparkov_argn_control_v2/CPAR_EXECUTION_RESULTS.md).
