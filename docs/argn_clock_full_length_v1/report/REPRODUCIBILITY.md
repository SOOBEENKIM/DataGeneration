# 로컬 전체 길이 대조의 재현과 복구

기준 checkout: `research-argn-state-first`, commit
`0d620a97bd91e4cd34a246a6ca8557782c07f470`.
이 디렉터리는 git-ignore된 artifacts 안에 있다. 사용자 지시에 따라 결과 판정 전
tracked 파일, 활성 후보 및 GitHub branch를 변경하지 않는다.

Python은 `../research-reporting/argn-state-first-2026-09-27/runtime/bin/python`이다.
`PYTHONPATH=.:scripts`, OPENBLAS/OMP/MKL thread는 각각4다.

1. `local_source/run_full_length.py prepare`: 기존 simulator와 수치 대조 후
   이전1024계수의 전체48평가를 재현한다. 전체 optimization 메타데이터의
   두 계수를 별도 적합하고 internal validation 진단을 저장한다.
2. 같은 script의 `dispatch`: 실제 빈 GPU를 반복 확인하고 두 worker를 시작한다.
   각각 기존 native checkpoint에 새 time head를 연결해4개 동일 draw를 생성한다.
3. `local_source/report_full_length.py`: raw 지표 독립 재계산, 이전 output SHA 대조,
   이전 후반 지표144개 재현 및 부모별 비용 판정을 수행한다.

전체 적합은619고객, 첫 gap 제외815664거래, 그중1024위치 이후302200거래다.
최대 관측 거래열은3106건이다. 이전 fit은 같은619고객의513464 nonfirst 거래다.
정상→정상 학습 거래는509367→810543건, 유효 onset은333→429건이다.
새 두 계수는 두 native 부모가 공유하며 독립적인 두 neural training으로 세지 않는다.

실제 prefix만 쓰는 학습은 입력 clock이 파라미터와 무관하다. 이를 이용해 조건부
sample을 한 번 계산하고 location만 변경하도록 벡터화했다. 원래 simulator의
세 계수점에서 모든 목적 지표 차이는0이었고 이전48평가의 궤적도 재현했다.
추가 위치를 위한 noise는 기존1024위치의 uniform/normal 블록 뒤에 덧붙인다.
따라서 기존 고객·위치의 noise가 학습 길이 때문에 달라지지 않는다.

원래 GMR은 바꾸지 않았고 기존 checkpoint의 buffers와 새 head는 correction
두 계수를 제외하고 정확히 같다. 새 파라미터 선택은 optimization 목적에만
의존한다. internal/development 점수로 계수나 예산을 바꾸지 않는다.
최종 test 이벤트는 읽지 않는다. 같은48평가 예산의 소진은 수렴 완료와 다르다.

`MANIFEST.json`은 소스/원래가중치/학습자료/프로토콜 해시를, `EVALUATION_MANIFEST.json`은
새 development 점수를 읽기 전 평가 소스 해시를 기록한다. `verify()`는 이전 manifest와
git HEAD/변경 없음까지 확인한다. 잘못된 실행은 기존 경로에 덮어쓰지 않는다.

판정 실패 시 이 폴더의 로컬 기록만 남기고 기존 `clock_prefix`와 이전 저장소 상태를
그대로 사용한다. 원래 파일을 수정하지 않으므로 파괴적인 reset이나 삭제가 필요 없다.
실패 결과를 이후 성공 실험에 섞어 선택 사실을 숨기지 않는다.
