# 실행·검증 기록

- 2026-09-21, branch `research/cs-saf-external-audit-v1`.
- 사전등록/설정 commit `e9e61d2`, 구현/단위 검사 commit `6726a8e` 이후 처음 수치 실행.
- 과학 실행 `attempt_001` 성공. 동일 과학 실험 재실행/설정 변경 없음. CPU 약39.73초. 신경망 fit 0, 생성 0; 관측 빈도/금액 평균표는 fit했다.
- 실행 환경: `/home/finx_sbk/.conda/envs/cofseq/bin/python3`, 단일 BLAS/OpenMP thread. 호스트 `finx-System-Product-Name`. 이 호스트에서 GPU driver 조회는 실패했고 GPU 학습을 시작하지 않았다. 별도 SSH 원격 실행으로 기록하지 않는다.
- 첫 matplotlib 실행은 기본 설정 폴더를 쓸 수 없어 임시 cache를 사용했으며 그림 저장은 성공했다. 보충 그림은 명시적 `/tmp` 설정 폴더를 사용했다.
- `tests/test_external_relations_v1.py`: 7 passed. 사건 경계·양끝 tie 제외·과거 반복 상태·gap/OOV·train partition·bootstrap weighting·라벨 비사용/정규화 검사.
- `verify_external_relations_v1.py`: 원 분석 코드를 import하지 않고 pandas 빈도 집계 및 개체별 순차 반복으로 검산. profile 270행, score 40행의 값·개체 평균·행/개체 수 일치; 최대3.34e-14. 원본 run/source/config/output 해시 확인. 신뢰구간은 원 구현의 단위 검사 범위이며 별도 구현 재계산은 하지 않았다.
- 결과 확인 뒤 추가한 탐색 진단: 희소한 학습 조합의 지원 수, 빈 run 셀까지 포함한 coverage, merchant→category 단일 대응 여부. 점수 재학습/계수 선택/평가 대상을 변경하지 않았다. `verification.json`의 `postrun_*` 필드에 분리했다.
- 등록 결과 두 그림과 CSV/JSON은 원본에서 byte-identical로 복사했다. 추가 `observed_relations.png`는 기존 집계값만 재표현했다. 원본 거래/개체 ID/checkpoint를 Git에 추가하지 않았다.
- 입력 필터는 train/validation만 반환한다. manifest의 전체 파일 hash 확인에는 test 행이 들어 있는 parquet byte도 포함되지만 test outcome 분석을 한 것은 아니다.

## 재현 명령

작업 폴더는 저장소 루트다. 이미 존재하는 결과 폴더는 덮어쓰지 않는다. 다시 실행한다면 출력 경로를 새 이름으로 지정한다.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
/home/finx_sbk/.conda/envs/cofseq/bin/python3 -m pytest tests/test_external_relations_v1.py -q

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
/home/finx_sbk/.conda/envs/cofseq/bin/python3 -m experiments.cs_saf_external_relations_v1 \
  --data-root ../cof-seqgen-0707-2119-Version3-complete/data/cof_seqgen_saf \
  --output artifacts/cs_saf/external_relations_v1/attempt_001

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
/home/finx_sbk/.conda/envs/cofseq/bin/python3 -m experiments.verify_external_relations_v1 \
  --data-root ../cof-seqgen-0707-2119-Version3-complete/data/cof_seqgen_saf \
  --run artifacts/cs_saf/external_relations_v1/attempt_001 \
  --output artifacts/cs_saf/external_relations_v1/verification_001.json

MPLCONFIGDIR=/tmp/cs-saf-relations-mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/home/finx_sbk/.conda/envs/cofseq/bin/python3 scripts/plot_external_relations_v1.py \
  --results artifacts/cs_saf/external_relations_v1/attempt_001 \
  --output artifacts/cs_saf/external_relations_v1/observed_relations.png
```
