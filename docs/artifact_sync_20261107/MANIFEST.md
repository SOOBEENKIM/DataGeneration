# artifact 동기화 매니페스트 (2026-11-07)

- 원본(읽기 전용): `C:\Users\user\4way\DataGeneration\research-argn-state-first\artifacts\`
- 대상: `C:\Users\user\DataGeneration\docs\<폴더명>\` (하위 폴더 구조 유지)
- **git 명령은 하나도 실행하지 않았음** (add/commit/push/stash 없음). 삭제한 파일 없음. 원본 변경 없음.
- 이미 docs/에 존재하는 16개 폴더(9/28 이전 실험)는 건너뜀. 나머지 33개 폴더를 처리.
- `cof-seqgen-0707-2119-Version3-complete/data`(원자료)는 열람하지 않음.
- 합계: 복사 3064개 파일 / 34,224,497 bytes (32.6 MB); 제외 2352개 파일 / 66,726,252 bytes. 상세는 `MANIFEST.json`.

## 1. 복사한 폴더

| 폴더 | 복사 파일 수 | 복사 bytes | 제외 파일 수 |
|---|---:|---:|---:|
| argn_calibration_specificity_v1 | 326 | 2,421,271 | 126 |
| argn_clock_adaptive_v1 | 214 | 1,778,864 | 56 |
| argn_clock_feedback_gain_v1 | 227 | 1,746,132 | 56 |
| argn_clock_full_length_v1 | 75 | 380,532 | 14 |
| argn_clock_initial_coupling_v1 | 167 | 1,354,405 | 42 |
| argn_clock_recheck_v1 | 25 | 122,424 | 0 |
| argn_clock_stratified_loss_v1 | 213 | 2,281,390 | 56 |
| argn_early_history_diagnostic_v1 | 52 | 5,409,313 | 0 |
| argn_event_weight_control_v1 | 45 | 203,024 | 21 |
| argn_followup_diagnostics_v1 | 26 | 43,429 | 0 |
| argn_history_component_ablation_v1 | 177 | 847,412 | 84 |
| argn_history_representation_diagnostic_v1 | 45 | 3,150,174 | 14 |
| argn_init_validation_v1 | 596 | 3,754,035 | 294 |
| argn_joint_clock_density_v1 | 94 | 346,142 | 28 |
| argn_matched_init_comparison_v1 | 112 | 2,840,031 | 21 |
| argn_mid_history_diagnostic_v1 | 42 | 3,332,732 | 0 |
| argn_whole_validation_v1 | 240 | 1,901,582 | 95 |
| cof_seqgen_saf | 1 | 49,466 | 0 |
| comparison_completion_v1 | 134 | 1,168,406 | 2 |
| comparison_direction_20261001 | 1 | 6,576 | 0 |
| cpar_adequacy_v1 | 30 | 108,356 | 1 |
| dot_research_handoff_20261002 | 2 | 14,473 | 0 |
| lbdtpp_sparkov_v1 | 32 | 140,216 | 33 |
| meld_500_pilot_v1 | 131 | 358,670 | 1 |
| meld_600_pilot_v1 | 2 | 2,091 | 0 |
| meld_compatibility_v1 | 26 | 171,738 | 1249 |
| research_direction_20261001 | 2 | 18,754 | 4 |
| research_process_audit_20260930 | 15 | 203,125 | 0 |
| research_reassessment_20260929 | 7 | 48,150 | 82 |
| workspace_cleanup_audit_20261002 | 2 | 4,362 | 3 |
| workspace_consolidation_20261002 | 3 | 17,222 | 0 |

복사 확장자 분포: json 1512, csv 1053, log 284, py 107, md 80, png 9, pdf 9, jsonl 8, txt 2.

## 2. CSV 점검 (원시 거래 행 여부)

- 복사 대상 CSV 1053개(+제외 1개 = 전체 1054개)의 헤더를 모두 확인. entity_id / event_index / amount_or_numeric_value / receiver_or_mark / timestamp 등 이벤트 단위 컬럼을 가진 CSV는 **없음**. 모두 지표/학습곡선/고객별 집계(customer_id 포함 CUSTOMER_DRAWS, CUSTOMER_ERRORS 등은 고객별 W1 집계)였음.
- 200 KB 초과 CSV 33개의 헤더도 개별 확인(예: argn_early_history_diagnostic_v1/report/all_diagnostics.csv 2.1 MB = 구간별 지표 집계, generated_path_*.csv / generated_*.csv = 구간/그룹별 W1 지표).
- 이름이 generated_* 인 CSV(16개)는 이름 때문에 따로 헤더를 열어 확인했으며 집계 지표임(행 = split/구간/group 단위).
- 원시 거래 행으로 보이는 파일은 발견되지 않음.

## 3. 제외 항목 (사유별)

- **vendored third-party code/dependencies/model assets**: 1282개, 27,659,024 bytes
  - lbdtpp_sparkov_v1: 33개
  - meld_compatibility_v1: 1249개
- **workspace dataset statistics/metadata (value codes, min5/max5 real values); flagged**: 896개, 10,329,088 bytes
  - argn_calibration_specificity_v1: 126개
  - argn_clock_adaptive_v1: 56개
  - argn_clock_feedback_gain_v1: 56개
  - argn_clock_full_length_v1: 14개
  - argn_clock_initial_coupling_v1: 42개
  - argn_clock_stratified_loss_v1: 56개
  - argn_event_weight_control_v1: 21개
  - argn_history_component_ablation_v1: 84개
  - argn_history_representation_diagnostic_v1: 14개
  - argn_init_validation_v1: 294개
  - argn_joint_clock_density_v1: 28개
  - argn_matched_init_comparison_v1: 21개
  - argn_whole_validation_v1: 84개
- **third-party papers/html/text/code (literature)**: 82개, 23,810,351 bytes
  - research_reassessment_20260929: 82개
- **record_only: third-party source folder (instructions)**: 70개, 482,830 bytes
  - reference_sources: 19개
  - reproduction_sources: 51개
- **vendored third-party package (tabulate) inside local_source**: 11개, 132,214 bytes
  - argn_whole_validation_v1: 11개
- **third-party MeLD upstream source files; record only**: 4개, 30,265 bytes
  - research_direction_20261001: 4개
- **lock file**: 4개, 0 bytes
  - comparison_completion_v1: 2개
  - cpar_adequacy_v1: 1개
  - meld_500_pilot_v1: 1개
- **bulk server file inventories with absolute server paths/username; low value; flagged**: 3개, 4,282,480 bytes
  - workspace_cleanup_audit_20261002: 3개

### 3.1 개별 기록 (record-only / 소수 항목)

`reference_sources`, `reproduction_sources`는 내용을 복사하지 않고 기록만 함 (전체 파일 목록은 MANIFEST.json):

- `reference_sources`: 19개 파일, 116,459 bytes. 내용: lbdtpp_20260927 (LBDTPP 공식 소스 main.py/model/dataloader/tpp_experiment, requirements, README, SOURCE_MANIFEST.json, TREE.json)
- `reproduction_sources`: 51개 파일, 366,371 bytes. 내용: vaegan_cpac_20260927 (VAE-GAN/CPAC 소스·LICENSE·scripts) 및 matplotlib_cache/fontlist-v390.json

기타 개별 제외:

- `comparison_completion_v1/COMPARATOR_LOCK` (0 B): lock file
- `comparison_completion_v1/CPAR_LOCK` (0 B): lock file
- `cpar_adequacy_v1/LOCK` (0 B): lock file
- `meld_500_pilot_v1/SUPERVISOR_LOCK` (0 B): lock file
- `research_direction_20261001/sources/1_train_vae.py` (13,830 B): third-party MeLD upstream source files; record only
- `research_direction_20261001/sources/3_train_DiT.sh` (155 B): third-party MeLD upstream source files; record only
- `research_direction_20261001/sources/4_sample_synthetic_data.py` (8,007 B): third-party MeLD upstream source files; record only
- `research_direction_20261001/sources/meld_commit.json` (8,273 B): third-party MeLD upstream source files; record only
- `workspace_cleanup_audit_20261002/CANDIDATE_FILES.csv` (1,190,026 B): bulk server file inventories with absolute server paths/username; low value; flagged
- `workspace_cleanup_audit_20261002/CHECKPOINT_CANDIDATES.json` (1,646,250 B): bulk server file inventories with absolute server paths/username; low value; flagged
- `workspace_cleanup_audit_20261002/CODE_CACHE_INVENTORY.json` (1,446,204 B): bulk server file inventories with absolute server paths/username; low value; flagged

(vendor/runtime_deps/clip/hf_cache/literature 등 대량 제외 파일의 개별 경로는 MANIFEST.json의 included=false 항목 참조. 가중치/체크포인트 확장자(.pt 등)는 인벤토리 대상 폴더 안에 존재하지 않았음; >5MB 파일은 `meld_compatibility_v1/vendor/MeLD-b1e5bb0.tar.gz` (6.8 MB, vendored 소스 아카이브, 어차피 vendor 제외).)

## 4. 사람이 판단해야 할 항목 (복사하지 않음 / 주의)

1. **`runs/**/workspace/` 안의 데이터 통계·메타 파일 896개 미복사**: `ModelStore/ctx-stats|tgt-stats/stats.json`, `OriginalData/ctx-meta|tgt-meta/*.json`, `independent_analysis_not_used_in_full.json`. 값 코드(카테고리/주 코드)와 실제 데이터의 min5/max5 극값 등 원자료 유래 통계를 포함하므로 공개 저장소에는 부적절할 수 있어 제외함. 같은 workspace의 `progress-messages.csv`(학습곡선)와 `model-configs.json`은 복사함. 공개해도 된다고 판단되면 별도 복사 필요.
2. **`workspace_cleanup_audit_20261002`의 대용량 인벤토리 3개 미복사** (CANDIDATE_FILES.csv 1.19 MB, CHECKPOINT_CANDIDATES.json 1.65 MB, CODE_CACHE_INVENTORY.json 1.45 MB): 서버 절대경로(사용자명 포함)·체크포인트 sha256 목록이며 연구 결과가 아님. SUMMARY.json, FINAL_CHECKPOINT_VERIFICATION.json은 복사함.
3. **`research_direction_20261001/sources/` 4개 미복사**: MeLD 업스트림 코드(1_train_vae.py, 4_sample_synthetic_data.py, 3_train_DiT.sh)와 meld_commit.json. 제3자 소스로 보고 제외. REASSESSMENT.md, REVIEW_COMPLETE.json은 복사.
4. **서버 절대경로 노출**: 복사된 JSON/로그 156개 파일이 `/home/ssd-990/soobeenkim/...`(141개) 또는 기타 `/home/` 경로(15개)를 포함, dot_research_handoff_20261002/CONTEXT.json은 사용자명이 대문자로 포함 (예: meld_500_pilot_v1/runs/*/EVALUATION_REQUEST.json, dot_research_handoff_20261002/CONTEXT.json). 기존 docs 72개 파일도 동일한 경로를 이미 포함하므로 기존 관행과 일치하지만, 공개 저장소 노출 여부는 사람이 결정. 비밀번호/토큰/API 키/IP/이메일 패턴은 복사 파일에서 발견되지 않음("token" 단어만 md 본문에서 7건, 토크나이저 맥락).
5. **`argn_whole_validation_v1/backbones/*/training_adapter_source.py` (39 KB, 3개) 및 `event_weighted_loss_source.py`**: 학습 어댑터 소스 스냅샷으로 판단해 복사했으나, 제3자 엔진 코드가 섞여 있을 가능성이 있어 라이선스 확인 권장.
6. **용량**: 복사본 약 34.2 MB. runs/ 아래 GENERATION_*.json, COMPLETE.json, 학습 로그 등 소형 파일이 3,000여 개로 파일 수가 많음. 필요하면 일부 폴더(runs/, logs/)를 줄이는 것도 고려.
7. 포함 목록에 없는 확장자인 `.jsonl`(학습/검증 곡선 8개, 지표만 포함)과 `.sh`는 지표 전용이거나 코드라 jsonl은 복사, `.sh`는 해당 폴더가 third-party sources라 제외됨. `.html`, `.ipynb`, `.gz`, `.old`, 확장자 없는 파일, 빈 LOCK 파일은 제외.

## 5. 제외 규칙 요약

- 확장자: .pt .pth .ckpt .parquet .arrow .npz .npy .pkl .bin .safetensors .h5 .so .pyc, __pycache__, lock 파일 (해당 없거나 LOCK 4개만 존재)
- 5 MB 초과, log 2 MB 이상, 그림 3 MB 이상 (해당 없음 / tar.gz 1개)
- 제3자 폴더: vendor, runtime_deps, clip, hf_cache, literature, tabulate(local_source 내), reference_sources, reproduction_sources, research_direction/sources
- 이벤트 단위 CSV: 없음 (전수 헤더 확인)

