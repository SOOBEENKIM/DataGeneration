# REPO_MAP.md — TabDiff Fork (CoF-SeqGen base)

실제 파일을 읽어 작성. 추측 없음.

---

## 1. Denoiser / Transformer 모델 파일

### `tabdiff/modules/transformer.py`
- `Tokenizer`: 수치형(linear) + 범주형(embedding matrix) → d_token 차원 토큰화. `[CLS]` 토큰 포함.
- `Transformer`: multi-head self-attention + FFN 스택. timestep conditioning 없음(AdaLN은 main_modules에서 처리).
- `Reconstructor`: Transformer 출력 → 수치형/범주형 복원 head.

### `tabdiff/modules/main_modules.py`
- `PositionalEmbedding`: sinusoidal timestep embedding.
- `MLPDiffusion`: 단순 MLP denoiser (row-level, 위치 축 없음). `x → proj → MLP → x̂`. timestep은 `map_noise → time_embed`로 조건화.
- `SiLU`: activation.
- **CoF-SeqGen 주목**: `MLPDiffusion`이 현재 row-level 처리. **여기에 position 축 + windowed self-attention + AdaLN-Zero 추가 필요.**

---

## 2. Diffusion Forward / Reverse

### `tabdiff/models/unified_ctime_diffusion.py` — `UnifiedCtimeDiffusion`
- **continuous-time** mixed-type diffusion.
- 수치형: EDM-style (variance-exploding), `power_mean` noise schedule.
- 범주형: masked diffusion, `log_linear` noise schedule. 각 카테고리별 독립 마스킹.
- `_denoise_fn`: 외부에서 주입(현재 MLPDiffusion).
- `y_only_model`: 라벨 전용 guidance 모델(imputation용, 현재 optional).
- `num_classes`: 각 범주형 feature의 클래스 수 벡터. label Y도 여기 포함시키면 joint diffusion 가능.
- **CoF-SeqGen 주목**: label Y를 `num_classes`에 추가하면 joint diffuse 구조 재사용 가능.

### `tabdiff/models/noise_schedule.py`
- `LogLinearNoise`: 범주형 마스킹 스케줄 (ε-max ~ ε-min log-linear).
- `PowerMeanNoise` (추정): 수치형 EDM schedule.

---

## 3. 데이터 전처리

### `src/data.py`
- `StandardScaler1d`, `LeaveOneOutEncoder` 기반 수치형/범주형 인코딩.
- train 통계로만 fit, val/test에 적용 (leakage 없음).
- `ArrayDict` / `TensorDict` 타입 사용.
- **CoF-SeqGen 주목**: 현재 row-level. entity group + timestamp 정렬 + Δt-bin 로직 추가 필요 (`data/prepare_sparkov.py`, `data/build_sequences.py`).

### `process_dataset.py`
- 원시 CSV → TabDiff 포맷(parquet/npz) 변환 스크립트.
- 현재 adult, default 등 UCI 데이터셋 기준.

---

## 4. Train Entrypoint

### `main.py` (repo root)
- argparse: `--dataname`, `--mode`, `--method`, `--gpu`, `--no_wandb`, `--exp_name`.
- GPU 선택: `cuda:{args.gpu}`.
- `tabdiff.main.main(args)` 호출.

### `tabdiff/main.py`
- 실제 학습 루프 오케스트레이션.
- config(`tabdiff_configs.toml`)에서 모델 하이퍼파라미터 로드.
- `Trainer` 클래스로 학습.

### `tabdiff/trainer.py` — `Trainer`
- EMA 모델 유지 (`ema_decay=0.997`).
- `ReduceLROnPlateau` 스케줄러.
- `c_lambda`, `d_lambda`: loss weight (현재 단일 L_diff).
- **CoF-SeqGen 주목**: `c_lambda` → `λ·L_coh` weight로 재활용 가능.

---

## 5. Config 위치

### `tabdiff/configs/tabdiff_configs.toml`
- 모델 하이퍼파라미터 (d_token, n_layers, n_heads, lr, steps, batch_size 등).
- 데이터셋별 설정 섹션.

---

## CoF-SeqGen 구현을 위해 추가/수정할 파일 (CLAUDE.md 기준)

| 파일 | 현재 상태 | 할 일 |
|---|---|---|
| `data/prepare_sparkov.py` | 없음 | 신규: cc_num group + unix_time 정렬 + Δt 분 단위 + drop 개인정보 |
| `data/prepare_amlsim.py` | 없음 | 신규: SENDER_ACCOUNT_ID entity + TIMESTAMP + TX_AMOUNT log |
| `data/build_sequences.py` | 없음 | 신규: Δt time-bin(B=16), window L, pad+mask, segment id |
| `data/splits.py` | 없음 | 신규: entity-disjoint 70/10/20 split + 캐시 저장 |
| `models/backbone.py` | 없음 | `MLPDiffusion` → sequence(position 축 + windowed attn + AdaLN(t)) |
| `models/embeddings.py` | 없음 | Δt-bin embed + window/segment posenc |
| `models/heads.py` | 없음 | XRestorationHead + LabelHead |
| `models/soft_g.py` | 없음 | 미분가능 행동 요약 (vel/gap/rep), detach 금지 |
| `models/teacher.py` | 없음 | MLP f_Ï + pretrain/freeze |
| `diffusion/process.py` | 수정 | label column을 `num_classes`에 포함 |
| `diffusion/losses.py` | 수정 | `L_diff + λ·L_coh` |
| `train_teacher.py` | 없음 | real train으로 f_Ï 학습 후 freeze |
| `train_cofseq.py` | 없음 | 메인 학습 루프 |
| `eval/metrics.py` | 수정/확장 | AUPRC, Recall@1%FPR, Lift Ratio, coherence gap |
| `eval/detector.py` | 없음 | HistGradientBoosting TSTR |

---

## 환경 정보
- Python 3.10, torch 2.1.2+cu121, numpy 1.26.4
- GPU: RTX 3090 × 4 (각 25.3 GB), CUDA 12.6
- conda env: `cofseq` (`<LOCAL_HOME>/.conda/envs/cofseq`)
- 원시 데이터: `<LOCAL_DATA_ROOT>/`
  - Sparkov: `Sparkov_fraud_train_test/{fraudTrain,fraudTest}.csv`
  - AMLSim: `ibm_amlsim_example_accounts_transactions_alerts/{accounts,transactions,alerts}.csv`
  - IEEE-CIS: `ieee_fraud_detection_train_test/{train_transaction,train_identity}.csv`
