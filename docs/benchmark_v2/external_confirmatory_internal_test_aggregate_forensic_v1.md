# External confirmatory internal-test aggregate and forensic v1

## 결론

기존 frozen non-v3 CoF-SeqGen이 IID, CTGAN, TVAE보다 **전체적으로 우수하다는 주장은 지지되지 않는다**. CoF는 confirmatory internal test에서 두 데이터 모두 coherence 1위였지만, fidelity와 combined score는 AMLSim 4위, Sparkov 3위였다. 두 데이터 equal-weight macro에서도 coherence만 1위이고 fidelity와 combined score는 3위다.

따라서 허용되는 엄격한 판정은 `ONLY_DESCRIPTIVE_COHERENCE_ADVANTAGE_SUPPORTED`이다. 이는 저장된 단일 실행들의 기술적(descriptive) 비교이며 통계적 또는 일반적 우월성 주장이 아니다.

## 범위와 무결성

이 분석은 8개 `attempt_001`의 저장된 `COMPLETE.json`, `artifact_index.json`, `manifest.json`, `evaluation.json`과 authorization checksum, 기존 validation aggregate JSON만 읽었다. sample/checkpoint를 열지 않았고 test-time selection, 재생성, 재학습, GPU/CUDA, TSTR, privacy, Sparkov fraudTest 접근은 모두 0회였다.

각 job에서 다음 연결을 실제 SHA-256으로 검증했다.

1. `COMPLETE.status == COMPLETE`
2. `COMPLETE.artifact_index_sha256 == SHA256(artifact_index.json)`
3. index의 `manifest.json` hash가 실제 manifest와 일치
4. index의 `evaluation.json` hash가 실제 evaluation과 일치
5. dataset/model job ID 일치
6. terminal과 evaluation의 `test_time_selection_performed == false`

8/8 job이 모두 통과했다. 전체 파일별 hash는 [JSON evidence](external_confirmatory_internal_test_aggregate_forensic_v1.json)에 고정했다. 실행 authorization은 source commit `5bebc990d49a7bbb034b3900bef86bf08fdd66c1`, config `c459da6a74d63f971d1855082dea8b062921def64f8e12c4750817abe2172500`, relevant source `377749127defdc73fe0bd776717f070687d297bcbb61edcce64664d7ca71a3da`에 묶여 있다.

## 점수 해석

모든 점수는 낮을수록 좋다. Fidelity는 저장된 fidelity metric/불변 train-only threshold ratio의 최댓값, coherence도 같은 방식의 최댓값이다. Combined score는 두 최댓값의 산술평균이다. Macro는 AMLSim과 Sparkov에 같은 가중치를 준 산술평균이다. 원시 metric이나 threshold를 재계산하거나 변경하지 않았다.

## Confirmatory internal-test 결과

### AMLSim

| Model | Fidelity max ratio | Rank | Coherence max ratio | Rank | Combined | Rank |
|---|---:|---:|---:|---:|---:|---:|
| empirical IID | 1.796531 | 1 | 15.307182 | 3 | 8.551856 | 1 |
| CTGAN | 12.103967 | 3 | 15.318657 | 4 | 13.711312 | 3 |
| TVAE | 6.166695 | 2 | 14.707800 | 2 | 10.437247 | 2 |
| frozen non-v3 CoF | 13.771584 | 4 | 14.457383 | 1 | 14.114484 | 4 |

AMLSim에서는 CoF가 coherence만 가장 낮다. Fidelity와 combined score는 네 모델 중 가장 높아 전체 우월성 근거가 없다.

### Sparkov internal test

| Model | Fidelity max ratio | Rank | Coherence max ratio | Rank | Combined | Rank |
|---|---:|---:|---:|---:|---:|---:|
| empirical IID | 2.142842 | 1 | 4.559339 | 3 | 3.351091 | 1 |
| CTGAN | 11.133907 | 2 | 4.152255 | 2 | 7.643081 | 2 |
| TVAE | 48.588298 | 4 | 430.002688 | 4 | 239.295493 | 4 |
| frozen non-v3 CoF | 12.681494 | 3 | 3.134546 | 1 | 7.908020 | 3 |

Sparkov에서도 CoF가 coherence는 가장 낮지만 fidelity와 combined score는 IID와 CTGAN보다 높다. TVAE보다는 세 점수가 모두 낮다.

### Equal-weight cross-dataset macro

| Model | Macro fidelity | Rank | Macro coherence | Rank | Macro combined | Rank |
|---|---:|---:|---:|---:|---:|---:|
| empirical IID | 1.969686 | 1 | 9.933261 | 3 | 5.951473 | 1 |
| CTGAN | 11.618937 | 2 | 9.735456 | 2 | 10.677197 | 2 |
| TVAE | 27.377496 | 4 | 222.355244 | 4 | 124.866370 | 4 |
| frozen non-v3 CoF | 13.226539 | 3 | 8.795964 | 1 | 11.011252 | 3 |

Macro는 dataset 간 표본수 차이를 반영한 추정량이 아니라 두 dataset의 저장 점수를 동일 가중한 기술 통계다.

## Validation 대비 순위와 방향

| Dataset | Metric | Validation order | Internal-test order | Concordant pairs | Discordant pairs | Spearman rho |
|---|---|---|---|---:|---:|---:|
| AMLSim | fidelity | IID < TVAE < CTGAN < CoF | IID < TVAE < CTGAN < CoF | 6 | 0 | 1.0 |
| AMLSim | coherence | CoF < TVAE < IID < CTGAN | CoF < TVAE < IID < CTGAN | 6 | 0 | 1.0 |
| AMLSim | combined | IID < TVAE < CoF < CTGAN | IID < TVAE < CTGAN < CoF | 5 | 1 | 0.8 |
| Sparkov | fidelity | IID < CTGAN < CoF < TVAE | IID < CTGAN < CoF < TVAE | 6 | 0 | 1.0 |
| Sparkov | coherence | CoF < CTGAN < IID < TVAE | CoF < CTGAN < IID < TVAE | 6 | 0 | 1.0 |
| Sparkov | combined | IID < CTGAN < CoF < TVAE | IID < CTGAN < CoF < TVAE | 6 | 0 | 1.0 |

여섯 dataset×metric 순위 중 다섯 개가 완전히 일치한다. 유일한 역전은 AMLSim combined의 CTGAN 대 CoF다. Validation에서는 CoF가 CTGAN보다 0.404728 낮았지만, internal test에서는 CoF가 CTGAN보다 0.403172 높았다. Cross-dataset macro 순위는 세 metric 모두 validation과 internal test가 일치한다.

CoF의 비교 방향은 다음과 같이 제한적으로 재현됐다.

- Coherence: validation과 internal test 모두에서, 두 dataset 각각 CoF가 나머지 세 모델보다 낮았다.
- Fidelity: 두 단계 모두 AMLSim에서 CoF가 최하위였고 Sparkov에서 IID·CTGAN보다 뒤였다.
- Combined: 두 단계 모두 dataset별 1위는 IID였다. CoF는 AMLSim validation에서만 CTGAN을 근소하게 앞섰으며 internal test에서 그 방향이 뒤집혔다.

## 엄격한 주장 판정

| Claim | Decision | 근거 |
|---|---|---|
| CoF가 IID·CTGAN·TVAE보다 전체적으로 우수 | **NOT SUPPORTED** | 양 dataset의 fidelity/combined에서 일관된 우위가 없고 macro도 3위 |
| CoF가 coherence에서 기술적으로 우수 | **SUPPORTED (descriptive only)** | validation 및 internal test의 양 dataset과 macro에서 모두 1위 |
| CoF의 통계적·일반적 우월성 | **NOT ESTABLISHED** | dataset/model당 저장된 단일 실행이며 inferential comparison이 아님 |

전체 수치와 검증 가능한 provenance는 [JSON evidence](external_confirmatory_internal_test_aggregate_forensic_v1.json), phase별 machine-readable 표는 [CSV evidence](external_confirmatory_internal_test_aggregate_forensic_v1.csv)에 기록했다.
