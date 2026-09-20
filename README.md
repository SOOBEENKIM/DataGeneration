# DataGeneration: CoF-SeqGen Research

**Latest research is on `research/cs-saf`; default `main` remains the earlier SAF base.**
Code, preregistrations, reports and numerical evidence are versioned. Large data,
checkpoints and raw paths remain on the workstation; GitHub is not their full backup.

Latest [calibrated U/E matched-history replay](docs/cs_saf/calibrated_replay_v1_report_2026_09_20.md)
is **COMPLETE: 400 immutable native paths, 800 predictor/path evaluations, zero
new training, calibration fits or generations**. On identical source histories
and current gaps, Ecal has higher probability-curve L1 than Ucal at 5/10/25%,
in 4/5 trials at each prevalence. The symmetric predictor term F satisfies the
registered .001 / 4-of-5 / 3-of-4 materiality screen; source-composition H and
empirical-score remainder S qualify only at 10%. Native error is recovered exactly
as F+H+S. F's between-trial intervals include zero at 5/25%; at 50% F is slightly
negative with mixed trials. This is an algebraic diagnosis, not unique causation
or proof that 32 added parameters are responsible. Conditional benefits remain
separate; E's generated-relation superiority is unsupported. All 1,200 empirical
and 2,400 probability-curve L1 values plus 600 decompositions passed independent
verification. GPU work finished. Next proposed bounded change is equal,
train-only gap-dependent calibration for U/E, not yet registered or executed.
[Professor brief](docs/cs_saf/professor_brief_calibrated_replay_2026_09_20.md),
[full statistics](docs/cs_saf/calibrated_replay_v1_result.json),
[all scalar values](docs/cs_saf/calibrated_replay_v1_scalars.csv),
[all bin curves](docs/cs_saf/calibrated_replay_v1_bins.csv),
[preregistration](docs/cs_saf/calibrated_replay_v1_preregistration.md).

Previous [fixed-model generation repeats](docs/cs_saf/generation_repeats_v1_report_2026_09_20.md)
is **COMPLETE: 800 fresh generation datasets (1,638,400 sequences), zero model or
calibration fits**. Five fresh tapes per frozen model were averaged within each
of five training trials; original tapes were excluded from the new primary mean.
Ecal has higher mean active repeat-curve L1 than Ucal at 5/10/25%, with positive
trial effects in 4/5, 5/5, 4/5 trials, satisfying the preregistered persistent-cost
screen. At 50% the mean difference is small and directions are mixed. Fixed-model
MC intervals exclude zero at 5/10/25%, but between-trial intervals include zero
at 5/25%; neither a unique cause nor general superiority is established.
Ucal-U passes the primary benefit screen at 5/10/25%; no comparison meets the
joint improvement screen. E's previously measured conditional/null-response
advantages remain separate fixed-model evidence. CPU/GPU gates, 1,248 cached-metric
comparisons, 800 saved-plan/model checks and 4,800 independent native values all
passed. GPU work is finished. Its proposed matched-history cross-evaluation
of calibrated U/E is now complete above.
[Professor briefing](docs/cs_saf/professor_brief_generation_repeats_2026_09_20.md),
[all 31,200 scalar metrics](docs/cs_saf/generation_repeats_v1_all_metrics.csv),
[nested statistics and verification](docs/cs_saf/generation_repeats_v1_result.json),
[preregistration](docs/cs_saf/generation_repeats_v1_preregistration.md).

Previous [equal-calibration U control](docs/cs_saf/calibration_u_control_v1_report_2026_09_20.md)
is **COMPLETE: 40 U calibration fits, zero neural retraining, 120 reused U/E/Ecal
references**. Ucal lowers mean active generated repeat-curve L1 at all four
prevalences; Ecal has higher mean L1 than Ucal at every prevalence. All four paired
L1 intervals include zero, and the registered consistency screen is not met.
Ecal nevertheless retains lower conditional TV and null response in 5/5 trials
at each prevalence. Calibration helps U too; E's conditional benefit does not
establish a native generation benefit. 40 frozen-base and paired-input checks,
CPU/GPU gates, and independent recomputation of 960 generation values passed
(maximum discrepancy zero). No technical retries; GPU jobs have ended.
[Professor briefing](docs/cs_saf/professor_brief_calibration_u_2026_09_20.md),
[verified numbers and fitted parameters](docs/cs_saf/calibration_u_control_v1_result.json),
[preregistration](docs/cs_saf/calibration_u_control_v1_preregistration.md),
[recoverable state](docs/cs_saf/STATUS.md).
Its proposed additional-tape evaluation is now complete above; independent
data confirmation remains unexecuted.

Previous [frozen repeat-probability calibration](docs/cs_saf/calibration_v1_report_2026_09_20.md)
is **COMPLETE: 80 four-scalar calibration fits, zero base-model retraining,
120 reused U/E/L003 references**. Calibration decreases active native repeat-curve
L1 in mean at all four prevalences, and decreases MI error in 5/5 paired trials
at every prevalence for both E and L003. However, only two of four prevalences
meet the preregistered primary consistency screen. L003cal still trades lower
null response for worse active generation than Ecal; Ecal has no consistent
primary advantage over U. This is a partial repair, not established model superiority.
17 CPU tests, CPU/GPU gates, a saved-state GPU regression and independent native
arithmetic (1,200 values, maximum discrepancy zero) passed. One TF32 verification
failure and its identical scientific rerun are preserved; GPU execution has ended.
[Professor briefing](docs/cs_saf/professor_brief_calibration_2026_09_20.md),
[verified numbers](docs/cs_saf/calibration_v1_result.json),
[preregistration](docs/cs_saf/calibration_v1_preregistration.md),
[execution amendment](docs/cs_saf/calibration_v1_execution_amendment_01.md).
U+calibration was missing at that stage and is now completed above. Independent-data
confirmation remains unexecuted.

Previous [cross-history replay and joint-oracle diagnosis](docs/cs_saf/replay_oracle_v1_report_2026_09_17.md)
is **COMPLETE: 120 saved models, 2,160 cross-history evaluations, 153,600 oracle
sequences, zero new fits**. The exact online oracle also loses gap–repeat fidelity
when observed gaps are forced and only marks are regenerated. Thus the earlier
OBS result does not uniquely identify a neural history-feedback defect.
On exactly the same generated histories, L003 still has worse repeat-curve
calibration than E at all four prevalences in each of five paired model trials.
Predictor differences are more consistent than source-history differences.
The report preserves all contrary results, quantifies the null-benefit/active-cost
tradeoff, and proposed the bounded calibration comparison subsequently completed
above. Its original diagnostic outcomes remain preserved.
[Professor briefing](docs/cs_saf/professor_brief_replay_oracle_2026_09_17.md),
[numerical evidence](docs/cs_saf/replay_oracle_v1_result.json),
[all binwise contrasts](docs/cs_saf/replay_oracle_v1_binwise.json),
[preregistration](docs/cs_saf/replay_oracle_v1_preregistration.md),
[recoverable status](docs/cs_saf/STATUS.md).

Previous [fixed-checkpoint generation diagnosis](docs/cs_saf/rollout_audit_v1_report_2026_09_17.md)
is **COMPLETE: 120 saved models, zero new fits, 368,640 prefix-anchored diagnostic
sequences**. Replacing real mark histories with recursively generated marks while
retaining observed gaps/values increases the paired expected repeat-curve error:
the preregistered descriptive screen passes for E-U at 3/4 prevalences and
L003-E at 4/4. This identifies a history-feedback sensitivity, not a unique causal
mechanism or an improved trained model. Hybrid gap/mark paths, reused data and
uncertain seed intervals limit the interpretation. All contrary results and
technical amendments are preserved. GPU execution has finished.
[Professor briefing](docs/cs_saf/professor_brief_rollout_audit_2026_09_17.md),
[verified evidence](docs/cs_saf/rollout_audit_v1_result.json),
[preregistration](docs/cs_saf/rollout_audit_v1_preregistration.md).

The preceding [lambda/prevalence/external follow-up](docs/cs_saf/followup_v1_report_2026_09_17.md)
is **COMPLETE: 170 new internal fits, 30 reused fits, 40 CPAR fits and 200 matched
internal generation evaluations**, across 5/10/25/50% prevalence and five paired trials.
Lambda .003 passes the mean-based conditional-accuracy screen at all four
prevalences, but its active paired intervals include zero and its generated
gap–repeat errors worsen vs E. All internal arms, including U, beat this pinned
CPAR implementation on the three active-context primary generation metrics;
this is not evidence that the new regularizer provides that advantage.
The report preserves the old FAIL, numerical results, gradient diagnosis,
CPAR implementation/budget limitations and concrete next experiments.
[Professor briefing](docs/cs_saf/professor_brief_followup_2026_09_17.md),
[verified evidence](docs/cs_saf/followup_v1_result.json),
[research status](docs/cs_saf/STATUS.md).
No independent-data, held-out-test or real-data confirmation has been performed.

Earlier stages below describe their status at the time of those experiments.

Historical [fixed-model replication](docs/cs_saf/replication_v1_report_2026_09_16.md):
**30 fresh GPU fits across five paired training trials, 135 tests and CPU checks PASS**.
ER passes response criteria in 5/5 trials (U 2/5, E 1/5) and improves null TV vs E
in 5/5. Its active TV is worse than E in 4/5, with mean change +.000047795, so the
registered accuracy gate FAILS and the conditional 90-fit prevalence expansion
was not started. ER and E both improve active accuracy vs internal U in all five
trials; this does not establish external-baseline superiority.
[Professor briefing](docs/cs_saf/professor_brief_replication_2026_09_16.md),
[evidence](docs/cs_saf/replication_v1_result.json), [status](docs/cs_saf/STATUS.md).

The [v4 discovery pilot](docs/cs_saf/v4_pilot_v1_report_2026_09_16.md) remains preserved:
its single-seed gate PASS motivated this replication. New trials are reported
separately; no post-hoc seed selection or threshold relaxation is used.

The previous [v3 result](docs/cs_saf/v3_pilot_v1_report_2026_09_16.md) is preserved:
R passed response criteria but failed accuracy against U. At that stage no later prevalence,
new seed, external baseline or held-out test had been run for ER; see the
completed follow-up above for the subsequent experiments.

Earlier v1/v2 state follows:

On `research/cs-saf`, **v2 passed implementation/CPU checks but failed its pilot
at pi=0.05 in the dependency-free rare context**. It reduced another null-cell
response but did not solve full null safety. V1's earlier failure at pi=0.25 is
preserved. Later stopped prevalences, five-seed confirmation and held-out
evaluation remain unexecuted.
See the [research status](docs/cs_saf/STATUS.md) and
[v2 pilot report](docs/cs_saf/v2_pilot_v1_report_2026_09_16.md).

A separate [three-objective diagnostic](docs/cs_saf/loss_control_v1_report_2026_09_16.md)
is complete: 82 tests and all CPU gates passed; six GPU fits reproduced the
earlier U/B results exactly. U/A/B all fail the rare-null response bound. The
added auxiliary and context balancing each increase that response, while
whole-route removal worsens factual repeat loss. No later prevalence study had started at that diagnostic stage.

The subsequent [fixed-checkpoint decomposition](docs/cs_saf/route_decomposition_v1_report_2026_09_16.md)
is complete: 92 tests, CPU/GPU checks, 12 existing snapshots, **zero new fits**.
Retaining the route's train-reference mean preserves useful history correction;
removing only gap variation improves rare-null validation loss slightly but
worsens train loss, and substantially harms active prediction. The registered
both-split null-harm hypothesis is not supported. A new trained solution remains
unproven. See the [model, preprocessing, related-work and claim audit](docs/cs_saf/research_claims_and_related_work_audit_2026_09_16.md)
for the contribution audit at that stage; the latest follow-up report updates
the external execution status.

This repository contains the reproducible research code for CoF-SeqGen and its
support-aligned autoregressive extension (SAF). The `main` branch records the
validated research base through the SAF v6 intervention preflight:

- train-support-aligned gap generation is strongly supported by repeated
  development experiments;
- a static-context codec defect affecting earlier dependency-routing runs has
  been corrected and regression-tested;
- the corrected scalar-gated v6 model failed its preregistered intervention
  preflight, so dependency-routing success is not claimed;
- held-out test data remains sealed.

The conference-oriented extension is developed separately on
`research/cs-saf`. It studies whether support-aligned generation can also
preserve rare, context-specific temporal transition mechanisms.

Raw datasets, model checkpoints, third-party repository clones, and large
runtime artifacts are intentionally excluded. Source contracts, acquisition
scripts, experiment configurations, tests, and compact reports are versioned.
Place locally obtained AMLSim files under `local_data/amlsim/` and Sparkov
files under `local_data/sparkov/`; their expected hashes remain pinned in the
acquisition configuration. The `local_data/` directory is ignored by Git.

The project builds on the TabDiff implementation below and retains its license
and attribution.

---

# TabDiff: a Mixed-type Diffusion Model for Tabular Data Generation

<p align="center">
  <a href="https://github.com/MinkaiXu/TabDiff/blob/main/LICENSE">
    <img alt="MIT License" src="https://img.shields.io/badge/License-MIT-yellow.svg">
  </a>
  <a href="https://openreview.net/forum?id=swvURjrt8z">
    <img alt="Openreview" src="https://img.shields.io/badge/review-OpenReview-blue">
  </a>
  <a href="https://arxiv.org/abs/2410.20626">
    <img alt="Paper URL" src="https://img.shields.io/badge/cs.LG-2410.20626-B31B1B.svg">
  </a>
</p>

<div align="center">
  <img src="images/tabdiff_demo.gif" alt="Model Logo" width="800" style="margin-left:'auto' margin-right:'auto' display:'block'"/>
  <p><em>Figure 1: Visualing the generative process of TabDiff. A high-quality version of this video can be found at <a href="images/tabdiff_demo.mp4" download>tabdiff_demo.mp4</a></em></p>
</div>

This repository provides the official implementation of TabDiff: a Mixed-type Diffusion Model for Tabular Data Generation (ICLR 2025).

## Latest Update
- [2025.04]：The categorical-heavy dataset **[Diabetes](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)** evaluated in the paper has now been released!
- [2025.02]：Our code is finally released! We have released part of the tested datasets. The rest will be released soon!

## Introduction

<div align="center">
  <img src="images/tabdiff_flowchart.jpg" alt="Model Logo" width="800" style="margin-left:'auto' margin-right:'auto' display:'block'"/>
  <p><em>Figure 2: The high-level schema of TabDiff</a></em></p>
</div>
TabDiff is a unified diffusion framework designed to model all muti-modal distributions of tabular data in a single model. Its key innovations include:  

1) Framing the joint diffusion process in continuous time,
2) A feature-wised learnable diffusion process that offsets the heterogeneity across different feature distributions,
3) Classifier-free guidance conditional generation for missing column value imputation. 

The schema of TabDiff is presented in the figure above. For more details, please refer to [our paper](https://arxiv.org/abs/2410.20626).


## Environment Setup

Create the main environment with [tabdiff.yaml](tabdiff.yaml). This environment will be used for all tasks except for the evaluation of additional data fidelity metrics (i.e., $\alpha$-precision and $\beta$-recall scores)

```
conda env create -f tabdiff.yaml
```

Create another environment with [synthcity.yaml](synthcity.yaml) to evaluate additional data fidelity metrics

```
conda env create -f synthcity.yaml
```

## Datasets Preparation

### Using the datasets experimented in the paper

Download raw datasets:

```
python download_dataset.py
```

Process datasets:

```
python process_dataset.py
```

### Using your own dataset

First, create a directory for your dataset in [./data](./data):
```
cd data
mkdir <NAME_OF_YOUR_DATASET>
```

Compile your raw tabular data in .csv format. **The first row should be the header** indicating the name of each column, and the remaining rows are records. After finishing these steps, place you data's csv file in the directory you just created and name it as <NAME_OF_YOUR_DATASET>.csv. 

Then, create <NAME_OF_YOUR_DATASET>.json in [./data/Info](./data/Info). Write this file with the metadata of your dataset, covering the following information:
```
{
    "name": "<NAME_OF_YOUR_DATASET>",
    "task_type": "[NAME_OF_TASK]", # binclass or regression
    "header": "infer",
    "column_names": null,
    "num_col_idx": [LIST],  # list of indices of numerical columns
    "cat_col_idx": [LIST],  # list of indices of categorical columns
    "target_col_idx": [list], # list of indices of the target columns (for MLE)
    "file_type": "csv",
    "data_path": "data/<NAME_OF_YOUR_DATASET>/<NAME_OF_YOUR_DATASET>.csv"
    "test_path": null,
}
```

### Important Notes When Creating the Info File
- The MLE evaluation and the imputation task (see later sections for details) assume that one column of your data is the regression or classification target. To enable these tasks, you will need to specify `target_col_idx`. If you don't need to evalute MLE, you can comment out the following line: https://github.com/MinkaiXu/TabDiff/blob/0c4fc3bbfa19046d36c5dce64628df52d5c73d15/tabdiff/main.py#L152
- The fields `target_col_idx`, `num_col_idx` and `cat_col_idx` must be multually exclusive—no column should appear in more than one of these lists. 
- Set the task_type to "regression" if the target column is numerical, or "binclass" if it is categorical.

Finally, run the following command to process your dataset:
```
python process_dataset.py --dataname <NAME_OF_YOUR_DATASET>
```

## Training TabDiff

To train an unconditional TabDiff model across the entire table, run

```
python main.py --dataname <NAME_OF_DATASET> --mode train
```

Current Options of ```<NAME_OF_DATASET>``` are: adult, default, shoppers, magic, beijing, news

Wanb logging is enabled by default. To disable it and log locally, add the ```--no_wandb``` flag.

To disable the learnable noise schedules, add the ```--non_learnable_schedule```. Please note that in order for the code to test/sample from such model properly, you need to add this flag for all commands below.

To specify your own experiment name, which will be used for logging and saving files, add ```--exp_name <your experiment name>```. This flag overwrites the default experiment name (learnable_schedule/non_learnable_schedule), so, similar to ```--non_learnable_schedule```, once added to training, you need to add it to all following commands as well.

## Sampling and Evaluating TabDiff (Density, MLE, C2ST)

To sample synthetic tables from trained TabDiff models and evaluate them, run
```
python main.py --dataname <NAME_OF_DATASET> --mode test --report --no_wandb
```

This will sample 20 synthetic tables randomly. Meanwhile, it will evaluate the density, mle, and c2st scores for each sample and report their average and standard deviation. The results will be printed out in the terminal, and the samples and detailed evaluation results will be placed in ./eval/report_runs/<EXP_NAME>/<NAME_OF_DATASET>/.

## Evaluating on Additional Fidelity Metrics ($\alpha$-precision and $\beta$-recall scores)
To evaluate TabDiff on the additional fidelity metrics ($\alpha$-precision and $\beta$-recall scores), you need to first make sure that you have already generated some samples by the previous commands. Then, you need to switch to the `synthcity` environment (as the synthcity packet used to compute those metrics conflicts with the main environment), by running
```
conda activate synthcity
```
Then, evaluate the metrics by running
```
python eval/eval_quality.py --dataname <NAME_OF_DATASET>
```

Similarly, the results will be printed out in the terminal and added to ./eval/report_runs/<EXP_NAME>/<NAME_OF_DATASET>/

## Evaluating Data Privacy (DCR score)
To evalute the privacy metric DCR score, you first need to retrain all the models, as the metric requires an equal split between the training and testing data (our initial splits employ a 90/10 ratio). To retrain with an equal split, run the training command but append `_dcr` to ```<NAME_OF_DATASET>```
```
python main.py --dataname <NAME_OF_DATASET>_dcr --mode train
```

Then, test the models on DCR with the same `_dcr` suffix
```
python main.py --dataname <NAME_OF_DATASET>_dcr --mode test --report --no_wandb
```



## Missing Value Imputation with Classifier-free Guidance (CFG)
Our current experiments only include imputing the target column. However, our implementation, located at ```sample_impute()``` in [unified_ctime_diffusion.py](./tabdiff/models/unified_ctime_diffusion.py), should support imputing multiple columns with different data types.

### Training Guidance Model
In order to enable classifier-free guidance (CFG), you need to first train an unconditional guidance model on the target column by running the training command with the `--y_only` flag
```
python main.py --dataname <NAME_OF_DATASET> --mode train --y_only
```

### Sampling Imputed Tables
With the trained guidance model, you can then impute the missing target column by running the testing command with the `--impute` flag
```
python main.py --dataname <NAME_OF_DATASET> --mode test --impute --no_wandb
```
This will, by default, randomly produce 50 imputed tables and save them to ./impute/<NAME_OF_DATASET>/<EXP_NAME>.

### Evaluating Imputation
You can then evaluate the imputation quality by running
```
python eval_impute.py --dataname <NAME_OF_DATASET>
```

## License

This work is licensed undeer the MIT License.

## Acknowledgement
This repo is built upon the previous work TabSyn's [[codebase]](https://github.com/amazon-science/tabsyn). Many thanks to Hengrui!

## Citation
Please consider citing our work if you find it helpful in your research!
```
@inproceedings{
shi2025tabdiff,
title={TabDiff: a Mixed-type Diffusion Model for Tabular Data Generation},
author={Juntong Shi and Minkai Xu and Harper Hua and Hengrui Zhang and Stefano Ermon and Jure Leskovec},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=swvURjrt8z}
}
```
## Contact
If you encounter any problem, please file an issue on this GitHub repo.

If you have any question regarding the paper, please contact Minkai at [minkai@stanford.edu](minkai@stanford.edu) or Juntong at [shisteve@usc.edu](shisteve@usc.edu).
