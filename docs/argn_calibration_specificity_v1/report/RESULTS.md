# Frozen calibration specificity results

18 new coefficient fits;36 new native generations;36 frozen outputs reused. Three existing time-fit seeds, two frozen parents, two draws. Lower errors are better.

| arm | normal_gap_seconds_log_w1 | onset_personal_gap_w1 | normal_stay_personal_gap_w1 | early_time | time100_199 | late1024_time |
|---|---|---|---|---|---|---|
| history_only | 0.123082 | 0.099052 | 0.033969 | 0.316533 | 0.182406 | 0.082123 |
| history_global | 0.069761 | 0.079037 | 0.034294 | 0.176931 | 0.07984 | 0.13371 |
| history_position3 | 0.060087 | 0.079722 | 0.033693 | 0.104139 | 0.090224 | 0.089703 |
| history_position6 | 0.058901 | 0.083316 | 0.033655 | 0.093152 | 0.087032 | 0.077758 |
| history_calibration_only | 0.052605 | 0.079223 | 0.036049 | 0.10501 | 0.089122 | 0.077265 |
| init_only | 0.063958 | 0.120572 | 0.006468 | 0.084596 | 0.099827 | 0.074017 |

| candidate | reference | normal_effect | onset_effect | costs | descriptive_screen | normal_better_fits | onset_better_fits | repeating_screen | normal_personal_delta |
|---|---|---|---|---|---|---|---|---|---|
| history_calibration_only | history_global | True | False |  | True | 3 | 2 | True | 0.0017548094372739978 |
| history_calibration_only | history_position3 | True | False |  | True | 3 | 2 | True | 0.0023556763059638178 |
| history_calibration_only | history_position6 | True | False |  | True | 3 | 3 | True | 0.002393582459343825 |
| history_calibration_only | history_only | True | True |  | True | 3 | 3 | True | 0.0020796231698937406 |
| history_calibration_only | init_only | True | True | early_time | False | 3 | 3 | False | 0.02958025289981848 |
| history_global | history_only | True | True | time500_1023,late1024_time | False | 3 | 3 | False | 0.00032481373261974283 |
| history_position3 | history_only | True | True |  | True | 3 | 3 | True | -0.00027605313607007714 |
| history_position6 | history_only | True | True |  | True | 3 | 3 | True | -0.0003139592894500845 |

All paired fit, parent and draw effects are saved. These are repeatedly reused development data, not untouched test or independent whole-generator refits.
Passing a control screen would not establish novelty or SOTA superiority. Failure stops the present personal-speed necessity claim; it is not proof that no useful research is possible.
No model replacement, commit or push.
