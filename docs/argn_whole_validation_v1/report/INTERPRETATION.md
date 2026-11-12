# Fresh whole-pipeline validation

| arm                      |   normal_gap_seconds_log_w1 |   onset_personal_gap_w1 |   fraud_gap_seconds_log_w1 |   normal_stay_personal_gap_w1 |   fraud_ratio_log_w1 |   normal_amount_log_w1 |   equal_customer_normal_gap_w1 |   class_0_merchant_tv |   median_unique_merchants_per_customer |   early_time |   time50_99 |   time100_199 |   time200_499 |   time500_1023 |   late1024_time |
|:-------------------------|----------------------------:|------------------------:|---------------------------:|------------------------------:|---------------------:|-----------------------:|-------------------------------:|----------------------:|---------------------------------------:|-------------:|------------:|--------------:|--------------:|---------------:|----------------:|
| history_calibration_only |                   0.0945153 |                0.130938 |                   0.136094 |                     0.0636288 |            0.0913194 |              0.0824535 |                       0.119108 |             0.0713429 |                                548     |     0.111278 |    0.119229 |     0.110355  |      0.138606 |       0.13491  |        0.117013 |
| history_only             |                   0.125798  |                0.135416 |                   0.140115 |                     0.0617198 |            0.0939432 |              0.0834171 |                       0.233232 |             0.0713658 |                                547.833 |     0.313952 |    0.256406 |     0.195098  |      0.201316 |       0.105454 |        0.150695 |
| history_position6        |                   0.0976926 |                0.132769 |                   0.134403 |                     0.0609321 |            0.095687  |              0.0816244 |                       0.114768 |             0.0713224 |                                547     |     0.106985 |    0.115787 |     0.0999695 |      0.120909 |       0.137938 |        0.121368 |

Independent fits: 3 complete pipelines; 2 draws per arm. Same existing development cohort; test untouched.

{
  "normal_relative_improvement": 0.032524312193151106,
  "onset_delta": -0.0018303642755871374,
  "lower_normal_independent_fits": 3,
  "primary_descriptive_screen": false,
  "promotion": false,
  "novelty_established": false
}

Costs remain explicit in FIT_MEANS / PAIRED_FIT_EFFECTS. This does not establish newest-generator superiority or novelty. CPAR and MeLD have separate completion records.