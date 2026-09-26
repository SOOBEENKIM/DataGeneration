# Primary comparison: descriptive ranges across generation draws

Results use the same 147-customer validation cohort. Neural generators condition on static attributes; resampling controls assign customer keys but ignore those attributes. ARGN: two independent fits and two draws per fit. CPAR is reported separately on the 143-customer common-support cohort. Ranges below are descriptive, not confidence intervals. Resampling controls are diagnostics, not privacy-preserving generators.

| Setting | Fraud (%) | Merchant/category TV | Fraud category/amount TV | Fraud history TV | P(fraud after fraud) (%) | Fraud amount mean | Fraud amount log-W1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ARGN AUTO, native cap | 0.443–0.683 | 0.730–0.779 | 0.455–0.496 | 0.424–0.466 | 77.195–82.181 | 2698.773–3412.208 | 1.308–1.474 |
| ARGN AUTO, cap relaxed | 0.448–0.599 | 0.410–0.651 | 0.415–0.502 | 0.394–0.444 | 70.093–87.432 | 2288.468–3558.204 | 1.068–1.504 |
| ARGN AUTO + category first | 0.554–0.718 | 0.153–0.176 | 0.323–0.413 | 0.339–0.433 | 77.075–88.934 | 3093.765–3588.450 | 0.964–1.254 |
| ARGN AUTO + fraud first | 0.242–0.387 | 0.411–0.654 | 0.401–0.481 | 0.438–0.505 | 67.320–86.701 | 2585.348–4126.431 | 1.071–1.495 |
| ARGN gap DIGIT, cap relaxed | 0.744–0.968 | 0.658–0.854 | 0.433–0.594 | 0.342–0.402 | 83.608–93.358 | 1985.310–3140.241 | 1.110–1.455 |
| ARGN gap DIGIT + category first | 0.894–1.191 | 0.208–0.342 | 0.369–0.495 | 0.349–0.358 | 88.373–93.373 | 2591.191–3132.569 | 0.907–1.207 |
| ARGN gap+amount DIGIT, cap relaxed | 0.487–2.825 | 0.630–0.765 | 0.440–0.525 | 0.475–0.587 | 84.834–97.655 | 371.442–538.881 | 0.180–0.609 |
| ARGN gap+amount DIGIT + category first | 0.994–4.165 | 0.171–0.279 | 0.386–0.423 | 0.533–0.628 | 93.399–98.269 | 545.913–650.466 | 0.237–0.398 |
| Row resampling | 0.563–0.605 | 0.041–0.042 | 0.062–0.093 | 0.200–0.225 | 0.651–0.850 | 524.618–541.224 | 0.055–0.064 |
| First-order resampling | 0.509–0.542 | 0.041–0.042 | 0.073–0.092 | 0.196–0.206 | 89.691–89.981 | 508.371–554.394 | 0.058–0.092 |

Validation reference: fraud 0.643831%; fraud amount mean 521.584904. TV and Wasserstein distances are lower-is-better but must be interpreted against training-customer resampling variation, support and raw-scale checks.

Full metrics, generated event counts and invalid values: `metrics.csv`, `numeric_metrics.csv`. Conditional support and customer-bootstrap intervals: `conditional_rates_customer_bootstrap.csv`.
