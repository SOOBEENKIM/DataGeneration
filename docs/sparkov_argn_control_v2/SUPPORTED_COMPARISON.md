# Common-support comparison: descriptive ranges across generation draws

Results use the same 143-customer validation cohort. Neural generators condition on static attributes; resampling controls assign customer keys but ignore those attributes. ARGN: two independent fits and two draws per fit. CPAR: one fit and two draws; four unsupported contexts excluded from EVERY model. Ranges below are descriptive, not confidence intervals. Resampling controls are diagnostics, not privacy-preserving generators.

| Setting | Fraud (%) | Merchant/category TV | Fraud category/amount TV | Fraud history TV | P(fraud after fraud) (%) | Fraud amount mean | Fraud amount log-W1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ARGN AUTO, native cap | 0.432–0.701 | 0.730–0.779 | 0.467–0.501 | 0.426–0.482 | 77.522–81.347 | 2709.030–3210.534 | 1.312–1.463 |
| ARGN AUTO, cap relaxed | 0.467–0.573 | 0.410–0.652 | 0.412–0.500 | 0.398–0.444 | 70.784–87.075 | 2318.417–3502.030 | 1.079–1.484 |
| ARGN AUTO + category first | 0.582–0.728 | 0.153–0.176 | 0.325–0.411 | 0.339–0.436 | 77.588–88.812 | 3102.497–3633.329 | 0.965–1.243 |
| ARGN AUTO + fraud first | 0.251–0.375 | 0.411–0.655 | 0.412–0.472 | 0.452–0.509 | 68.597–86.486 | 2640.490–4026.087 | 1.078–1.479 |
| ARGN gap DIGIT, cap relaxed | 0.760–0.943 | 0.658–0.854 | 0.434–0.592 | 0.342–0.402 | 83.911–93.319 | 2006.444–3119.250 | 1.091–1.434 |
| ARGN gap DIGIT + category first | 0.885–1.182 | 0.208–0.342 | 0.374–0.496 | 0.353–0.356 | 88.427–93.051 | 2598.384–3122.806 | 0.907–1.193 |
| ARGN gap+amount DIGIT, cap relaxed | 0.497–2.785 | 0.630–0.765 | 0.437–0.521 | 0.481–0.588 | 85.145–97.686 | 373.895–539.155 | 0.187–0.588 |
| ARGN gap+amount DIGIT + category first | 1.006–4.205 | 0.170–0.279 | 0.386–0.422 | 0.534–0.628 | 93.354–98.323 | 545.359–650.879 | 0.236–0.408 |
| CPAR, pinned implementation | 1.387–1.441 | 0.920–0.921 | 0.881–0.900 | 0.742–0.759 | 2.024–2.881 | 68.481–70.416 | 2.074–2.145 |
| Row resampling | 0.564–0.605 | 0.041–0.042 | 0.063–0.090 | 0.194–0.220 | 0.660–0.794 | 525.899–540.906 | 0.050–0.069 |
| First-order resampling | 0.500–0.535 | 0.041–0.043 | 0.080–0.088 | 0.206–0.213 | 89.485–89.794 | 509.216–556.291 | 0.058–0.104 |

Validation reference: fraud 0.637584%; fraud amount mean 518.550951. TV and Wasserstein distances are lower-is-better but must be interpreted against training-customer resampling variation, support and raw-scale checks.

Full metrics, generated event counts and invalid values: `supported_metrics.csv`, `supported_numeric_metrics.csv`. Conditional support and customer-bootstrap intervals: `supported_conditional_rates_customer_bootstrap.csv`.
