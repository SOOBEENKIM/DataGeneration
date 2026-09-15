# CoF-HCMTTPP-v2 H1 RQS inverse correction

## Scope and frozen evidence

This is a source-only mathematical correction. No GPU/CUDA access, external
data-body read, fit, sample, evaluation, authorization, launch plan, or retry
was performed. AMLSim and Sparkov H1 `attempt_001` remain append-only FAILED
evidence from source commit
`70481e2e4b612002fe7d3d074b9b67294869a52b`.

| Evidence | SHA-256 or tree inventory |
|---|---|
| AMLSim attempt tree | `07bca9894bf6b619a4b4d6359773b2da2ac40a7112bebe70096712360671b389` (13 files, 106696 bytes) |
| Sparkov attempt tree | `309fbd56bc6c522b818e842b8c6082df5a0114b7f840537a0361164c744faf28` (13 files, 17706 bytes) |
| AMLSim `FAILED.json` | `5a65749584bb9f4a584bd75ed74eb7d10e0547c66c7ffd3c3f9da00431712042` |
| Sparkov `FAILED.json` | `94af493b9dd3f3ddb2847f3cc7d6375ef744d713caf0ad83276d386ba53ed8a3` |
| AMLSim artifact index / checksum manifest | `79df11abbfe6d354750c855b556c054d39d8600314807a68bf2c21cc1668ee61` / `42aab98b620657596a54e8c1aebc287ba03c61d8775e226d9967e6ffbfef3d6d` |
| Sparkov artifact index / checksum manifest | `79df11abbfe6d354750c855b556c054d39d8600314807a68bf2c21cc1668ee61` / `feca6df638f70c549b67b8562744ceae041fc0d4779e787e2023beb41bfcef01` |
| AMLSim worker failure marker | `39f2cbbea92494ae53f7aeaca24d42da6e4452c7725ef10ef4227728f336b9ec` |
| Sparkov worker failure marker | `456f5df3ecde0591c26bbaac8917bafd028678125dc7056462f8306da9d8d04d` |
| AMLSim authorization / launch log | `7dd78b9334a50be39b88328dbf5ff5dfdebbf9c8453c8e58d72fe9c3e3f60840` / `a8caf0ea264e8aa092a2f83fa8c33f3f2d9b591189ac99eb4b2e1e2440edc000` |
| Sparkov authorization / launch log | `21cfac1cd098bb2cd7154c7f051f9338a4e349b834e6f4ccbf22d9d5ab8719fb` / `1d1388525c28d7d0962dccc3ca01a183f5ee1c78bfc32d948ff91b2a3a92f6fe` |

Both checksum manifests were independently verified against every indexed
file. The terminal markers bind those manifests and indexes. AMLSim failed
after 2.6282053909962997 seconds and Sparkov after 2.5463845540070906
seconds, at the identical call chain `event_nll -> positive_log_prob_u ->
central_inverse`. Neither run produced a checkpoint or a performance/gate
result.

## Hypothesis audit

### 1. Quadratic inverse numerical instability — SUPPORTED

A CPU-only reproduction used the two frozen thresholds, 2.3978952727983707
and 11.30861593474973, with finite initial H1 parameters. Before correction,
all 100 tested initialization seeds failed when the valid input was exactly
`u_tail`. In the seed-0 diagnostic the computed root ranged from
0.9999722838401794 to 1.00002920627594, so a mathematically valid endpoint
was rejected by the strict `[0,1]` guard.

After the endpoint geometry was pinned, the unmodified float32 quadratic
formula still produced roots up to 1.0000025033950806 and rejected 177 of
512 exact endpoints. Therefore cumulative geometry drift was not the only
cause. Exact bin endpoints have known analytic roots and must not be routed
through a rounded quadratic calculation.

### 2. Central/tail routing or selected-bin domain error — SUPPORTED only at the geometry boundary

The route predicate `u <= u_tail` and the bin-index rule select the intended
central and last-bin routes. The defect was that the former geometry used a
float32 cumulative sum of all heights without pinning its final knot. Its
last knot differed from `u_tail` by as much as ±4.76837158203125e-7, and the
last-bin local coordinate exceeded its computed height by as much as
4.544854164123535e-7. Thus a globally valid `u_tail` could appear locally
outside the selected bin. There is no evidence of a broader central/tail or
bin-index routing error after exact endpoint construction.

### 3. Finite but effectively degenerate parameterization — REFUTED for the observed failure

The reproduced initial state was finite. Its selected last-bin height was at
least 0.017107024788856506 and its secant `delta` at least
0.2737123966217041, far above the frozen positive minima. An explicit
extreme-but-valid minimum-height/minimum-derivative fixture and a near-linear
fixture both round-trip after the endpoint and arithmetic-precision
correction. Non-finite geometry, nonpositive derived heights, and genuine
inputs outside `(0,u_tail]` remain fail-closed.

## Minimal correction

`H1HurdleRQSGapDecoder._spline_geometry` now constructs interior cumulative
knots from the first 15 raw heights, pins the outer knots to exact zero and
exact `u_tail`, and derives all 16 effective heights from adjacent knots. It
raises `InvalidH1GapStateError` for non-finite knots or nonpositive heights.

RQS geometry, forward, and inverse calculations promote float16, bfloat16,
and float32 model parameters to float64 arithmetic. This preserves the
invertibility of the frozen minimum-height bins without changing their
parameterization. `central_inverse` retains the same stable quadratic formula
for every strictly interior value and uses the exact analytic solutions
`theta=0` when the local ordinate is the lower bin endpoint and `theta=1`
when it is the upper endpoint. This is neither clipping nor fallback: those
are the exact roots of the same RQS equation. The existing discriminant,
root-domain, reconstruction, Jacobian, and non-finite guards remain active.

The correction changes no architecture, candidate, objective, threshold,
seed, optimizer, training budget, SamplingPlan, C0/C1 reference, or H1 gate.
The failed attempts and their authorizations are not reusable under the
corrected source.

## Corrected source identity

- model SHA-256: `f8b73074934b4bb9e2ea46de5cfdd3e35b447b23c9a1f8302b87c4182c0dcbf6`
- source config SHA-256: `f623c2c6db5843a02404da4ab3824fa1464cbc1380659dbd717e512315aee12e`
- runner config SHA-256: `37136d7b2cdf705c86ddeb4ed1d17fb0252c633a7f9203331d30695e2c8a2d53`
- execution-relevant source SHA-256: `0572f528a043aae6068bfefc428a63c86227c166a05e2be7895a94243c4a3b9d`

The corrective commit SHA is reported after commit creation; no execution
authorization or `attempt_002` plan is part of this correction.
