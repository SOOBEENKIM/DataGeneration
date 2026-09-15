# External empirical IID wall-cap diagnosis and source correction

## Scope and frozen evidence

This is a source-only diagnosis of the failed AMLSim `empirical_iid`
`attempt_001`. It does not authorize or perform a retry, external sampling,
validation, a GPU wave, or any test-split access. The job cap remains 7,200
seconds and the external protocol, selection rule, thresholds, data, seed,
conditioning plan, PAD/UNK contract, and empirical IID definition are
unchanged.

The following append-only inputs were inventoried before any source edit:

| Preserved object | Files | Bytes | Tree SHA-256 |
|---|---:|---:|---|
| AMLSim `empirical_iid/attempt_001` | 4 | 5,071 | `e0d2bf55ca9eaf6404294168a309b66d0c1c34ce63c247f15af853a4b37e7810` |
| Sparkov `empirical_iid/attempt_001` | 11 | 1,140,406 | `64abdd9ec13d52b25f8f9f79de15e6fd95d81bb321058a90d42b77e10b4a8592` |
| External validation authorization history | 5 | 70,976 | `281372d002df2af1c42dc941410cb25348ab875d078da396c77c2c867e4b7246` |

Critical AMLSim marker hashes are:

- `OWNERSHIP.json`: `851d5920f590c08ce869aef2e8e26d1cd9ec3574b69c8402e4e4f88a4ca43701`
- `RUNNING.json`: `6b66233dc1745de2a8261c7eb49e5001f4cecc9b576dc94f995103afd6558db9`
- `manifest.json`: `9e5f61ff5f3714fc2be0d8d40006179e5475507d9772c36cd54090cefb203784`
- `FAILED.json`: `57674e7ec91405fb53c963287a9d6cbfc273783a5db89826e65e722ef231131c`

The frozen bundle tree hashes remain
`cc8f4c9c6ff415ccd4f78ec193122100145e10084751b971dcc0ce5e705fa024`
for AMLSim and
`62fefda6207911104a9bff0ebb41d5edf44c20a4c78dccef5875d9768be6addc`
for Sparkov. Only manifest and summary metadata were read during diagnosis;
no frozen NPZ body was loaded.

## Reproduction and localization

AMLSim stopped at `7,200.113177566091` seconds with
`failure_class=wall_cap` and `runner_owned_child_terminated=true`. It contains
no `train_bootstrap_thresholds.json`, sampling plan, sample, metric, or
evaluation artifact. The worker computes the 1,000-replicate train-only
bootstrap before constructing or fitting the adapter and before loading the
validation array. Therefore the failure is localized before empirical IID
fit/sample and before validation.

Sparkov completed the same protocol in `1,088.9125865140231` seconds. Its
train receiver cardinality is 695; AMLSim's is 9,656, a 13.89-fold increase.
The train valid-row counts are comparable (898,168 and 914,756), so the
cardinality-dependent path is the discriminating scale factor.

The old receiver total-variation implementation first formed the union of
categories, then evaluated `(rows == category).sum()` once per category for
both inputs. It did not materialize one dense `[N, K]` array, but it performed
the equivalent repeated memory scan: time `O(N*K)` with an `O(N)` temporary
boolean vector allocated K times. For the dominant AMLSim Y=0 rows, this is on
the order of billions of comparisons per bootstrap replicate and trillions
over 1,000 replicates.

The entity bootstrap also has a secondary `O(E*S)` scan because it calls
`flatnonzero(inverse == selected_entity)` per selected entity. It is not the
receiver-cardinality-dependent defect and was deliberately not changed in
this minimal correction. The empirical IID sampler itself uses one class-wise
`rng.choice` over row-pool indices and vectorized indexed assignments. It has
neither a dense `[N, K]` probability matrix nor an event-wise Python loop.

## Synthetic measurements

All measurements used generated integer arrays only; no external bundle was
opened.

| Seam | Scale | Before | After | Interpretation |
|---|---|---:|---:|---|
| receiver TV | N=100,000, K=695 | 0.08246 s | 0.00729 s mean | 11.3x faster |
| receiver TV | N=100,000, K=9,656 | 1.01950 s | 0.01043 s mean | 97.7x faster |
| full public metric | 200,000 positions, K=9,656 | 2.26328 s | 0.06666 s | 34.0x faster |
| empirical IID sample | 200,000 positions, K=9,656 | already vectorized | 0.01255 s | sampler hypothesis refuted |
| entity bootstrap | S=30,349, E=636 | 0.01088 s | unchanged | secondary cost |
| entity bootstrap | S=30,349, E=6,657 | 0.08675 s | unchanged | secondary cost |

The corrected 200,000-position metric used 12,017,030 bytes of measured peak
temporary memory; the sampler used 9,886,942 bytes. Both are far below an
`N*K` allocation. Before correction, changing K from 695 to 9,656 increased
the isolated TV time 12.36-fold. After correction it increased only 1.43-fold.

## Minimal correction and semantic invariants

Receiver and gap total variation now compute unique value/count pairs once,
align the two frequency vectors on their union support, and take the same
one-half L1 distance. Complexity changes from `O(N*K)` repeated scans to
`O(N log N + K log K)` time and `O(N+K)` temporary memory. No sampler,
conditioning, bootstrap draw, model, config, threshold, budget, or artifact
code was changed.

The small asymmetric fixture continues to return exactly `1/3` receiver total
variation. The AMLSim-scale sampler fixture preserves complete sampled rows
(amount, gap, receiver), Y/L/mask, padding, deterministic seed behavior, and
train-only class-conditional row-pool sampling. The scale regression forbids
dense-memory or per-category repeated-scan behavior without encoding an
observed threshold or changing a scientific decision rule.

## Disposition

The exact wall-cap cause is the `O(N*K)` categorical fidelity calculation
inside every train-bootstrap replicate, not model capacity, GPU availability,
or empirical IID generation. The source-only correction removes that
cardinality amplification without increasing the cap. Actual AMLSim execution
has not been repeated, so `attempt_001` remains the official failed attempt.
A retry would require a new append-only attempt and separate authorization;
neither is created here.
