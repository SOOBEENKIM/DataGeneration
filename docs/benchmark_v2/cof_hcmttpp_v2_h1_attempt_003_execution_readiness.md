# CoF-HCMTTPP-v2 H1 attempt_003 execution-readiness gate

## Scope

This is a source-only CPU-synthetic integration gate. It is not an execution
authorization and does not make `attempt_003` runnable through the production
CLI. No AMLSim/Sparkov body, actual runtime root, GPU/CUDA path, model fit,
sampling, evaluation, internal test, or Sparkov fraudTest is accessed.

The production runner remains bound to the prior corrective `attempt_002`
configuration. A separate, explicit API admits only a scratch-root H1,
seed 4001, `attempt_003` readiness fixture. Production backend attachment to
`attempt_003` remains fail-closed unless the source-only readiness flag plus
the exact ownership and authorization identities are supplied.

## Integrated boundary

The single regression test
`tests/test_cof_hcmttpp_v2_execution_readiness.py` traverses, for AMLSim and
Sparkov scopes:

1. append-only parent ownership and manifest creation under pytest `tmp_path`;
2. a pickle round trip compatible with multiprocessing `spawn`;
3. exact parent ownership-path forwarding to the execution backend;
4. strict `H1AttemptStore.attach_existing` verification;
5. construction of the frozen H1 model on CPU;
6. train-only synthetic tail-state fitting with zero non-train rows;
7. a synthetic masked batch containing exact-zero, central-body,
   central-endpoint and tail gaps, both Y classes, and padding;
8. exactly one `event_nll -> backward -> AdamW.step` transition;
9. finite loss, gradients, parameters, open-endpoint RQS round trips, exact
   central endpoint inverse, and finite central/tail boundary CDF/density; and
10. fail-closed rejection after independently mutating ownership ID, dataset,
    candidate, seed, attempt, or authorization identity.

No clipping, fallback, redraw, or ownership relaxation is introduced. True
RQS domain violations continue to raise, and arbitrary attempts remain
unclaimable and unattachable.

## Preserved runtime evidence

These read-only tree digests were verified before the source-only gate work:

| Evidence | Tree SHA-256 | Files | Bytes |
|---|---|---:|---:|
| AMLSim H1 attempt_001 | `07bca9894bf6b619a4b4d6359773b2da2ac40a7112bebe70096712360671b389` | 13 | 106696 |
| Sparkov H1 attempt_001 | `309fbd56bc6c522b818e842b8c6082df5a0114b7f840537a0361164c744faf28` | 13 | 17706 |
| AMLSim H1 attempt_002 | `91a4670b6b8bebd47f289f9124d100d4d0f52e704ad02dac73ec23c70a767095` | 7 | 7006 |
| Sparkov H1 attempt_002 | `b4a900aa1e88fde24299d613fbdf90b67ba648493d904beaf9734ff348fce6a7` | 7 | 7018 |
| authorization history | `39cb1c14bb1ab1cf7bd6bd478b4d170fd4e7cf8cfa6ab2f3a5f6b142188814a4` | 6 | 41463 |
| launch logs | `5d60c50f6d081e9692e23fa76bfd7b5a994ea7f611e3ac1a570d5ab64c6e471f` | 4 | 11067 |

Only after this source state passes focused and repository-wide validation may
separate AMLSim and Sparkov `attempt_003` authorizations and launch plans be
prepared. None are created by this change.
