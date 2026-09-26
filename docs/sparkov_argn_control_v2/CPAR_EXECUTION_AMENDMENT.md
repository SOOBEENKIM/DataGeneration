# CPU execution amendment

The first native full-sequence CPAR attempt reached its initial loss 4.58142853
at 91.4 seconds, then spent over four minutes in the first backward pass without
an optimizer update. That attempt was stopped and preserved under `attempts`.
No data, segments, epochs, objective, optimizer, model size or stopping criterion
was changed. The new attempt starts from the original seed.

`experiments/sparkov_cpar_dense.py` replaces the differentiable packed/padded
roundtrips with dense full-sequence GRU evaluation and an explicit padding mask.
Every valid prefix is mathematically identical; padded outputs are zeroed as in
the native packed output. The pinned native continuous-target alignment and loss
normalization are preserved. Float32 and float64 tests compare outputs, full loss
and gradients of every network parameter with variable sequence lengths. Three
tests pass, in addition to the existing loss accelerator's six CPU tests.
Small floating-point differences can compound during training; this is numerical
equivalence, not a claim of bit-identical fitted weights. The installed package
is unchanged and class methods are restored after fitting.

The dense full-batch attempt still took approximately 90 seconds per epoch and
was preserved as a second stopped attempt, not used for quality comparisons.
The final execution sorts customers by length and accumulates gradients over
memory chunks of at most 64 customers. It still makes exactly ONE Adam update
per epoch after all 688 customers; there is no stochastic minibatch training or
sequence truncation. Numeric targets retain the ORIGINAL global padding offset,
even when a chunk has a smaller maximum length. The global denominator remains
688 squared times the number of modeled fields. Tests also compare accumulated
loss and every parameter gradient against native full-batch evaluation, including
chunks with shorter maxima and a smaller last chunk. This avoids silently fixing
the native numeric-target convention. Final launch fixes PYTHONHASHSEED as well
as Python/NumPy/Torch seeds; previous speed attempts are not paired quality runs.
