# Benchmark v2 preflight

- Timestamp: 2026-07-28T17:22:49+09:00
- Repository root: `<REPO_ROOT>`
- Branch: `cof-seqgen`
- Initial commit: `5ecdb3356261aea72716cc9a779f31d7ad083bf4`
- Initial status:

```text
 D images/tabdiff_demo.gif
 D images/tabdiff_demo.mp4
 D images/tabdiff_flowchart.jpg
?? IMPLEMENTATION_DIRECTIVE_V2.md
?? REPO_MAP.md
?? figs/
?? logs/
?? models/
?? results/
?? scripts/
?? tests/
```

The three image deletions and all untracked content above predate v2 work and are
preserved as user-owned state. No cleanup or restoration was performed.

## GPU

```text
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
```

The project environment also reports `torch.cuda.is_available() == False` and
zero CUDA devices. GPU-only smoke work therefore requires the workstation
driver/device to become available.

## Python/CUDA environment

The shell has no `python` command. System `python3` is 3.8.10 with NumPy 1.23.5
and scikit-learn 1.3.2, but does not contain PyTorch.

The legacy run scripts identify the intended environment:
`<COFSEQ_PYTHON>`.

```text
Python 3.10.20
numpy 1.26.4
scikit-learn 1.7.2
torch 2.1.2+cu121
torch CUDA build 12.1
CUDA available False
CUDA device count 0
cuDNN 8902
```

CTGAN, PyYAML, SciPy, and pandas are installed in that environment.
