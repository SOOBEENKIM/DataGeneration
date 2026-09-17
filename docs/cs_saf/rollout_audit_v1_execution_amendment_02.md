# Technical precision amendment 02

Before restarting the scientific jobs after amendment 01, the additional GPU functional gate caught a numerical discrepancy between streamed GRU inference and full-sequence re-evaluation on the same six functional input sequences. With cuDNN TF32 enabled the maximum repeat-probability discrepancy was OBS 1.0555e-4, QUANT 8.7019e-5, GAP 6.1415e-5, FULL 6.2800e-5. With TF32 disabled the corresponding maxima were 9.2086e-8, 1.3738e-7, 1.0382e-7, 1.1271e-7, below the originally registered 2e-6 tolerance.

Disable both cuDNN and matrix-multiplication TF32 for the complete diagnostic run. Keep float32 learned parameters/inputs, float64 oracle/TV arithmetic, deterministic algorithms, the same checkpoints, input panels, sampling tapes, scientific contrasts and original numerical tolerances. Add a committed GPU gate covering U/E/L003 and all four rollout modes; it must pass along with the CPU gate on the exact execution source. Legacy full-validation TV and original generation metric reproduction remain mandatory at their registered tolerances; do not widen them.

No complete scientific checkpoint job had finished; no comparative scientific outcomes were inspected or used for this precision choice. This amendment fixes arithmetic consistency, not model behavior through training, a scientific endpoint, or a selection rule.
