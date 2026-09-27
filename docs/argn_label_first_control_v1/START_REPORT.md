# Label-first control started, 2026-09-27

At 17:19:34 KST, fresh event-weighted label-first fits started on GPU 2
(seed 20260930, PID 1290079) and GPU 3 (seed 20261001, PID 1290086).
The first optimizer step was logged at 17:19:41–42. Observed GPU utilization was
76% / 71%, with 2.7 / 3.1 GiB total device memory. Model parameter count is
4,815,662; optimization/internal validation customer counts remain 619/69.

Four fresh fits and eight native free-generation sets are registered. After each
fit, the worker generates two sets, evaluates them and runs the frozen real-past
transition diagnostic. The dispatcher then rechecks idle GPU 2/3 before starting
the customer-weighted order controls. It creates a partial report after each job
and a final report after all four. No new fit was complete at this snapshot.

The estimated first-pair turnaround is 20–30 minutes from launch and the four-fit
study about 40–60 minutes, based on earlier fits and generation durations. These
are estimates; native early stopping controls completion, with the original
60-epoch / 180-minute per-fit upper bounds retained.

Validation: 14 checks passed (13 state/order/loss/evaluation checks and one GPU
process-classification check). Tests established identical initial weights,
unchanged capacity, no current-field leakage into the first label, teacher and
stepwise parity, native sequential loss dispatch and state filtering.

The zero-fit initial queue attempt and MPS guard correction are documented in
[LAUNCH_AMENDMENT.md](LAUNCH_AMENDMENT.md). No foreign process was stopped.
Original frozen studies, checkpoints and generated datasets are unchanged.
The final test event attributes have not been opened.

Live local state: `artifacts/argn_label_first_control_v1/DISPATCH_STATUS.json`;
per-job `queue_*.json`; logs under `artifacts/argn_label_first_control_v1/logs`.
`RESULTS.md` and aggregate CSV files update after completed jobs; an uncompleted
study or its estimate must not be described as a result or architectural gain.
