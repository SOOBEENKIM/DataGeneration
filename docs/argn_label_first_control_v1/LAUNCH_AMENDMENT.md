# Pre-training GPU dispatch correction, 2026-09-27

The first dispatcher started at 08:16 UTC but launched **zero fits**: its
no-compute-process check treated the host's 28 MiB `nvidia-cuda-mps-server`
resident daemon as an active job on otherwise idle GPU 2/3. GPU 0/1 have actual
20+ GiB VLLM jobs and remain excluded.

Stopped only our dispatcher PID 1288966, preserved its manifest, logs, queue
state, dispatcher source and protocol under
`artifacts/argn_label_first_control_v1/attempts/idle_mps_guard_before_any_fit`.
No model process was stopped and no training result exists for this attempt.

Amend the operational guard: ignore only a named MPS daemon with at most 32 MiB
memory, while retaining two observations plus immediate pre-launch checks of
total GPU memory below 500 MiB, utilization at most 5%, and absence of every
other compute process. Unknown memory and larger MPS allocations block launch.
This does not change any model, data, objective, budget or evaluation condition.
A dispatch test covers other active processes, tiny Python workloads and unknown
memory. Prepare a new manifest containing the updated source hashes; keep the
original protocol unchanged and include this amendment in the new hash manifest.
