# Frozen representation preparation — 2026-09-29

This preparation is authorized by the user's fixed-init_only 2x2 experiment.
It does not fit any parameters, generate evaluation samples, or select a model.
Read the existing optimization/internal metadata and native encoded inputs;
capture the native context/strict-past history representation immediately before
the first gap digit. Parent seed 20260930 is frozen. Check every native gap-digit
logit against its cloned head and exact record/event/label mapping. Write decoded
gap versus metadata mismatches explicitly; do not change either source.
The native forward uses real past fields for this cache. Subsequent conditional
time-rollout training must not call this a full native free rollout: the time
clock can be recursive while this backbone representation stays teacher-forced.
Final comparison must use the existing native generation integration so that
its history is updated with all generated fields. Test events are not read.
No original files, active model, commit, or push are changed.
