"""Continue generation on another GPU after a fully completed, verified fit.

This does not resume an interrupted training trajectory. The original fit and
registered runner are unchanged; each independent generation is seeded anew.
"""
import argparse
import logging
import os
import time

import run_argn_event_weight_control as original


def run(fs, device):
    original.check_manifest()
    folder = original.OUT / "runs" / f"B_event_weighted_{fs}"
    fit = original.json.loads((folder / "FIT.json").read_text())
    assert fit["source_hashes"] == original.code_hashes()
    ws = original.Workspace(folder / "workspace")
    weight_hash = fit["weights_sha256"]
    assert original.digest(ws.model_tabular_weights_path) == weight_hash
    status = original.OUT / f"queue_{fs}.json"
    original.write(folder / "GENERATION_RELOCATION.json", dict(
        pid=os.getpid(), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        completed_fit_sha256=original.digest(folder / "FIT.json"),
        runner_sha256=original.digest(__file__), training_resumed=False))
    try:
        original.write(status, dict(stage="generating", seed=fs, pid=os.getpid()))
        context = original.pd.read_parquet(original.PARENT / "prepared/validation_context.parquet")
        with original.engine_adapter(ws.tgt_stats.read(), enabled=False):
            for gs in original.CONTROL["generation_seeds"]:
                metadata = folder / f"generation_validation_{gs}.json"
                dest = folder / f"generated_validation_{gs}.parquet"
                if metadata.exists():
                    info = original.json.loads(metadata.read_text())
                    assert info["sha256"] == original.digest(dest)
                    assert info["weights_sha256"] == weight_hash
                    continue
                assert not dest.exists(), "Archive incomplete generation before continuing"
                assert not (folder / f"native_validation_{gs}.parquet").exists()
                original.seed(gs)
                started = time.monotonic()
                original.generate(ctx_data=context, device=device, workspace_dir=folder / "workspace")
                raw = original.pd.read_parquet(folder / "workspace/SyntheticData")
                raw.to_parquet(folder / f"native_validation_{gs}.parquet", index=False)
                frame = raw.rename(columns={"customer_id": "entity_id"}).copy()
                frame["event_index"] = frame.groupby("entity_id", sort=False).cumcount()
                frame.loc[frame.event_index.eq(0), "gap"] = original.np.nan
                frame.to_parquet(dest, index=False)
                assert original.digest(ws.model_tabular_weights_path) == weight_hash
                original.write(metadata, dict(seconds=time.monotonic()-started,
                    events=len(frame), customers=frame.entity_id.nunique(), sha256=original.digest(dest),
                    weights_sha256=weight_hash, native_generated_length=True, labels_generated_jointly=True,
                    no_prevalence_or_run_length_repair=True))
                print(f"CONTROL_GENERATION_COMPLETE {fs} {gs}", flush=True)
        original.write(status, dict(stage="evaluating", seed=fs, pid=os.getpid()))
        original.evaluate(folder, fs)
        original.write(status, dict(stage="complete", seed=fs, pid=os.getpid(), generated_datasets=2,
            source_hashes=fit["source_hashes"], test_events_read=False))
        print(f"CONTROL_COMPLETE {fs}", flush=True)
    except Exception as error:
        original.write(status, dict(stage="failed", seed=fs, pid=os.getpid(), error=repr(error)))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s", force=True)
    run(args.seed, args.device)
