"""One-factor objective control; native sequence-length loss stays customer weighted."""
from contextlib import contextmanager
import inspect
import hashlib
from benchmarks.argn_state_adapter import engine_adapter


@contextmanager
def event_weighted_adapter(stats, mean_training_length, source_dir=None):
    assert mean_training_length > 0
    import mostlyai.engine._tabular.training as native
    with engine_adapter(stats, enabled=False, source_dir=source_dir) as train:
        source = inspect.getsource(native._calculate_sample_losses)
        old = "            losses_by_column.append(masked_loss)"
        assert source.count(old) == 1
        new = (
            "            if col not in slen_cols and col not in sidx_cols and col not in sdec_cols:\n"
            f"                masked_loss = masked_loss * slen_mask.sum(dim=1) / {float(mean_training_length)!r}\n"
            + old
        )
        modified = source.replace(old, new)
        namespace = dict(train.__globals__)
        exec(compile(modified, "registered_event_weighted_loss", "exec"), namespace)
        train.__globals__["_calculate_sample_losses"] = namespace["_calculate_sample_losses"]
        if source_dir:
            (source_dir / "event_weighted_loss_source.py").write_text(modified)
        yield train
