"""Adaptive frozen-checkpoint probes of the first-event failure."""
import json
import numpy as np
import pandas as pd
import torch
from run_argn_state_first import ROOT, OUT, SOURCE, CFG, digest, write
from benchmarks.argn_past_state import STATE_COLUMN, FEATURES, PastState
from benchmarks.argn_state_adapter import model_class, FullHistoryCollator
from mostlyai.engine._workspace import Workspace
from mostlyai.engine._common import get_cardinalities, get_ctx_sequence_length, encode_slen_sidx_sdec, SLEN_SUB_COLUMN_PREFIX
from mostlyai.engine._tabular.encoding import encode_df, flatten_frame, _enrich_slen_sidx_sdec
from mostlyai.engine._tabular.common import load_model_weights

DOCS = ROOT / "docs/argn_state_first_v1/evaluation"
CORE = ["gap", "receiver_or_mark", "amount_or_numeric_value", "category", "event_is_fraud"]


def objective_audit():
    workspace = Workspace(OUT / "prepared/workspace")
    codec = PastState(workspace.tgt_stats.read())
    frame = pd.read_parquet(workspace.encoded_data_trn.fetch_all()[0])
    key = codec.prefixes["event_is_fraud"] + "__cat"
    result = []
    groups = [("all", 0, np.inf), ("first_event", 0, 1), ("first_10_events", 0, 10),
              ("positions_10_99", 10, 100), ("positions_100_plus", 100, np.inf)]
    for name, lower, upper in groups:
        events = frauds = 0
        loss_mass = fraud_loss_mass = 0.
        customers = 0
        for sequence in frame[key]:
            pos = np.arange(len(sequence))
            selected = np.asarray(sequence)[(pos >= lower) & (pos < upper)]
            if len(selected): customers += 1
            f = int(np.sum(selected == codec.codes["1"]))
            events += len(selected); frauds += f
            loss_mass += len(selected) / len(sequence)
            fraud_loss_mass += f / len(sequence)
        result.append(dict(position_group=name, customers=customers, events=events, frauds=frauds,
                           ordinary_event_fraud_share=frauds / events,
                           native_loss_fraud_share=fraud_loss_mass / loss_mass,
                           transaction_loss_mass=loss_mass, fraud_transaction_loss_mass=fraud_loss_mass))
    pd.DataFrame(result).to_csv(DOCS / "native_loss_weight_audit.csv", index=False)


def first_batch(raw, context, ts, cs):
    lengths = raw.groupby("customer_id", sort=False).size()
    first = raw.groupby("customer_id", sort=False).head(1).copy()
    first = first.set_index("customer_id").loc[context.customer_id].reset_index()
    encoded, _, key = encode_df(first[["customer_id", *CORE]], ts, tgt_context_key="customer_id", n_jobs=1)
    maximum = ts["seq_len"]["max"]
    encoded = flatten_frame(_enrich_slen_sidx_sdec(encoded, key, maximum), key)
    true_lengths = pd.Series([lengths.loc[k] for k in encoded[key]])
    tokens = encode_slen_sidx_sdec(true_lengths, max_seq_len=maximum, prefix=SLEN_SUB_COLUMN_PREFIX)
    for col in tokens:
        encoded[col] = [[int(v)] for v in tokens[col]]
    static, pk, _ = encode_df(context, cs, ctx_primary_key="customer_id", n_jobs=1)
    frame = encoded.merge(static, left_on=key, right_on=pk, validate="one_to_one").drop(columns=[key, pk])
    # Reindexing above fixes corresponding customer IDs before any interventions.
    records = frame.to_dict("records")
    for row in records:
        row[STATE_COLUMN] = np.zeros((1, len(FEATURES)), dtype=np.float32)
    return FullHistoryCollator(True, None, torch.device("cpu"))(records)


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    assert not (DOCS / "first_event_probes.csv").exists()
    objective_audit()
    context = pd.read_parquet(SOURCE / "prepared/validation_context.parquet").sort_values("customer_id").reset_index(drop=True)
    real = pd.read_parquet(SOURCE / "prepared/validation.parquet").rename(columns={"entity_id": "customer_id"})
    real.gap = real.gap.fillna(0)
    rows, inputs = [], {}
    for arm in CFG["arms"]:
        for fs in CFG["fit_seeds"]:
            folder = OUT / "runs" / f"{arm}_{fs}"
            ws = Workspace(folder / "workspace")
            ts, cs = ws.tgt_stats.read(), ws.ctx_stats.read()
            codec = PastState(ts)
            model = model_class(ts, arm == "B_S")(
                tgt_cardinalities=get_cardinalities(ts), ctx_cardinalities=get_cardinalities(cs),
                tgt_seq_len_median=ts["seq_len"]["median"], tgt_seq_len_max=ts["seq_len"]["max"],
                ctxseq_len_median=get_ctx_sequence_length(cs, key="median"),
                model_size=ws.model_tabular_configs.read()["model_units"], column_order=None, device=torch.device("cpu"))
            before = digest(ws.model_tabular_weights_path)
            load_model_weights(model=model, path=ws.model_tabular_weights_path, device=torch.device("cpu"))
            model.eval()
            truth = first_batch(real, context, ts, cs)
            fraud_key = codec.prefixes["event_is_fraud"] + "__cat"
            for gs in CFG["generation_seeds"]:
                path = folder / f"native_validation_{gs}.parquet"
                inputs[str(path)] = digest(path)
                synthetic = pd.read_parquet(path)
                syn = first_batch(synthetic, context, ts, cs)
                cases = {"real_first_row_real_length": truth, "generated_first_row_reencoded": syn}
                for field in ["gap", "receiver_or_mark", "amount_or_numeric_value", "category"]:
                    prefix = codec.prefixes[field] + "__"
                    cases["replace_" + field + "_with_real"] = {k:truth[k] if k.startswith(prefix) else v for k,v in syn.items()}
                cases["replace_length_with_real"] = {k:truth[k] if k.startswith(SLEN_SUB_COLUMN_PREFIX) else v for k,v in syn.items()}
                cases["real_first_fields_generated_length"] = {k:syn[k] if k.startswith(SLEN_SUB_COLUMN_PREFIX) else v for k,v in truth.items()}
                for mode, batch in cases.items():
                    output, _ = model(batch, mode="trn")
                    prob = output[fraud_key].softmax(-1)[:, 0, codec.codes["1"]].numpy()
                    real_labels = truth[fraud_key][:, 0, 0].numpy()
                    rows.append(dict(arm=arm, fit_seed=fs, generation_seed=gs, mode=mode,
                                     mean_fraud_probability=float(prob.mean()),
                                     mean_probability_real_normal_customers=float(prob[real_labels == codec.codes["0"]].mean()),
                                     mean_probability_real_fraud_customers=float(prob[real_labels == codec.codes["1"]].mean()),
                                     customers=len(prob)))
            assert digest(ws.model_tabular_weights_path) == before
            print("PROBED", arm, fs, flush=True)
    pd.DataFrame(rows).to_csv(DOCS / "first_event_probes.csv", index=False)
    write(DOCS / "diagnostic_manifest.json", dict(
        adaptive_after_evaluation=True, inputs=inputs, weights_unchanged=True, test_events_read=False,
        script_sha256=digest(__file__), protocol_sha256=digest(ROOT / "docs/argn_state_first_v1/DIAGNOSTIC_PROTOCOL.md"),
        interpretation="reencoded first-row conditional sensitivity; true lengths used only in teacher diagnostic; not free-generation results"))


if __name__ == "__main__":
    main()
