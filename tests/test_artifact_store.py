import json

from experiments.artifact_store import ArtifactStore, REQUIRED


def test_atomic_manifest_complete_and_incomplete(tmp_path):
    store = ArtifactStore(tmp_path)
    incomplete = store.complete({})
    assert json.loads(incomplete.read_text())["status"] == "incomplete"
    (tmp_path / "sample.npz").write_bytes(b"sample")
    manifest = {key: "value" for key in REQUIRED}
    manifest["sample_path"] = "sample.npz"
    complete = store.complete(manifest)
    assert json.loads(complete.read_text())["status"] == "complete"
