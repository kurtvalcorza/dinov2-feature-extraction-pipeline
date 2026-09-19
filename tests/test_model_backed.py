"""Model-backed checks that run only where the pinned snapshot is staged (local pre-flight): the k-NN baseline
and a linear probe on a dozen synthetic swatches, a one-epoch unfreeze of the last block, and the artifact
round trip with head and block tensors. Skipped when the weights are absent."""

from __future__ import annotations

import hashlib
import io
import json

import pytest
import torch
from PIL import Image

from dinov2_feature_extraction_pipeline import (
    DEFAULT_WEIGHTS_DIR,
    WEIGHTS_FILE,
    DINOv2FeatureExtractionPipeline,
)

pytest.importorskip("timm")
if not (DEFAULT_WEIGHTS_DIR / WEIGHTS_FILE).is_file():
    pytest.skip("snapshot not staged", allow_module_level=True)

COLOURS = {"red": (200, 30, 30), "green": (30, 200, 30), "blue": (30, 30, 200)}


def _swatch(colour, i):
    image = Image.new("RGB", (96, 96), colour)
    px = image.load()
    for k in range(i + 1):
        px[(7 * k) % 96, (11 * k) % 96] = (255, 255, 255)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=90)
    return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")


RECORDS = [
    {"id": f"s{i:02d}", "image": _swatch(list(COLOURS.values())[i % 3], i), "label": list(COLOURS)[i % 3]}
    for i in range(12)
]


@pytest.fixture(scope="module")
def pipe():
    return DINOv2FeatureExtractionPipeline.from_pretrained(device="cpu")


def test_knn_baseline_separates_colours(pipe):
    baseline = pipe.knn_baseline(RECORDS[:9], RECORDS[9:], k=3)
    assert baseline["n"] == 3 and baseline["accuracy"] >= 2 / 3


def test_probe_then_one_epoch_unfreeze_and_artifact_round_trip(pipe, tmp_path):
    result = pipe.adapt(RECORDS[:9], RECORDS[9:], probe_steps=50, trainable_blocks=1, epochs=1, batch_size=4)
    assert result["classes"] == ["blue", "green", "red"] and result["history"][0]["stage"].startswith(
        "linear probe"
    )
    assert result["n_trainable_head"] == 384 * 3 + 3 and result["n_trainable_blocks"] == 1_775_232
    assert result["n_total"] == 22_056_192 and len(result["history"]) == 2 and result["best_epoch"] in (0, 1)
    metrics = pipe.evaluate(RECORDS[9:])
    assert metrics["n"] == 3 and metrics["adapted"] is True and metrics["policy"] == result["policy"]
    labels = pipe.classify([r["image"] for r in RECORDS[:3]])
    assert labels["classes"] == result["classes"] and len(labels["probabilities"][0]) == 3
    artifact = pipe.save_artifact(tmp_path / "adapter", {"note": "test"})
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert "head.weight" in manifest["tensors"] and manifest["adapter"]["classes"] == result["classes"]
    assert (len(manifest["tensors"]) > 2) == (result["best_epoch"] > 0)
    reloaded = DINOv2FeatureExtractionPipeline.from_artifact(artifact, device="cpu")
    assert reloaded.classify([r["image"] for r in RECORDS[:3]])["probabilities"] == labels["probabilities"]
    assert reloaded.embed(RECORDS[0]["image"])["embeddings"] == pipe.embed(RECORDS[0]["image"])["embeddings"]
    assert reloaded.adapter["best_epoch"] == result["best_epoch"] and reloaded.classes == result["classes"]


def test_no_validation_keeps_the_final_unfrozen_epoch_and_reloads_it(pipe, tmp_path):
    """Without a validation split the recorded policy is "final epoch": the head and blocks after the last of
    three unfrozen epochs stay in memory and are what the artifact carries."""
    result = pipe.adapt(RECORDS[:9], None, probe_steps=20, trainable_blocks=1, epochs=3, batch_size=4)
    assert result["best_epoch"] == 3 == result["epochs"] and result["selection"] == "final epoch"
    assert result["policy"] == pipe.adapter["policy"] == "unfrozen last 1 blocks + linear head"
    assert all(entry["val"] is None for entry in result["history"]) and len(result["history"]) == 4
    artifact = pipe.save_artifact(tmp_path / "final")
    reloaded = DINOv2FeatureExtractionPipeline.from_artifact(artifact, device="cpu")
    state, other = pipe._model.state_dict(), reloaded._model.state_dict()
    assert all(torch.equal(state[name], other[name]) for name in result["trainable_names"])
    assert torch.equal(reloaded._head.weight, pipe._head.weight) and reloaded.adapter["best_epoch"] == 3
    assert reloaded.classify([r["image"] for r in RECORDS[:2]]) == pipe.classify(
        [r["image"] for r in RECORDS[:2]]
    )


def test_load_artifact_refuses_a_tensor_set_that_differs_from_the_recorded_policy(pipe, tmp_path):
    """A manifest that claims the frozen policy but carries block tensors, records other blocks than the
    tensors it lists, or whose payload differs from its list is refused before any tensor is applied."""
    import json as _json
    import shutil

    from safetensors.torch import load_file, save_file

    pipe.adapt(RECORDS[:9], None, probe_steps=20, trainable_blocks=1, epochs=1, batch_size=4)
    artifact = pipe.save_artifact(tmp_path / "ok")
    manifest = _json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert any(name.startswith("blocks.11.") for name in manifest["tensors"])
    claims_frozen = tmp_path / "claims_frozen"
    shutil.copytree(artifact, claims_frozen)
    adapter = {**manifest["adapter"], "policy": "frozen backbone + linear probe", "trainable_blocks": 0}
    (claims_frozen / "manifest.json").write_text(_json.dumps({**manifest, "adapter": adapter}))
    with pytest.raises(ValueError, match="does not match its recorded policy"):
        DINOv2FeatureExtractionPipeline.from_artifact(claims_frozen, device="cpu")
    other_blocks = tmp_path / "other_blocks"
    shutil.copytree(artifact, other_blocks)
    adapter = {**manifest["adapter"], "policy": "unfrozen last 2 blocks + linear head", "trainable_blocks": 2}
    (other_blocks / "manifest.json").write_text(_json.dumps({**manifest, "adapter": adapter}))
    with pytest.raises(ValueError, match="does not match its recorded policy"):
        DINOv2FeatureExtractionPipeline.from_artifact(other_blocks, device="cpu")
    extra = tmp_path / "extra"
    shutil.copytree(artifact, extra)
    tensors = load_file(str(extra / "adapter.safetensors"))
    tensors["head.extra"] = torch.zeros(1)
    save_file(tensors, str(extra / "adapter.safetensors"), metadata={"format": "pt"})
    digest = hashlib.sha256((extra / "adapter.safetensors").read_bytes()).hexdigest()
    size = (extra / "adapter.safetensors").stat().st_size
    files = [{**manifest["files"][0], "bytes": size, "sha256": digest}]
    (extra / "manifest.json").write_text(_json.dumps({**manifest, "files": files}))
    with pytest.raises(ValueError, match="tensor names differ"):
        DINOv2FeatureExtractionPipeline.from_artifact(extra, device="cpu")


def test_adapt_is_transactional_when_the_progress_callback_raises(pipe):
    """A failure inside the unfreeze leaves the base exactly as it was, frozen, with no head or adapter."""
    before = {k: v.clone() for k, v in pipe._model.state_dict().items()}

    def boom(entry):
        if entry["epoch"] == 1:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        pipe.adapt(
            RECORDS[:9], None, probe_steps=20, trainable_blocks=1, epochs=2, batch_size=4, progress=boom
        )
    after = pipe._model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before) and pipe.adapter is None
    assert pipe._head is None and not any(p.requires_grad for p in pipe._model.parameters())
