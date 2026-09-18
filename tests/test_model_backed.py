"""Model-backed checks that run only where the pinned snapshot is staged (local pre-flight): the k-NN baseline
and a linear probe on a dozen synthetic swatches, a one-epoch unfreeze of the last block, and the artifact
round trip with head and block tensors. Skipped when the weights are absent."""

from __future__ import annotations

import io
import json

import pytest
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
