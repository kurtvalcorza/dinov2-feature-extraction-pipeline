"""Offline tests for the public validation and evaluation stage helpers (DAT24 / EVAL21)."""

from __future__ import annotations

import pytest
from PIL import Image

from dinov2_feature_extraction_pipeline import (
    EMBED_DIM,
    INPUT_SCHEMA,
    MAX_BATCH,
    MAX_IMAGE_SIDE,
    MODEL_ID,
    MODEL_REVISION,
    evaluation_report,
    validate_inputs,
)


def _image(side: int = 32) -> Image.Image:
    return Image.new("RGB", (side, side), (10, 20, 30))


def _result(n: int = 2) -> dict:
    return {
        "embeddings": [[0.0] * EMBED_DIM for _ in range(n)],
        "dim": EMBED_DIM,
        "pooling": "cls",
        "normalized": True,
    }


def test_validate_inputs_returns_manifest_with_schema_and_identity() -> None:
    manifest = validate_inputs([_image(), _image(48)], names=["a", "b"])
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["image_side_px"] == [1, MAX_IMAGE_SIDE]
    assert manifest["schema"]["batch"] == [1, MAX_BATCH]
    assert str(EMBED_DIM) in manifest["schema"]["output"]
    assert manifest["inputs"] == [
        {"id": "a", "mode": "RGB", "size": [32, 32]},
        {"id": "b", "mode": "RGB", "size": [48, 48]},
    ]
    assert manifest["batch_size"] == 2
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_validate_inputs_single_image_default_ids() -> None:
    manifest = validate_inputs(_image())
    assert [entry["id"] for entry in manifest["inputs"]] == ["image-0"]
    assert manifest["batch_size"] == 1


def test_validate_inputs_rejects_like_embed() -> None:
    with pytest.raises(ValueError, match="MAX_IMAGE_SIDE"):
        validate_inputs(Image.new("RGB", (MAX_IMAGE_SIDE + 1, 8)))
    with pytest.raises(ValueError, match="MAX_BATCH"):
        validate_inputs([_image()] * (MAX_BATCH + 1))
    with pytest.raises(TypeError):
        validate_inputs("not an image")
    with pytest.raises(ValueError, match="names must have one entry per image"):
        validate_inputs([_image()], names=["a", "b"])


def test_evaluation_report_is_always_not_measurable() -> None:
    report = evaluation_report(_result(3))
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert report["baselines"] == []
    assert report["n_embeddings"] == 3
    assert "representation" in report["reason"]
    assert "linear probe" in report["needs"]
    assert "cosine similarity" in report["score_semantics"]
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_evaluation_report_stays_not_measurable_with_targets() -> None:
    report = evaluation_report(_result(2), ["cat", "dog"], sample_kind="BYOD")
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert report["sample_kind"] == "BYOD"
    assert "labels alone cannot" in report["reason"]


def test_evaluation_report_names_the_task_and_no_decision_rule() -> None:
    report = evaluation_report(_result(1))
    assert "feature extraction" in report["task"]
    assert "decision_rule" not in report
    assert "no threshold is shipped" in report["score_semantics"]
