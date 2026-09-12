import hashlib
import json
import math
import re
from pathlib import Path

import pytest
import torch
from PIL import Image

from dinov2_feature_extraction_pipeline import (
    DEFAULT_WEIGHTS_DIR,
    EMBED_DIM,
    MAX_BATCH,
    MAX_IMAGE_SIDE,
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    POOLING,
    DINOv2FeatureExtractionPipeline,
    stage_missing_files,
    verify_snapshot,
)
from dinov2_feature_extraction_pipeline import pipeline as pipeline_module

HEX40 = re.compile(r"^[0-9a-f]{40}$")


def _fake_transform(image: Image.Image) -> torch.Tensor:
    return torch.full((3, 8, 8), float(image.size[0]))


def _fake_runner(batch: torch.Tensor) -> torch.Tensor:
    features = torch.zeros(batch.shape[0], EMBED_DIM)
    features[:, 0] = 3.0  # un-normalised on purpose: embed() must normalise
    features[:, 1] = batch[:, 0, 0, 0] * 4.0  # depends on the (fake) image width
    return features


def _pipeline() -> DINOv2FeatureExtractionPipeline:
    return DINOv2FeatureExtractionPipeline(_fake_runner, _fake_transform, "cpu", (518, 518), "injected")


def _write_snapshot(root: Path, payload: bytes = b"weights") -> Path:
    (root / "model.safetensors").write_bytes(payload)
    manifest = {
        "modelKey": MODEL_KEY,
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {
                "path": "model.safetensors",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
    }
    path = root / "dimer-base-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_identity_constants_are_40_hex_and_named():
    assert HEX40.match(MODEL_REVISION)
    assert MODEL_ID == "timm/vit_small_patch14_dinov2.lvd142m"
    assert DEFAULT_WEIGHTS_DIR.name == MODEL_KEY
    assert DEFAULT_WEIGHTS_DIR.parent.name == "weights"
    assert EMBED_DIM == 384
    assert POOLING == "cls"


def test_identity_matches_local_manifest_and_config_when_present():
    manifest_path = DEFAULT_WEIGHTS_DIR / "dimer-base-manifest.json"
    if not manifest_path.is_file():
        pytest.skip("local snapshot manifest not staged")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["modelId"] == MODEL_ID
    assert manifest["revision"] == MODEL_REVISION
    assert manifest["modelKey"] == MODEL_KEY
    config = json.loads((DEFAULT_WEIGHTS_DIR / "config.json").read_text(encoding="utf-8"))
    assert config["num_features"] == EMBED_DIM
    assert config["num_classes"] == 0
    assert config["global_pool"] == "token"


def test_verify_snapshot_accepts_matching_manifest(tmp_path: Path):
    _write_snapshot(tmp_path)
    result = verify_snapshot(tmp_path)
    assert result["revision"] == MODEL_REVISION
    assert result["path"] == str(tmp_path)


def test_verify_snapshot_rejects_tampered_digest(tmp_path: Path):
    manifest_path = _write_snapshot(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = manifest["files"][0]["sha256"]
    flipped = ("0" if digest[0] != "0" else "1") + digest[1:]
    manifest["files"][0]["sha256"] = flipped
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(tmp_path)


def test_verify_snapshot_rejects_tampered_bytes_and_missing_file(tmp_path: Path):
    _write_snapshot(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"weightz")
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"short")
    with pytest.raises(ValueError, match="size"):
        verify_snapshot(tmp_path)
    (tmp_path / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError):
        verify_snapshot(tmp_path)


def test_verify_snapshot_rejects_wrong_identity(tmp_path: Path):
    manifest_path = _write_snapshot(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["revision"] = "0" * 40
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(tmp_path)
    with pytest.raises(FileNotFoundError):
        verify_snapshot(tmp_path / "missing")


def test_from_pretrained_refuses_without_snapshot_or_download(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="allow_download=False"):
        DINOv2FeatureExtractionPipeline.from_pretrained(weights_dir=tmp_path, allow_download=False)


def test_hub_reference_carries_pinned_revision():
    reference = pipeline_module._hub_reference(MODEL_ID, revision=MODEL_REVISION)
    assert reference == f"hf-hub:{MODEL_ID}@{MODEL_REVISION}"


def test_embed_rejects_bad_inputs():
    pipe = _pipeline()
    image = Image.new("RGB", (32, 32))
    with pytest.raises(TypeError):
        pipe.embed("not-an-image")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        pipe.embed([image, 42])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="batch size"):
        pipe.embed([])
    with pytest.raises(ValueError, match="batch size"):
        pipe.embed([image] * (MAX_BATCH + 1))
    with pytest.raises(ValueError, match="MAX_IMAGE_SIDE"):
        pipe.embed(Image.new("RGB", (MAX_IMAGE_SIDE + 1, 1)))
    with pytest.raises(ValueError, match="MAX_IMAGE_SIDE"):
        pipe.embed(Image.new("RGB", (1, MAX_IMAGE_SIDE + 1)))


def test_embed_output_fields_and_unit_norm():
    pipe = _pipeline()
    result = pipe.embed([Image.new("L", (16, 16)), Image.new("RGB", (32, 32))])
    assert result["model_id"] == MODEL_ID
    assert result["model_revision"] == MODEL_REVISION
    assert result["dim"] == EMBED_DIM
    assert result["pooling"] == "cls"
    assert result["normalized"] is True
    assert result["input_size"] == [518, 518]
    assert len(result["embeddings"]) == 2
    for vector in result["embeddings"]:
        assert len(vector) == EMBED_DIM
        assert all(isinstance(v, float) for v in vector)
        assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-5)
    first, second = result["embeddings"]
    assert first != second  # the fake runner encodes image width, so different inputs differ
    assert math.isclose(sum(a * b for a, b in zip(first, first, strict=True)), 1.0, rel_tol=1e-5)


def test_embed_rejects_runner_with_wrong_shape():
    pipe = DINOv2FeatureExtractionPipeline(
        lambda batch: torch.zeros(batch.shape[0], EMBED_DIM + 1), _fake_transform, "cpu", (518, 518)
    )
    with pytest.raises(RuntimeError, match="EMBED_DIM"):
        pipe.embed(Image.new("RGB", (16, 16)))


def test_stage_missing_files_fetches_only_absent_entries_then_verifies(tmp_path):
    """Fresh-clone shape: manifest committed, weight file absent. allow_download fetches exactly that file."""
    payload = b"weights-bytes"
    (tmp_path / "config.json").write_bytes(b"{}")
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {"path": "config.json", "bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()},
            {"path": "model.bin", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()},
        ],
    }
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(tmp_path)
    fetched = []

    def fake_download(relative_path, root):
        fetched.append(relative_path)
        (root / relative_path).write_bytes(payload)

    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == ["model.bin"]
    assert fetched == ["model.bin"]
    listed = verify_snapshot(tmp_path)["files"]
    assert (listed if isinstance(listed, int) else len(listed)) == 2
    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == []


def test_stage_missing_files_refuses_foreign_manifest(tmp_path):
    manifest = {"modelId": "someone/else", "revision": MODEL_REVISION, "files": []}
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(tmp_path, allow_download=True, downloader=lambda *_: None)
