"""Self-supervised image feature extraction with the pinned ``timm/vit_small_patch14_dinov2.lvd142m`` weights.

DINOv2 ViT-S/14 with no classifier head (``num_classes=0``): each image becomes one 384-d vector,
the class token after the final norm (``POOLING = "cls"``), L2-normalised here. Weights load only
from a digest-verified local snapshot (``weights/<key>/``) or, when explicitly allowed, from the
Hugging Face Hub at the pinned revision. Preprocessing is the upstream ``pretrained_cfg``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

MODEL_ID = "timm/vit_small_patch14_dinov2.lvd142m"
MODEL_REVISION = "4610ca143709d58a633b6397a74412c2c3842454"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "vit-small-dinov2"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"
WEIGHTS_FILE = "model.safetensors"
CONFIG_FILE = "config.json"

EMBED_DIM = 384  # ViT-S width; the snapshot config.json reports num_features = 384
POOLING = "cls"  # snapshot config.json global_pool = "token": the class token after the final LayerNorm
NORMALIZED = True  # embed() L2-normalises every vector, so a dot product is a cosine similarity
MAX_IMAGE_SIDE = 4096  # pixels; larger images are rejected before any decode-to-tensor work
MAX_BATCH = 32  # images per embed() call; 518-px inputs cost 46.8 GMACs each (upstream card)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local snapshot against its manifest; raise naming the first mismatch."""
    root = Path(path or DEFAULT_WEIGHTS_DIR)
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"snapshot manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    for entry in manifest.get("files", []):
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {"path": str(root), **manifest}


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def _hub_reference(model_id: str, revision: str) -> str:
    """timm's ``hf-hub:owner/name@revision`` form; ``hf_split`` passes ``revision=`` to hf_hub_download."""
    return f"hf-hub:{model_id}@{revision}"


INPUT_SCHEMA: dict[str, Any] = {
    "input": "PIL.Image.Image or a sequence of them; any mode, converted to RGB",
    "image_side_px": [1, MAX_IMAGE_SIDE],
    "batch": [1, MAX_BATCH],
    "output": f"one L2-normalised {EMBED_DIM}-d vector per image, pooling {POOLING!r}",
    "preprocessing": (
        "resize shorter side to 518 px, center-crop 518x518 (crop_pct 1.0, bicubic), ImageNet mean/std"
    ),
}


def _check_inputs(images: Any) -> list[Image.Image]:
    """Raise TypeError/ValueError naming the first violated ceiling; return the images as a list."""
    if isinstance(images, Image.Image):
        images = [images]
    if not isinstance(images, Sequence) or isinstance(images, str | bytes):
        raise TypeError("images must be a PIL.Image.Image or a sequence of them")
    if not 1 <= len(images) <= MAX_BATCH:
        raise ValueError(f"batch size must be between 1 and MAX_BATCH={MAX_BATCH}, got {len(images)}")
    for image in images:
        if not isinstance(image, Image.Image):
            raise TypeError(f"each image must be a PIL.Image.Image, got {type(image).__name__}")
        width, height = image.size
        if width < 1 or height < 1 or max(width, height) > MAX_IMAGE_SIDE:
            raise ValueError(f"image side outside 1..MAX_IMAGE_SIDE={MAX_IMAGE_SIDE} px: {image.size}")
    return list(images)


def validate_inputs(
    images: Image.Image | Sequence[Image.Image], *, names: Sequence[str] | None = None
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, per-input observations, verdict).

    Rejection is reported by raising exactly as ``embed`` would; a caller that wants the finding
    recorded catches the exception and stores ``str(exc)`` under ``findings``.
    """
    checked = _check_inputs(images)
    if names is not None and len(names) != len(checked):
        raise ValueError("names must have one entry per image")
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [
            {"id": names[i] if names else f"image-{i}", "mode": image.mode, "size": list(image.size)}
            for i, image in enumerate(checked)
        ],
        "batch_size": len(checked),
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def evaluation_report(
    result: Mapping[str, Any], targets: Sequence[Any] | None = None, *, sample_kind: str = "synthetic"
) -> dict[str, Any]:
    """Evaluation stage: always ``not-measurable`` — the output is a representation (EVAL9).

    The repository ships no metric helper because a vector has nothing to be scored against on its
    own. ``targets`` is accepted so the signature matches the fleet's other pipelines, but class or
    relevance labels alone cannot score an embedding: the report stays ``not-measurable`` and names
    the downstream task that would make the features measurable.
    """
    embeddings = result["embeddings"]
    reason = "an embedding is a representation, not a prediction: no intrinsic performance measure exists"
    if targets is not None:
        reason = (
            "targets were supplied, but this pipeline exposes no metric helper and labels alone cannot "
            "score a representation; score them through a downstream task instead"
        )
    return {
        "task": "self-supervised image feature extraction (pooled per-image embedding)",
        "score_semantics": (
            f"cosine similarity between two L2-normalised {EMBED_DIM}-d vectors (a dot product); "
            "a similarity is not a probability, an accuracy or a calibrated score; no threshold is shipped"
        ),
        "sample_kind": sample_kind,
        "n_embeddings": len(embeddings),
        "metrics": [],
        "baselines": [],
        "verdict": "not-measurable",
        "reason": reason,
        "needs": (
            "a downstream labelled task: a retrieval set with relevance labels (mean average precision), "
            "a labelled image set for a linear probe or k-NN classifier (accuracy), or human-judged "
            "duplicate pairs to calibrate a similarity threshold"
        ),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


@dataclass
class DINOv2FeatureExtractionPipeline:
    """``_runner`` maps a float tensor (N, 3, H, W) to pooled features (N, EMBED_DIM); injectable."""

    _runner: Callable[[Any], Any]
    _transform: Callable[[Image.Image], Any]
    device: str = "cpu"
    input_size: tuple[int, int] = (0, 0)
    source: str = "injected"

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> DINOv2FeatureExtractionPipeline:
        import timm
        import torch
        from timm.data import create_transform, resolve_model_data_config

        root = Path(weights_dir or DEFAULT_WEIGHTS_DIR)
        arch_name = MODEL_ID.split("/", 1)[1]
        if (root / MANIFEST_NAME).is_file():
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
            with open(root / CONFIG_FILE, encoding="utf-8") as fh:
                config = json.load(fh)
            snapshot_name = f"{config['architecture']}.{config['pretrained_cfg']['tag']}"
            if snapshot_name != arch_name:
                raise ValueError(f"snapshot config names {snapshot_name!r}, expected {arch_name!r}")
            overlay = dict(config["pretrained_cfg"])
            overlay["file"] = str(root / WEIGHTS_FILE)  # 'file' takes precedence over hf_hub_id in timm
            model = timm.create_model(
                arch_name, pretrained=True, pretrained_cfg_overlay=overlay, num_classes=0
            )
            source = "local-snapshot"
        elif allow_download:
            model = timm.create_model(
                _hub_reference(MODEL_ID, revision=MODEL_REVISION), pretrained=True, num_classes=0
            )
            source = "hf-hub"
        else:
            raise FileNotFoundError(
                f"no verified snapshot at {root} and allow_download=False; "
                f"stage it with: hf download {MODEL_ID} --revision {MODEL_REVISION} --local-dir {root}"
            )
        if getattr(model, "num_features", EMBED_DIM) != EMBED_DIM:
            raise ValueError(f"model num_features {model.num_features} != EMBED_DIM={EMBED_DIM}")
        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.eval().to(resolved_device)
        data_config = resolve_model_data_config(model)
        transform = create_transform(**data_config, is_training=False)
        input_size = tuple(int(s) for s in data_config["input_size"][1:])

        def runner(batch: Any) -> Any:
            with torch.inference_mode():
                return model(batch.to(resolved_device))  # num_classes=0 -> pooled class token, (N, 384)

        return cls(runner, transform, resolved_device, input_size, source)

    def _validate(self, images: Any) -> list[Image.Image]:
        return _check_inputs(images)

    def embed(self, images: Image.Image | Sequence[Image.Image]) -> dict[str, Any]:
        """Return one L2-normalised float32 vector of length EMBED_DIM per image; no labels, no scores."""
        import torch

        batch_images = self._validate(images)
        batch = torch.stack([self._transform(image.convert("RGB")) for image in batch_images])
        features = self._runner(batch)
        if not isinstance(features, torch.Tensor) or features.shape != (len(batch_images), EMBED_DIM):
            raise RuntimeError("runner must return a tensor of shape (batch, EMBED_DIM)")
        vectors = torch.nn.functional.normalize(features.float(), dim=-1).cpu()
        return {
            "embeddings": [[float(v) for v in row] for row in vectors.tolist()],
            "dim": EMBED_DIM,
            "pooling": POOLING,
            "normalized": NORMALIZED,
            "input_size": list(self.input_size),
            "device": self.device,
            "source": self.source,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
