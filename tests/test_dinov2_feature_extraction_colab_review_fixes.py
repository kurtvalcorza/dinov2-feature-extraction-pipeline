"""Regression tests for the dinov2_feature_extraction_colab review fixes (DV2-M1..M3, DV2-m1..m4).

They run within CI's install budget (CPU torch, Pillow, NumPy; no model weights): adaptation is exercised on a tiny
random-initialised stand-in whose parameters carry the backbone's `blocks.<k>.` names, and the notebook's own
Section 4 cell is executed with the carried package functions. None of this is pretrained-inference or
clean-runtime evidence.
"""
# ruff: noqa: E501  -- test cases quote notebook source lines and refusal messages in full

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import types
import zipfile
from pathlib import Path
from typing import Any

import pytest

with contextlib.suppress(ImportError):  # load torch before any NumPy BLAS call (Windows DLL load-order trap)
    import torch

from PIL import Image

import dinov2_feature_extraction_pipeline as package
from dinov2_feature_extraction_pipeline import samples as sm
from dinov2_feature_extraction_pipeline.pipeline import (
    EMBED_DIM,
    TRANSFORMER_BLOCKS,
    DINOv2FeatureExtractionPipeline,
)

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "dinov2_feature_extraction_colab.ipynb"
COLOURS = {"red": (200, 30, 30), "green": (30, 200, 30), "blue": (30, 30, 200)}


@pytest.fixture(scope="module")
def nb() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _src(cell: dict) -> str:
    return "".join(cell["source"])


def _code_cells(nb: dict) -> list[dict]:
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def _cell(nb: dict, marker: str) -> str:
    found = [_src(c) for c in _code_cells(nb) if marker in _src(c)]
    assert len(found) == 1, marker
    return found[0]


def _markdown(nb: dict) -> str:
    return "\n".join(_src(c) for c in nb["cells"] if c["cell_type"] == "markdown")


def _jpeg(colour: tuple[int, int, int], noise: int) -> bytes:
    image = Image.new("RGB", (32, 24), colour)
    image.putpixel((noise % 32, (noise // 32) % 24), (0, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def _records(per_class: int, prefix: str = "r") -> list[dict[str, Any]]:
    out = []
    for c, (label, colour) in enumerate(COLOURS.items()):
        for k in range(per_class):
            image = Image.open(io.BytesIO(_jpeg(colour, 7 * c + k + 1))).convert("RGB")
            out.append({"id": f"{prefix}{c}{k:02d}", "image": image, "label": label})
    return out


# --- DV2-M1: isolated runtime, no in-kernel install -----------------------------------------------------------------


def test_exactly_two_kernel_cells_and_a_hash_locked_isolated_install(nb: dict) -> None:
    kernel = [_src(c) for c in _code_cells(nb) if "# dimer: kernel cell" in _src(c)]
    assert len(kernel) == 2
    install = kernel[0]
    for needed in ('"--managed-python"', '"--require-hashes"', '"--only-binary"', "UV_SHA256", "LOCK_SHA256"):
        assert needed in install
    assert "_ip.input_transformers_cleanup.append(_route_to_isolated_runtime)" in kernel[1]
    lock = (ROOT / "tutorials" / "requirements-colab.lock.txt").read_text(encoding="utf-8")
    for pin in ("torch==2.14.0", "timm==1.0.29", "numpy==2.5.3", "pillow==11.3.0", "safetensors==0.8.0"):
        assert pin in lock


@pytest.mark.parametrize("real_google", [False, True])
def test_worker_colab_stubs_have_specs(nb: dict, monkeypatch: pytest.MonkeyPatch, real_google: bool) -> None:
    """find_spec("google.colab") (accelerate does this) must not raise on the worker's stubs (fleet Colab failure)."""
    import importlib.util

    namespace: dict[str, Any] = {"SKIP_INSTALL": True, "__name__": "__main__"}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(_src(_code_cells(nb)[1]), "router", "exec"), namespace)
    worker = namespace["_WORKER_SOURCE"]
    start = worker.index('if os.environ.get("DIMER_KERNEL_IS_COLAB") == "1":')
    shim = worker[start : worker.index('_main = types.ModuleType("__main__")', start)]
    names = ("google", "google.colab", "google.colab.files")
    saved = {name: sys.modules[name] for name in names if name in sys.modules}
    fake_google = types.ModuleType("google")
    fake_google.__path__ = []
    try:
        for name in names:
            sys.modules.pop(name, None)
        sys.modules["google"] = fake_google if real_google else None
        monkeypatch.setenv("DIMER_KERNEL_IS_COLAB", "1")
        shim_globals = {"os": os, "sys": sys, "types": types, "_send": None, "_recv": None}
        exec(compile(shim, "worker-colab-shim", "exec"), shim_globals)
        for name in ("google.colab", "google.colab.files"):
            spec = importlib.util.find_spec(name)
            assert spec is not None and spec.name == name
        assert sys.modules["google.colab"].__path__ == [] and callable(sys.modules["google.colab.files"].upload)
        if not real_google:
            assert importlib.util.find_spec("google") is not None
    finally:
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


def test_release_record_no_longer_counts_the_restarted_run_as_a_pass() -> None:
    text = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    assert "(1 restart after install cell)" not in text
    assert "**Passed only after a manual restart** — not a one-pass Run all, not promotion evidence." in text
    assert "restart after the install is expected" not in text
    assert "`restarted: false`" in text
    for name in ("STATUS.md", "README.md", "MODEL_CARD.md", "tutorials/README.md"):
        assert "(1 restart after install cell)" not in (ROOT / name).read_text(encoding="utf-8"), name
    assert "Current status: **Candidate" in (ROOT / "STATUS.md").read_text(encoding="utf-8")


def test_no_learner_cell_installs_into_the_kernel(nb: dict) -> None:
    assert "Restart the runtime, then rerun" not in _markdown(nb)
    runtime = _cell(nb, "# @title Infrastructure: record the runtime")
    assert "SKIP_INSTALL = os.environ.get('DIMER_NOTEBOOK_CI_PREINSTALLED') == '1'" in runtime


# --- DV2-M2: every adaptation starts from the pretrained base ---------------------------------------------------


class _TinyBackbone(torch.nn.Module if "torch" in sys.modules else object):
    """Mean colour -> 384-d features through 12 residual 'blocks.<k>.' layers, so adapt() can unfreeze the last ones."""

    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = torch.nn.Linear(3, EMBED_DIM)
        self.blocks = torch.nn.ModuleList(torch.nn.Linear(EMBED_DIM, EMBED_DIM) for _ in range(TRANSFORMER_BLOCKS))

    def forward(self, batch: Any) -> Any:
        x = self.patch_embed(batch.mean(dim=(2, 3)))
        for block in self.blocks:
            x = x + 0.1 * torch.tanh(block(x))
        return x


def _transform(image: Image.Image) -> Any:
    import numpy as np

    arr = np.asarray(image.convert("RGB").resize((8, 8)), dtype=np.float32) / 255.0
    return torch.tensor(arr).permute(2, 0, 1)


def _tiny_pipeline() -> DINOv2FeatureExtractionPipeline:
    torch.manual_seed(1234)
    model = _TinyBackbone().eval()
    for param in model.parameters():
        param.requires_grad_(False)

    def runner(batch: Any) -> Any:
        with torch.inference_mode():
            return model(batch)

    return DINOv2FeatureExtractionPipeline(runner, _transform, "cpu", (8, 8), "injected", _model=model)


def _state(pipe: DINOv2FeatureExtractionPipeline) -> dict[str, Any]:
    return {k: v.detach().clone() for k, v in pipe._model.state_dict().items()}


@pytest.fixture()
def data() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    pytest.importorskip("torch")
    return _records(4, "t"), _records(2, "v"), _records(2, "s")


def test_adapt_rerun_starts_from_the_pretrained_base(data) -> None:
    """Review acceptance check: after an unfrozen run, a TRAINABLE_BLOCKS=1 rerun's epoch 0 equals the probe."""
    train, val, test = data
    pipe = _tiny_pipeline()
    pristine = _state(pipe)
    probe = pipe.adapt(train, val, probe_steps=50, trainable_blocks=0)
    assert not pipe.backbone_adapted
    pipe.adapt(train, None, probe_steps=50, trainable_blocks=2, epochs=2, lr=1e-2, batch_size=4)  # final epoch kept
    assert pipe.adapter["policy"].startswith("unfrozen") and pipe.backbone_adapted
    assert not torch.equal(pipe._model.state_dict()["blocks.10.weight"], pristine["blocks.10.weight"])
    rerun = pipe.adapt(train, val, probe_steps=50, trainable_blocks=1, epochs=1, lr=1e-2, batch_size=4)
    assert abs(rerun["history"][0]["val"]["log_loss"] - probe["history"][0]["val"]["log_loss"]) < 1e-6
    state = pipe._model.state_dict()
    for key, value in pristine.items():
        if not key.startswith("blocks.11."):
            assert torch.equal(state[key], value), key
    assert all(not p.requires_grad for p in pipe._model.parameters())


def test_artifact_reproduces_the_in_memory_model_after_a_rerun(data, tmp_path: Path) -> None:
    """With TRAINABLE_BLOCKS=1 after a two-block run, base + saved tensors must equal the in-memory model (VER4)."""
    train, val, test = data
    pipe = _tiny_pipeline()
    pipe.adapt(train, None, probe_steps=50, trainable_blocks=2, epochs=2, lr=1e-2, batch_size=4)
    pipe.adapt(train, None, probe_steps=50, trainable_blocks=1, epochs=1, lr=1e-2, batch_size=4)
    after = pipe.classify([r["image"] for r in test])
    pipe.save_artifact(tmp_path / "adapter")
    fresh = _tiny_pipeline()
    fresh.load_artifact(tmp_path / "adapter")
    assert fresh.classify([r["image"] for r in test])["probabilities"] == after["probabilities"]
    # A probe-only artifact loaded onto an adapted pipeline overlays onto the base, not onto the adapted blocks.
    pipe.adapt(train, val, probe_steps=50, trainable_blocks=0)
    probe_only = pipe.classify([r["image"] for r in test])
    pipe.save_artifact(tmp_path / "probe")
    fresh.load_artifact(tmp_path / "probe")
    assert not fresh.backbone_adapted
    assert fresh.classify([r["image"] for r in test])["probabilities"] == probe_only["probabilities"]


def test_reset_to_pretrained_restores_the_base_and_drops_the_head(data) -> None:
    train, val, test = data
    pipe = _tiny_pipeline()
    pristine = _state(pipe)
    before = pipe.features(test)
    pipe.adapt(train, None, probe_steps=20, trainable_blocks=3, epochs=1, lr=1e-2, batch_size=4)
    assert pipe.backbone_adapted
    assert pipe.reset_to_pretrained() is pipe
    assert not pipe.backbone_adapted and pipe.adapter is None and pipe._head is None and pipe.classes is None
    assert all(torch.equal(pipe._model.state_dict()[k], v) for k, v in pristine.items())
    assert (pipe.features(test) == before).all()
    with pytest.raises(ValueError, match="no classification head"):
        pipe.classify([test[0]["image"]])


def test_reset_is_harmless_on_an_injected_pipeline_without_a_model() -> None:
    pipe = DINOv2FeatureExtractionPipeline(lambda b: b, lambda i: i, "cpu", (8, 8), "injected")
    assert pipe.reset_to_pretrained() is pipe and not pipe.backbone_adapted


def test_notebook_resets_before_the_floors_and_checks_the_rerun(nb: dict) -> None:
    embed = _cell(nb, "embedding_result = pipe.embed([r['image'] for r in probe_records])")
    frozen = _cell(nb, "probe_result = pipe.adapt(")
    unfrozen = _cell(nb, "adapt_result = pipe.adapt(")
    assert "pipe.reset_to_pretrained()" in embed and "pipe.reset_to_pretrained()" in frozen
    assert frozen.index("pipe.reset_to_pretrained()") < frozen.index("pipe.knn_baseline(")
    assert "'starts_from_pretrained_base': start_check" in unfrozen and "abs_difference'] > 1e-3" in unfrozen
    md = _markdown(nb)
    assert "re-run from that cell" not in md
    assert "**Runtime → Run after**" in md
    assert "run_history = globals().get('run_history', [])" in _cell(nb, "adapted_test = pipe.evaluate(test_records)")


# --- DV2-m1: BYOD intake ---------------------------------------------------------------------------------------------


def _byod_zip(rows: list[tuple[str, str, str]], files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("labels.csv", "id,file,label\n" + "".join(f"{i},{f},{label}\n" for i, f, label in rows))
        for name, data in files.items():
            archive.writestr(f"images/{name}", data)
    return buffer.getvalue()


def _in_contract_zip(per_class: int = 20) -> bytes:
    rows, files = [], {}
    for c, (label, colour) in enumerate(list(COLOURS.items())[:2]):
        for k in range(per_class):
            name = f"{label}_{k:02d}.png"
            rows.append((f"{label}-{k:02d}", name, label))
            files[name] = _jpeg(colour, 5 * c + k + 1)
    return _byod_zip(rows, files)


def test_a_40_image_two_class_byod_zip_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "birds.zip"
    path.write_bytes(_in_contract_zip(20))
    splits = sm.split_dataset(sm.load_byod_dataset(path), seed=42)
    manifests = sm.validate_splits(splits)
    assert {k: m["n_records"] for k, m in manifests.items()} == {"test": 8, "validation": 6, "train": 26}


def test_validate_splits_names_the_split() -> None:
    splits = {"train": _records(3), "validation": [], "test": _records(1)}
    with pytest.raises(ValueError, match=r"^validation split: 0 records; 1\.\.20000 are required$"):
        sm.validate_splits(splits)
    with pytest.raises(ValueError, match=r"^train split: 6 records; 8\.\.20000 are required$"):
        sm.validate_splits({"train": _records(2), "validation": _records(1), "test": _records(1)})
    one_class = [r for r in _records(3) if r["label"] == "red"]
    with pytest.raises(ValueError, match=r"^test split: 1 distinct labels"):
        sm.validate_splits({"train": _records(3), "validation": _records(1), "test": one_class})


@pytest.mark.parametrize(
    ("rows", "files", "message"),
    [
        ([("a1", "ghost.png", "red")], {}, r"labels\.csv line 2 \(id 'a1', file 'ghost\.png'\): the file is not in the dataset"),
        ([("a1", "note.png", "red")], {"note.png": b"not an image"}, r"labels\.csv line 2 \(id 'a1', file 'note\.png'\): not a decodable image \(UnidentifiedImageError\)"),
    ],
)
def test_byod_missing_or_undecodable_file_names_the_row(tmp_path: Path, rows, files, message: str) -> None:
    good = [("g1", "good.png", "blue")]
    path = tmp_path / "bad.zip"
    path.write_bytes(_byod_zip(good + rows, {"good.png": _jpeg((0, 0, 200), 3), **files}))
    message = message.replace("line 2", "line 3")
    with pytest.raises(ValueError, match=message):
        sm.load_byod_dataset(path)
    folder = tmp_path / "dir"
    folder.mkdir()
    (folder / "labels.csv").write_text("id,file,label\n" + "".join(f"{i},{f},{label}\n" for i, f, label in good + rows), encoding="utf-8")
    (folder / "good.png").write_bytes(_jpeg((0, 0, 200), 3))
    for name, data in files.items():
        (folder / name).write_bytes(data)
    with pytest.raises(ValueError, match=message):
        sm.load_byod_dataset(folder)


def test_section_4_cell_runs_a_byod_path_without_the_upload_dialog(nb: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The notebook's own Section 4 cell, BYOD_PATH set: accepted 40-image zip, then a refused one naming the row."""
    source = _cell(nb, "dataset_manifests = validate_splits(splits)")
    archive = tmp_path / "birds.zip"
    archive.write_bytes(_in_contract_zip(20))
    monkeypatch.chdir(tmp_path)
    cell = source.replace("USE_BYOD = False", "USE_BYOD = True").replace("BYOD_PATH = ''", f"BYOD_PATH = {str(archive)!r}")
    ns: dict[str, Any] = {name: getattr(package, name) for name in package.__all__}
    ns.update({"os": os, "Path": Path, "Image": Image, "SPECIES": sm.SPECIES, "read_corpus": sm.read_corpus, "fetch_corpus": sm.fetch_corpus})
    with contextlib.redirect_stdout(io.StringIO()) as out:
        exec(compile(cell, "section4", "exec"), ns)
    assert ns["disjoint"] == {"test": 8, "validation": 6, "train": 26}
    assert "'data_source': 'BYOD (birds.zip)'" in out.getvalue()
    assert (tmp_path / "outputs" / "dinov2_feature_extraction_train.csv").is_file()
    archive.write_bytes(_byod_zip([("g1", "ghost.png", "red")], {}))
    with pytest.raises(ValueError, match=r"labels\.csv line 2 \(id 'g1', file 'ghost\.png'\)"):
        exec(compile(cell, "section4", "exec"), dict(ns))


# --- DV2-m2 / DV2-m3 / DV2-m4 / DV2-M3: notebook text ------------------------------------------------------------


def test_data_contract_shows_single_braces_and_the_enforced_id_pattern(nb: dict) -> None:
    md = _markdown(nb)
    assert "{{" not in md and "}}" not in md
    assert f"`{sm._ID_RE.pattern}`" in md


def test_section_9_frozen_column_is_the_section_6_probe(nb: dict) -> None:
    frozen = _cell(nb, "probe_result = pipe.adapt(")
    export = _cell(nb, "pipe.save_artifact(artifact_dir")
    assert "frozen_before = pipe.classify([r['image'] for r in show])" in frozen
    assert "before = frozen_before" in export and "from_pretrained(" not in export
    md = _markdown(nb)
    assert "recomputed from the probe's stored head" not in md
    assert "kept in `frozen_before`" in md


def test_measurement_claims_name_their_run(nb: dict) -> None:
    md = _markdown(nb)
    for stale in ("the build record", "lost eight points"):
        assert stale not in md
    assert "local Windows CPU pre-flight (2026-09-19" in md
    record = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    assert "183.7 s" in record and "18.7 s" in record and "96.6 s" in record
    assert "no metric helper" not in (ROOT / "README.md").read_text(encoding="utf-8")
    assert "lost eight points" not in (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")


def test_guided_layer_and_infrastructure_cells(nb: dict) -> None:
    md = _markdown(nb)
    for marker, least in (
        ("**Who this is for.**", 1),
        ("**Input → Model → Output.**", 1),
        ("**How to use this notebook.**", 1),
        ("**Roadmap:**", 1),
        ("**Predict before running:**", 6),
        ("**What to notice:**", 6),
        ("<summary>Check your reasoning</summary>", 7),
        ("**Predict → Change one thing → Run → Observe → Explain.**", 1),
        ("## Troubleshooting", 1),
        ("## Glossary", 1),
        ("## Conclusion (your notes)", 1),
    ):
        assert md.count(marker) >= least, marker
    titled = [c for c in _code_cells(nb) if _src(c).startswith("# @title Infrastructure:")]
    assert len(titled) == 7 and all(c["metadata"].get("cellView") == "form" for c in titled)
    assert nb["metadata"]["dimer"]["notebook_spec"] == "2.2"
