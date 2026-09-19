"""Offline tests for the labelled-image dataset contract, the pinned iNaturalist record table and reader, the
seeded stratified split, classification metrics with the majority and k-NN baselines, BYOD loaders (directory
and zip), the injected-runner feature / evaluation path, artifact-manifest rejections and adapt() argument
validation. No model library is loaded by the fake pipeline beyond torch tensors; the corpus is served through
an injected fetcher of small synthetic JPEGs."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile

import numpy as np
import pytest
import torch
from PIL import Image

from dinov2_feature_extraction_pipeline import (
    ARTIFACT_FORMAT,
    EMBED_DIM,
    MODEL_ID,
    MODEL_REVISION,
    SAMPLE_RECORDS,
    SAMPLE_SPLIT,
    SPECIES,
    TRANSFORMER_BLOCKS,
    WEIGHT_SHA256,
    DINOv2FeatureExtractionPipeline,
    build_sample_dataset,
    check_split_disjoint,
    class_names,
    classification_metrics,
    dataset_digest,
    fetch_sample_dataset,
    image_digest,
    knn_baseline,
    knn_predict,
    load_byod_dataset,
    majority_baseline,
    observer_overlap,
    split_dataset,
    validate_dataset,
    write_dataset_csv,
)
from dinov2_feature_extraction_pipeline import pipeline as pl
from dinov2_feature_extraction_pipeline import samples as sm
from dinov2_feature_extraction_pipeline.samples import fetch_corpus, read_corpus

COLOURS = {"red": (200, 30, 30), "green": (30, 200, 30), "blue": (30, 30, 200), "gold": (220, 200, 40)}


def _jpeg(colour, size=(64, 48), noise=0):
    image = Image.new("RGB", size, colour)
    if noise:
        px = image.load()
        px[noise % size[0], noise % size[1]] = (0, 0, 0)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=90)
    return buffer.getvalue()


def _records(n=12):
    """Twelve labelled colour swatches (three per colour), distinguishable by mean colour."""
    keys = list(COLOURS)
    out = []
    for i in range(n):
        key = keys[i % 4]
        out.append(
            {
                "id": f"r{i:03d}",
                "image": Image.open(io.BytesIO(_jpeg(COLOURS[key], noise=i + 1))).convert("RGB"),
                "label": key,
            }
        )
    return out


def _fake_transform(image):
    arr = np.asarray(image.convert("RGB").resize((8, 8)), dtype=np.float32) / 255.0
    return torch.tensor(arr).permute(2, 0, 1)


def _fake_runner(batch):
    """Mean colour of the image as a 384-d feature (first three dims carry it; the rest are zero)."""
    features = torch.zeros(batch.shape[0], EMBED_DIM)
    features[:, :3] = batch.mean(dim=(2, 3))
    return features


def _pipeline_without_model():
    return DINOv2FeatureExtractionPipeline(_fake_runner, _fake_transform, "cpu", (518, 518), "injected")


def _pin(monkeypatch, n_per_species=4):
    """Replace the pinned table with synthetic swatches (one colour per species) via an injected fetcher."""
    files = {}
    table = []
    for s, (key, _names) in enumerate(SPECIES.items()):
        colour = list(COLOURS.values())[s % 4]
        for j in range(n_per_species):
            rid = f"{key}-{j:02d}"
            data = _jpeg(colour, noise=100 * s + j + 1)
            photo_id = 1000 * (s + 1) + j
            files[photo_id] = data
            table.append(
                (
                    rid,
                    key,
                    photo_id,
                    5000 + photo_id,
                    f"user{(s * n_per_species + j) % 7}",
                    len(data),
                    hashlib.sha256(data).hexdigest(),
                )
            )
    monkeypatch.setattr(sm, "SAMPLE_RECORDS", tuple(table))
    return files


# --- pinned table and reader --------------------------------------------------------------------------


def test_pinned_record_table_is_complete_and_traceable():
    assert len(SAMPLE_RECORDS) == 180 and len(SPECIES) == 6
    assert all(len(r) == 7 and len(r[6]) == 64 and r[5] > 10_000 and r[1] in SPECIES for r in SAMPLE_RECORDS)
    assert len({r[0] for r in SAMPLE_RECORDS}) == 180 and len({r[2] for r in SAMPLE_RECORDS}) == 180
    counts = {}
    for r in SAMPLE_RECORDS:
        counts[r[1]] = counts.get(r[1], 0) + 1
    assert set(counts.values()) == {30}
    per_species_users = {}
    for r in SAMPLE_RECORDS:
        per_species_users.setdefault(r[1], []).append(r[4])
    assert all(
        len(set(u)) == len(u) for u in per_species_users.values()
    )  # one photo per observer per species
    assert sm.photo_url(SAMPLE_RECORDS[0][2]).startswith(
        "https://inaturalist-open-data.s3.amazonaws.com/photos/"
    )
    assert sum(r[5] for r in SAMPLE_RECORDS) == sm.CORPUS_BYTES
    assert sum(SAMPLE_SPLIT.values()) == 30 and set(SAMPLE_SPLIT) == {"train", "validation", "test"}


def test_fetch_corpus_verifies_each_file_and_caches(tmp_path, monkeypatch, forbid_model_imports):
    files = _pin(monkeypatch)
    calls = []

    def fetcher(url):
        calls.append(url)
        return files[int(url.rsplit("/", 2)[1])]

    corpus = fetch_corpus(cache_dir=tmp_path, fetcher=fetcher)
    assert len(corpus) == 24 and fetch_corpus(cache_dir=tmp_path, fetcher=fetcher) == corpus
    assert len(calls) == 24 and all(u.startswith(sm.CORPUS_BASE_URL) for u in calls)
    with pytest.raises(ValueError, match="pinned"):
        fetch_corpus(cache_dir=tmp_path / "other", fetcher=lambda url: b"tampered")


def test_read_corpus_decodes_records_with_provenance(tmp_path, monkeypatch, forbid_model_imports):
    files = _pin(monkeypatch)
    corpus = read_corpus(
        fetch_corpus(cache_dir=tmp_path, fetcher=lambda url: files[int(url.rsplit("/", 2)[1])])
    )
    first = corpus[0]
    assert first["id"] == "song_sparrow-00" and first["label"] == "song_sparrow"
    assert isinstance(first["image"], Image.Image) and first["image"].size == (64, 48)
    assert first["scientific_name"] == "Melospiza melodia" and first["inat_observation_url"].endswith("/6000")
    with pytest.raises(ValueError, match="missing"):
        read_corpus({})


def test_sample_split_is_stratified_seeded_and_disjoint(tmp_path, monkeypatch, forbid_model_imports):
    files = _pin(monkeypatch)
    corpus = read_corpus(
        fetch_corpus(cache_dir=tmp_path, fetcher=lambda url: files[int(url.rsplit("/", 2)[1])])
    )
    sizes = {"train": 2, "validation": 1, "test": 1}
    splits = build_sample_dataset(corpus, seed=1, sizes=sizes)
    assert {k: len(v) for k, v in splits.items()} == {"train": 12, "validation": 6, "test": 6}
    assert all(sum(r["label"] == s for r in splits["train"]) == 2 for s in SPECIES)
    assert (
        splits["train"][0]["id"] == "train-000" and splits["train"][0]["source_id"].split("-")[0] in SPECIES
    )
    assert check_split_disjoint(splits) == {"train": 12, "validation": 6, "test": 6}
    assert {r["source_id"] for r in build_sample_dataset(corpus, seed=1, sizes=sizes)["test"]} == {
        r["source_id"] for r in splits["test"]
    }
    assert {r["source_id"] for r in build_sample_dataset(corpus, seed=2, sizes=sizes)["test"]} != {
        r["source_id"] for r in splits["test"]
    }
    overlap = observer_overlap(splits)
    assert overlap["observers"] == 7 and 0 <= overlap["in_more_than_one_split"] <= 7
    with pytest.raises(ValueError, match="only"):
        build_sample_dataset(corpus, sizes={"train": 4, "validation": 1, "test": 1})
    leaky = {"train": splits["train"], "test": [{**splits["train"][0], "id": "leak"}]}
    with pytest.raises(ValueError, match="appears in both"):
        check_split_disjoint(leaky)


def test_fetch_sample_dataset_end_to_end_with_injected_fetcher(tmp_path, monkeypatch, forbid_model_imports):
    files = _pin(monkeypatch)
    splits = fetch_sample_dataset(
        cache_dir=tmp_path,
        fetcher=lambda url: files[int(url.rsplit("/", 2)[1])],
        sizes={"train": 2, "validation": 1, "test": 1},
    )
    assert validate_dataset(splits["train"])["n_records"] == 12


# --- dataset validation -------------------------------------------------------------------------------


def test_validate_dataset_reports_and_rejects(tmp_path, forbid_model_imports):
    report = validate_dataset(_records())
    assert report["n_records"] == 12 and report["classes"] == ["blue", "gold", "green", "red"]
    assert report["label_counts"] == {"blue": 3, "gold": 3, "green": 3, "red": 3} and report[
        "image_side"
    ] == {"min": 64, "max": 64}
    assert report["digest"] == dataset_digest(report["records"]) and report["model_id"] == MODEL_ID
    path = tmp_path / "one.jpg"
    path.write_bytes(_jpeg(COLOURS["red"]))
    by_path = validate_dataset([{"id": "p", "image": str(path), "label": "red"}, *_records()[1:]])
    assert isinstance(by_path["records"][0]["image"], Image.Image)
    assert class_names(_records()) == ["blue", "gold", "green", "red"]
    good = _records()
    for bad, message in (
        (good[:7], "8..20000"),
        ([{**good[0], "id": "bad id"}, *good[1:]], "id must match"),
        ([{**good[0], "id": good[1]["id"]}, *good[1:]], "duplicate id"),
        ([{**good[0], "image": "nope.jpg"}, *good[1:]], "image file not found"),
        ([{**good[0], "image": 5}, *good[1:]], "PIL.Image.Image"),
        ([{**good[0], "image": Image.new("RGB", (5000, 10))}, *good[1:]], "MAX_IMAGE_SIDE"),
        ([{**good[0], "label": " "}, *good[1:]], "label must be"),
        ([{**r, "label": "same"} for r in good], "2..100"),
        ([{"id": "a", "image": good[0]["image"]}, *good[1:]], "missing 'label'"),
        (["not a mapping", *good[1:]], "must be a mapping"),
        ({"a": 1}, "must be a list"),
    ):
        with pytest.raises(ValueError, match=message):
            validate_dataset(bad)
    with pytest.raises(ValueError, match="at least 2"):
        class_names([{**good[0], "label": "x"}])


def test_split_dataset_deduplicates_and_is_seeded(forbid_model_imports):
    records = [*_records(20), {**_records()[0], "id": "dup"}]
    splits = split_dataset(records, val_fraction=0.2, test_fraction=0.2, seed=3)
    assert (
        sum(len(v) for v in splits.values()) == 20
        and len(splits["test"]) == 4
        and len(splits["validation"]) == 4
    )
    assert check_split_disjoint(splits)
    again = split_dataset(records, val_fraction=0.2, test_fraction=0.2, seed=3)
    assert [r["id"] for r in again["test"]] == [r["id"] for r in splits["test"]]
    assert image_digest(records[0]["image"]) == image_digest(records[-1]["image"])
    with pytest.raises(ValueError, match="fractions"):
        split_dataset(records, val_fraction=0.5, test_fraction=0.6)
    with pytest.raises(ValueError, match="at least"):
        split_dataset(records, val_fraction=0.0, test_fraction=0.9)


# --- metrics and baselines ----------------------------------------------------------------------------


def test_classification_metrics_and_majority_baseline(forbid_model_imports):
    classes = ["a", "b", "c"]
    metrics = classification_metrics(["a", "a", "b", "c"], ["a", "b", "b", "b"], classes)
    assert (
        metrics["accuracy"] == 0.5
        and metrics["n"] == 4
        and metrics["confusion"] == [[1, 1, 0], [0, 1, 0], [0, 1, 0]]
    )
    assert metrics["per_class"]["a"] == {
        "precision": 1.0,
        "recall": 0.5,
        "f1": pytest.approx(2 / 3),
        "support": 2,
    }
    assert metrics["per_class"]["c"]["f1"] == 0.0 and metrics["macro_f1"] == pytest.approx(
        (2 / 3 + 0.5 + 0.0) / 3
    )
    with pytest.raises(ValueError, match="not in classes"):
        classification_metrics(["a"], ["z"], classes)
    with pytest.raises(ValueError, match="align"):
        classification_metrics(["a"], ["a", "b"], classes)
    majority = majority_baseline(["a", "a", "b"], ["a", "b", "c"], classes)
    assert majority["accuracy"] == pytest.approx(1 / 3) and "'a'" in majority["baseline"]


def test_knn_baseline_votes_by_cosine():
    train = np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [0, 0.9, 0.1]], dtype=np.float32)
    labels = ["x", "x", "y", "y"]
    assert knn_predict(train, labels, np.array([[0.8, 0.2, 0], [0.1, 1, 0]]), k=3) == ["x", "y"]
    result = knn_baseline(
        train, labels, np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32), ["x", "y"], ["x", "y"], k=1
    )
    assert result["accuracy"] == 1.0 and result["k"] == 1 and "1-NN" in result["baseline"]
    with pytest.raises(ValueError, match="k must be"):
        knn_predict(train, labels, train, k=0)
    with pytest.raises(ValueError, match="same width"):
        knn_predict(train, labels, np.zeros((1, 2)), k=1)
    pipe = _pipeline_without_model()
    baseline = pipe.knn_baseline(_records()[:8], _records()[8:], k=1)
    assert baseline["accuracy"] == 1.0 and baseline["n"] == 4


# --- BYOD loaders and CSV -----------------------------------------------------------------------------


def test_byod_directory_and_zip_round_trip_and_rejections(tmp_path, forbid_model_imports):
    records = _records()
    folder = tmp_path / "byod"
    folder.mkdir()
    rows = []
    for r in records:
        r["image"].save(folder / f"{r['id']}.jpg", "JPEG", quality=95)
        rows.append({"id": r["id"], "file": f"{r['id']}.jpg", "label": r["label"]})
    with open(folder / "labels.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "file", "label"])
        writer.writeheader()
        writer.writerows(rows)
    loaded = load_byod_dataset(folder)
    assert [r["id"] for r in loaded] == [r["id"] for r in records] and loaded[0]["image"].size == (64, 48)
    archive = tmp_path / "byod.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for name in folder.iterdir():
            zf.write(name, f"inner/{name.name}")
    assert [r["label"] for r in load_byod_dataset(archive)] == [r["label"] for r in records]
    written = write_dataset_csv(
        [{**r, "observer": "me", "inat_photo_id": 7, "inat_observation_url": "u"} for r in records[:2]],
        tmp_path / "out.csv",
    )
    text = written.read_text(encoding="utf-8")
    assert text.startswith("id,file,label,observer,inat_observation_url") and "7.jpg" in text
    (folder / "labels.csv").write_text("id,file\nx,y\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        load_byod_dataset(folder)
    with zipfile.ZipFile(tmp_path / "nolabels.zip", "w") as zf:
        zf.writestr("a.txt", "x")
    with pytest.raises(ValueError, match="labels.csv"):
        load_byod_dataset(tmp_path / "nolabels.zip")
    with pytest.raises(ValueError, match="directory or a .zip"):
        load_byod_dataset(tmp_path / "missing.tar")


# --- features, evaluation, adaptation and artifacts without a model -----------------------------------


def test_features_and_evaluation_through_the_public_contract(tmp_path):
    pipe = _pipeline_without_model()
    feats = pipe.features(_records(40))  # more than one MAX_BATCH
    assert feats.shape == (40, EMBED_DIM) and np.allclose(np.linalg.norm(feats, axis=1), 1.0, atol=1e-5)
    with pytest.raises(ValueError, match="no classification head"):
        pipe.evaluate(_records())
    with pytest.raises(ValueError, match="no classification head"):
        pipe.classify(_records()[0]["image"])


def test_adapt_and_artifacts_need_a_loaded_model(tmp_path, forbid_model_imports):
    pipe = _pipeline_without_model()
    with pytest.raises(ValueError, match="probe_steps"):
        pipe.adapt(_records(), probe_steps=0)
    with pytest.raises(ValueError, match="probe_lr"):
        pipe.adapt(_records(), probe_lr=2.0)
    with pytest.raises(ValueError, match="epochs"):
        pipe.adapt(_records(), epochs=-1)
    with pytest.raises(ValueError, match="lr"):
        pipe.adapt(_records(), lr=1.0)
    with pytest.raises(ValueError, match="batch_size"):
        pipe.adapt(_records(), batch_size=0)
    with pytest.raises(ValueError, match="trainable_blocks"):
        pipe.adapt(_records(), trainable_blocks=TRANSFORMER_BLOCKS + 1)
    with pytest.raises(ValueError, match="from_pretrained"):
        pipe.adapt(_records())
    with pytest.raises(ValueError, match="call adapt"):
        pipe.save_artifact(tmp_path)


def test_load_artifact_rejects_bad_manifests_before_touching_weights(tmp_path, forbid_model_imports):
    pipe = _pipeline_without_model()
    manifest = {
        "format": ARTIFACT_FORMAT,
        "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION, "weight_sha256": WEIGHT_SHA256},
        "format_version": pl.ARTIFACT_FORMAT_VERSION,
        "files": [{"path": pl.ARTIFACT_WEIGHTS_NAME, "bytes": 1, "sha256": "0" * 64}],
        "tensors": ["blocks.11.mlp.fc1.weight", "head.bias", "head.weight"],
        "adapter": {"classes": ["a", "b"], "policy": pl.POLICY_UNFROZEN.format(k=1), "trainable_blocks": 1},
    }
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps({**manifest, "format": "other"}))
    with pytest.raises(ValueError, match="artifact format"):
        pipe.load_artifact(tmp_path)
    bad_base = {**manifest, "base_model": {**manifest["base_model"], "weight_sha256": "0" * 64}}
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(bad_base))
    with pytest.raises(ValueError, match="different base model"):
        pipe.load_artifact(tmp_path)
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(FileNotFoundError, match="artifact weights missing"):
        pipe.load_artifact(tmp_path)
    (tmp_path / pl.ARTIFACT_WEIGHTS_NAME).write_bytes(b"x")
    with pytest.raises(ValueError, match="digest or size mismatch"):
        pipe.load_artifact(tmp_path)
    no_classes = {**manifest, "adapter": {}}
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(no_classes))
    with pytest.raises(ValueError, match="at least two unique classes"):
        pipe.load_artifact(tmp_path)


def test_load_artifact_refuses_unsupported_versions_extra_files_traversal_and_policies(
    tmp_path, forbid_model_imports
):
    pipe = _pipeline_without_model()
    good = {
        "format": ARTIFACT_FORMAT,
        "format_version": pl.ARTIFACT_FORMAT_VERSION,
        "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION, "weight_sha256": WEIGHT_SHA256},
        "files": [{"path": pl.ARTIFACT_WEIGHTS_NAME, "bytes": 1, "sha256": "0" * 64}],
        "tensors": ["head.bias", "head.weight"],
        "adapter": {"classes": ["a", "b"], "policy": pl.POLICY_FROZEN, "trainable_blocks": 0},
    }

    def write(manifest):
        (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest))

    write({**good, "format_version": "0.9"})
    with pytest.raises(ValueError, match="format_version"):
        pipe.load_artifact(tmp_path)
    write({**good, "files": good["files"] * 2})
    with pytest.raises(ValueError, match="exactly one file"):
        pipe.load_artifact(tmp_path)
    write({**good, "files": [{**good["files"][0], "path": "../" + pl.ARTIFACT_WEIGHTS_NAME}]})
    with pytest.raises(ValueError, match="must name exactly|inside the artifact directory"):
        pipe.load_artifact(tmp_path)
    write({**good, "base_model": {**good["base_model"], "weight_file": "pytorch_model.bin"}})
    with pytest.raises(ValueError, match="different base weight file"):
        pipe.load_artifact(tmp_path)
    write({**good, "adapter": {**good["adapter"], "classes": ["a", "a"]}})
    with pytest.raises(ValueError, match="unique classes"):
        pipe.load_artifact(tmp_path)
    write({**good, "adapter": {**good["adapter"], "policy": "something else"}})
    with pytest.raises(ValueError, match="not a canonical policy"):
        pipe.load_artifact(tmp_path)
    write(
        {
            **good,
            "adapter": {**good["adapter"], "policy": pl.POLICY_UNFROZEN.format(k=2), "trainable_blocks": 1},
        }
    )
    with pytest.raises(ValueError, match="not a canonical policy"):
        pipe.load_artifact(tmp_path)
    write({**good, "adapter": {**good["adapter"], "trainable_blocks": 99}})
    with pytest.raises(ValueError, match="trainable_blocks"):
        pipe.load_artifact(tmp_path)
    write(good)  # every manifest check passes; the weights file is still missing, and no model was imported
    with pytest.raises(FileNotFoundError, match="artifact weights missing"):
        pipe.load_artifact(tmp_path)
