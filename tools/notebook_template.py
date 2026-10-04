"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.2 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (pipeline.py, metrics.py, samples.py), and the model pin/stage/verify cells are produced by
the generator from repository sources so they cannot drift from the package.

This template configures an E2E supervised-adaptation workflow: the pinned DINOv2 ViT-S/14 snapshot is
digest-verified and loaded, a digest-pinned real image corpus (CC0 iNaturalist bird photographs) is fetched,
validated and split, the embedding contract is exercised, the frozen features are scored by k-NN and by a linear
probe (the frozen policy) beside a majority floor, a bounded unfreeze of the last blocks is trained and selected
against the probe on validation (the unfrozen policy), the held-out split is scored, and the adapter is exported
and reloaded.

Review fixes (Notebook Review Framework v1, review PR #9, DV2-M1..M3 / DV2-m1..m4): the runtime is the fleet's uv
isolated environment (no in-kernel install, no restart); every learner section that needs DINOv2 itself starts from
the pretrained base (`reset_to_pretrained`, and `adapt` restores the base on every call), so the optional
experiments and the BYOD rerun are comparable and the reload parity holds; the frozen column of Section 9 is the
Section 6 probe's own predictions; BYOD splits are validated with the minimums `adapt` applies and the errors name
the split, the labels.csv row and the file; measured times name their run; and the guided layer (who it is for,
how to use, roadmap, predictions, worked answers, a change-one-thing activity, troubleshooting, glossary,
conclusion) is added.

Code cells are written with single braces and escaped by ``_py`` for the generator's ``str.format`` pass; ``@STEM@``
becomes the output stem. Markdown cells are formatted too, so they contain no literal braces.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering


def _py(code: str) -> str:
    """Escape a code cell for the generator's ``str.format`` pass; ``@STEM@`` stands for ``{stem}``."""
    return code.replace("{", "{{").replace("}", "}}").replace("@STEM@", "{stem}")


_DATA_CODE = _py(
    """import hashlib
import io
import json
import time

USE_BYOD = False  # @param {type:"boolean"}
BYOD_PATH = ''  # @param {type:"string"}
SPLIT_SEED = 42  # @param {type:"integer"}

os.makedirs('outputs', exist_ok=True)
t0 = time.perf_counter()
if USE_BYOD:
    if BYOD_PATH:
        byod_path = Path(BYOD_PATH)
        file_name = byod_path.name
    else:
        from google.colab import files
        uploaded = files.upload()
        if len(uploaded) != 1:
            raise ValueError(f'Upload exactly one .zip holding labels.csv and the images (got {len(uploaded)}). Outside Colab, set BYOD_PATH to a .zip or a directory instead.')
        file_name, payload = next(iter(uploaded.items()))
        byod_path = Path('work') / file_name
        byod_path.parent.mkdir(parents=True, exist_ok=True)
        byod_path.write_bytes(payload)
    records = load_byod_dataset(byod_path)
    splits = split_dataset(records, seed=SPLIT_SEED)
    data_source = 'BYOD (' + file_name + ')'
    raw_count = {'byod': len(records)}
else:
    corpus = read_corpus(fetch_corpus(cache_dir='weights/inat-birds'))
    raw_count = {'photographs': len(corpus), 'species': len(SPECIES)}
    splits = build_sample_dataset(corpus, seed=SPLIT_SEED)
    data_source = f'{CORPUS_NAME} ({CORPUS_RELEASE}; {CORPUS_LICENSE})'
fetch_seconds = round(time.perf_counter() - t0, 1)
train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']
# Training split: 8..20,000 records; validation and test: at least 1 (what adapt and evaluate accept). A refusal names the split.
dataset_manifests = validate_splits(splits)
classes = class_names(train_records)
disjoint = check_split_disjoint(splits)
overlap = observer_overlap(splits)
write_dataset_csv(train_records, 'outputs/@STEM@_train.csv')
print({'data_source': data_source, 'raw': raw_count, 'splits': disjoint, 'classes': classes, 'observer_overlap': overlap, 'fetch_seconds': fetch_seconds, 'corpus_bytes': CORPUS_BYTES})
for name, manifest in dataset_manifests.items():
    print({name: {'n': manifest['n_records'], 'label_counts': manifest['label_counts'], 'image_side': manifest['image_side'], 'digest': manifest['digest'][:16] + '...'}})
example = train_records[0]
print({'example': {k: example[k] for k in ('id', 'label', 'observer', 'inat_observation_url') if k in example}, 'size': example['image'].size})

probes = {
    'duplicate id': [{**r, 'id': 'same'} for r in train_records[:8]],
    'oversized image': [{**train_records[0], 'image': Image.new('RGB', (MAX_IMAGE_SIDE + 1, 8))}, *train_records[1:8]],
    'single class': [{**r, 'label': 'bird'} for r in train_records[:8]],
    'too small': train_records[:3],
}
for name, probe in probes.items():
    try:
        validate_dataset(probe)
        print({'probe': name, 'verdict': 'accepted'})
    except (TypeError, ValueError) as exc:
        print({'probe': name, 'rejected': str(exc)[:110]})"""
)

_EMBED_CODE = _py(
    """# Start from DINOv2 itself: undo any adaptation an earlier run of Sections 6-10 left on `pipe`.
pipe.reset_to_pretrained()
probe_records = test_records[:3]
print({'ceilings': {'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MAX_BATCH': MAX_BATCH}, 'contract': {'EMBED_DIM': EMBED_DIM, 'POOLING': POOLING, 'NORMALIZED': NORMALIZED, 'TRANSFORMER_BLOCKS': TRANSFORMER_BLOCKS, 'PARAMETER_COUNT': PARAMETER_COUNT}})
input_manifest = validate_inputs([r['image'] for r in probe_records], names=[r['id'] for r in probe_records])
try:
    validate_inputs(Image.new('RGB', (MAX_IMAGE_SIDE + 1, 8)))
except ValueError as exc:
    input_manifest['findings'].append({'input': 'oversized-probe', 'verdict': 'rejected', 'message': str(exc)})
with open('outputs/@STEM@_input_manifest.json', 'w', encoding='utf-8') as handle:
    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)
started = time.perf_counter()
embedding_result = pipe.embed([r['image'] for r in probe_records])
embed_seconds = round(time.perf_counter() - started, 3)
vectors = np.asarray(embedding_result['embeddings'], dtype=np.float32)
checks = {
    'one_vector_per_image': vectors.shape == (len(probe_records), EMBED_DIM) and embedding_result['dim'] == EMBED_DIM,
    'unit_norm': bool(embedding_result['normalized']) and bool(np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)),
    'contract_fields': embedding_result['pooling'] == POOLING and embedding_result['input_size'] == [518, 518],
    'all_values_finite': bool(np.isfinite(vectors).all()),
}
if not all(checks.values()):
    raise RuntimeError(f'embed output failed a sanity check: {checks}')
# One same-label and one other-label partner for the first image (a small BYOD test split may lack one).
partners = [r for r in test_records[1:]]
same = next((r for r in partners[2:] + partners[:2] if r['label'] == probe_records[0]['label']), None)
other = next((r for r in partners[2:] + partners[:2] if r['label'] != probe_records[0]['label']), None)
cosines = {}
for name, partner in (('cosine_same_species', same), ('cosine_other_species', other)):
    if partner is not None:
        partner_vector = np.asarray(pipe.embed([partner['image']])['embeddings'][0], dtype=np.float32)
        cosines[name] = round(float(vectors[0] @ partner_vector), 4)
print({'probe_ids': [r['id'] for r in probe_records], 'probe_labels': [r['label'] for r in probe_records], 'seconds': embed_seconds, 'device': pipe.device, 'backbone': 'pretrained', 'checks': checks, 'findings': len(input_manifest['findings'])})
print({**cosines, 'note': 'one pair each; a similarity, not a score'})"""
)

_FROZEN_CODE = _py(
    """PROBE_STEPS = 300  # @param {type:"integer"}
PROBE_LR = 0.01  # @param {type:"number"}

def brief(m):
    return {'accuracy': round(m['accuracy'], 4), 'macro_f1': round(m['macro_f1'], 4), 'n': m['n']}

# The floors and the frozen policy describe DINOv2 itself, so start from the pretrained base on every run.
pipe.reset_to_pretrained()
floor = majority_baseline([r['label'] for r in train_records], [r['label'] for r in test_records], classes)
print({'majority_floor': brief(floor), 'baseline': floor['baseline']})
t0 = time.perf_counter()
baseline_knn = pipe.knn_baseline(train_records, test_records, k=5)
print({'knn_baseline': brief(baseline_knn), 'baseline': baseline_knn['baseline'], 'seconds': round(time.perf_counter() - t0, 1)})
t0 = time.perf_counter()
probe_result = pipe.adapt(train_records, val_records, probe_steps=PROBE_STEPS, probe_lr=PROBE_LR, trainable_blocks=0)
frozen_test = pipe.evaluate(test_records)
show = test_records[:6]
frozen_before = pipe.classify([r['image'] for r in show])  # the frozen column of Section 9, kept from this probe
probe_val_log_loss = probe_result['history'][0]['val']['log_loss'] if probe_result['history'][0]['val'] else None
print({'frozen_policy': probe_result['policy'], 'probe_final_loss': round(probe_result['probe_final_loss'], 4), 'validation': {k: round(v, 4) for k, v in probe_result['history'][0]['val'].items()}, 'seconds': round(time.perf_counter() - t0, 1)})
print({'frozen_policy_test': brief(frozen_test), 'log_loss': round(frozen_test['log_loss'], 4), 'verdict': frozen_test['verdict'], 'per_class_recall': {c: round(v['recall'], 2) for c, v in frozen_test['per_class'].items()}})
print({'definitions': frozen_test['definitions']})
assert frozen_test['accuracy'] > floor['accuracy'] and probe_result['policy'].startswith('frozen')"""
)

_UNFROZEN_CODE = _py(
    """EPOCHS = 4  # @param {type:"integer"}
LEARNING_RATE = 3e-5  # @param {type:"number"}
BATCH_SIZE = 8  # @param {type:"integer"}
TRAINABLE_BLOCKS = 2  # @param {type:"integer"}

def report(entry):
    row = {'epoch': entry['epoch'], 'stage': entry['stage'], 'train_loss': round(entry['train_loss'], 4)}
    if entry.get('val'):
        row['val_log_loss'] = round(entry['val']['log_loss'], 4)
        row['val_accuracy'] = round(entry['val']['accuracy'], 4)
        row['val_macro_f1'] = round(entry['val']['macro_f1'], 4)
    print(row)

# adapt() restores the pretrained base before it trains, so this run does not build on an earlier one.
t0 = time.perf_counter()
adapt_result = pipe.adapt(train_records, val_records, probe_steps=PROBE_STEPS, probe_lr=PROBE_LR, trainable_blocks=TRAINABLE_BLOCKS, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, progress=report)
adapt_seconds = round(time.perf_counter() - t0, 1)
epoch0 = adapt_result['history'][0]['val']
if epoch0 and probe_val_log_loss is not None:
    start_check = {'epoch0_val_log_loss': round(epoch0['log_loss'], 6), 'section6_probe_val_log_loss': round(probe_val_log_loss, 6), 'abs_difference': abs(epoch0['log_loss'] - probe_val_log_loss)}
    print({'starts_from_pretrained_base': start_check})
    if start_check['abs_difference'] > 1e-3:
        raise RuntimeError(f'Epoch 0 is not the Section 6 probe ({start_check}): run Sections 6 and 7 again in order.')
print({'selected_policy': adapt_result['policy'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'trainable_head': adapt_result['n_trainable_head'], 'trainable_blocks': adapt_result['n_trainable_blocks'], 'total_parameters': adapt_result['n_total'], 'seconds': adapt_seconds})"""
)

_EVAL_CODE = _py(
    """adapted_test = pipe.evaluate(test_records)
adapted_val = pipe.evaluate(val_records)
comparison = {
    metric: {'majority_floor': round(floor[metric], 4), 'knn5': round(baseline_knn[metric], 4), 'frozen_policy': round(frozen_test[metric], 4), 'selected_policy': round(adapted_test[metric], 4)}
    for metric in ('accuracy', 'macro_f1')
}
comparison['log_loss'] = {'frozen_policy': round(frozen_test['log_loss'], 4), 'selected_policy': round(adapted_test['log_loss'], 4)}
comparison['per_class_recall'] = {c: {'knn5': round(baseline_knn['per_class'][c]['recall'], 2), 'frozen': round(frozen_test['per_class'][c]['recall'], 2), 'selected': round(adapted_test['per_class'][c]['recall'], 2)} for c in classes}
comparison['delta_vs_frozen'] = {metric: round(adapted_test[metric] - frozen_test[metric], 4) for metric in ('accuracy', 'macro_f1')}
comparison['selected_policy'] = adapt_result['policy']
for metric, row in comparison.items():
    print({metric: row})
print({'confusion_selected': adapted_test['confusion'], 'classes': classes})
evaluation_report_payload = {
    'model': {'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY},
    'data_source': data_source,
    'dataset_digests': {name: manifest['digest'] for name, manifest in dataset_manifests.items()},
    'splits': disjoint,
    'observer_overlap': overlap,
    'classes': classes,
    'baselines': {'majority_floor': floor, 'knn5': baseline_knn},
    'frozen_policy': {'adaptation': {k: v for k, v in probe_result.items() if k not in ('history', 'trainable_names')}, 'history': probe_result['history'], 'test': frozen_test},
    'validation_metrics': adapted_val,
    'test_metrics': adapted_test,
    'comparison': comparison,
    'adaptation': {k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')},
    'history': adapt_result['history'],
    'adaptation_seconds': adapt_seconds,
}
with open('outputs/@STEM@_evaluation_report.json', 'w', encoding='utf-8') as f:
    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False)
assert adapted_test['accuracy'] > floor['accuracy']
print({'report': 'outputs/@STEM@_evaluation_report.json'})

# One row per Section 7 run in this session, so a changed setting is read next to the default run.
run_history = globals().get('run_history', [])
run_history.append({'run': len(run_history) + 1, 'trainable_blocks': TRAINABLE_BLOCKS, 'learning_rate': LEARNING_RATE, 'epochs': EPOCHS, 'data': 'BYOD' if USE_BYOD else 'sample', 'probe_val_log_loss': round(probe_val_log_loss, 4) if probe_val_log_loss is not None else None, 'best_epoch': adapt_result['best_epoch'], 'selected_policy': adapt_result['policy'], 'test_accuracy': round(adapted_test['accuracy'], 4), 'test_macro_f1': round(adapted_test['macro_f1'], 4), 'frozen_test_accuracy': round(frozen_test['accuracy'], 4)})"""
)

_EXPORT_CODE = _py(
    """import csv
import shutil

after = pipe.classify([r['image'] for r in show])
before = frozen_before  # Section 6's probe on the pretrained base
rows = []
for i, record in enumerate(show):
    rows.append({'id': record['id'], 'gold': record['label'], 'frozen_label': before['labels'][i], 'frozen_top_probability': round(max(before['probabilities'][i]), 4), 'selected_label': after['labels'][i], 'selected_top_probability': round(max(after['probabilities'][i]), 4), 'observer': record.get('observer', ''), 'inat_observation_url': record.get('inat_observation_url', '')})
    print({k: rows[-1][k] for k in ('id', 'gold', 'frozen_label', 'frozen_top_probability', 'selected_label', 'selected_top_probability')})
single_report = evaluation_report(embedding_result, sample_kind='three iNaturalist test photographs' if not USE_BYOD else 'three BYOD test images')
print({'batch_report_verdict': single_report['verdict'], 'selected_policy': after['policy'], 'predictions_changed': sum(r['frozen_label'] != r['selected_label'] for r in rows), 'of': len(rows)})
with open('outputs/@STEM@_predictions.csv', 'w', encoding='utf-8', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

artifact_dir = Path('outputs/@STEM@_adapter')
shutil.rmtree(artifact_dir, ignore_errors=True)
pipe.save_artifact(artifact_dir, metadata={'tutorial': '@STEM@', 'data_source': data_source})
artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))
print({'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'policy': artifact_manifest['adapter']['policy'], 'classes': artifact_manifest['adapter']['classes'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...'})

reloaded = DINOv2FeatureExtractionPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)
reloaded_after = reloaded.classify([r['image'] for r in show])
reloaded_test = reloaded.evaluate(test_records)
parity = {'probabilities_identical': reloaded_after['probabilities'] == after['probabilities'], 'accuracy_in_memory': round(adapted_test['accuracy'], 6), 'accuracy_reloaded': round(reloaded_test['accuracy'], 6), 'classes_identical': reloaded.classes == pipe.classes}
print({'reload_parity': parity, 'reloaded_policy': reloaded.adapter['policy'], 'reloaded_best_epoch': reloaded.adapter['best_epoch']})
assert parity['probabilities_identical'] and parity['classes_identical'] and abs(adapted_test['accuracy'] - reloaded_test['accuracy']) < 1e-9
del reloaded

weight_entry = next(entry for entry in snapshot['files'] if entry['path'] == WEIGHTS_FILE)
result_payload = {
    'notebook_source': NOTEBOOK_SOURCE,
    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],
    'model_id': MODEL_ID,
    'model_revision': MODEL_REVISION,
    'model_license': MODEL_LICENSE,
    'snapshot': {'path': str(WEIGHTS_DIR), 'files': len(snapshot['files']), 'total_bytes': snapshot.get('totalBytes'), 'fetched_this_run': fetched, 'weight_file': WEIGHTS_FILE, 'weight_format': 'safetensors, digest-verified', 'weight_sha256': weight_entry['sha256']},
    'data_source': data_source,
    'corpus': {'name': CORPUS_NAME, 'release': CORPUS_RELEASE, 'base_url': CORPUS_BASE_URL, 'photographs': len(SAMPLE_RECORDS), 'bytes': CORPUS_BYTES, 'license': CORPUS_LICENSE, 'species': SPECIES},
    'inference_contract': {'input_manifest': input_manifest, 'sanity_checks': checks, 'probe_ids': [r['id'] for r in probe_records], 'seconds': embed_seconds},
    'comparison': comparison,
    'run_history': run_history,
    'predictions_before_after': rows,
    'batch_report': single_report,
    'artifact': {'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes'], 'tensors': len(artifact_manifest['tensors']), 'policy': artifact_manifest['adapter']['policy']},
    'reload_parity': parity,
    'runtime': {'python': platform.python_version(), 'torch': torch.__version__, 'timm': timm.__version__, 'device': pipe.device, 'dtype': 'float32', 'source': pipe.source},
}
with open('outputs/@STEM@_result.json', 'w', encoding='utf-8') as handle:
    json.dump(result_payload, handle, indent=2, ensure_ascii=False)
print(sorted(os.listdir('outputs')))"""
)

_ACTIVITY_CODE = _py(
    """# Section 8 adds one row per Section 7 run in this session; this cell only prints them.
columns = ['run', 'trainable_blocks', 'learning_rate', 'epochs', 'data', 'probe_val_log_loss', 'best_epoch', 'selected_policy', 'test_accuracy', 'test_macro_f1', 'frozen_test_accuracy']
print(' | '.join(columns))
for row in run_history:
    print(' | '.join(str(row[column]) for column in columns))
if len(run_history) == 1:
    print('One run so far. Set TRAINABLE_BLOCKS = 1 in Section 7, select that cell and choose Runtime > Run after; this table then shows both runs.')"""
)

TEMPLATE = {
    "package": "dinov2_feature_extraction_pipeline",
    "repo_name": "dinov2-feature-extraction-pipeline",
    "stem": "dinov2_feature_extraction",
    "notebook_name": "dinov2_feature_extraction_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "infrastructure_labels": True,
    "collapse_model_cell": True,
    "isolated_runtime": True,
    # The fleet's uv isolated-environment mechanism (ast-audio-classification-pipeline / bioclip2-biodiversity-pipeline):
    # managed CPython, a size- and SHA-256-verified uv wheel, and a lock compiled from the pyproject pins with
    # `uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes
    # --only-binary :all: -o tutorials/requirements-colab.lock.txt`. The pins equal convnext-classification-pipeline's,
    # so the lock is that repository's (625fd1b) with the requesting project renamed.
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime builds an isolated environment from the hash-locked pins (the "
        "kernel's own packages are left alone, so no restart is needed), stages and digest-verifies the "
        "pinned DINOv2 ViT-S/14 snapshot (safetensors, 88 MB), fetches 180 digest-pinned CC0 iNaturalist photographs of six "
        "bird species from the iNaturalist open-data bucket (19 MB, no credential), validates them and draws 108 / 24 / 48 "
        "training, validation and test images by a seeded stratified split, embeds three test images through the "
        "inference contract with an input manifest and a rejection probe, scores the frozen features on the test split by "
        "a majority floor, a cosine 5-NN vote and a linear probe (the **frozen policy**), trains a bounded unfreeze of the "
        "last two transformer blocks with the head and selects between it and the probe by validation log-loss (the "
        "**unfrozen policy**), scores the held-out split with the selected model, prints predictions before and after, "
        "exports the head and any trained blocks as safetensors with a manifest, and reloads that artifact into a fresh "
        "pipeline to verify parity. The default path needs no repository clone, no DIMER worker or service, no credential, "
        "no upload dialog and no configuration edit (NOTEBOOK_SPEC 2.2 §5). Measured times come from the previous notebook "
        "version, whose model stages are the same: a Kaggle Tesla T4 run took 293.8 s of cell time (2026-09-19, including a "
        "pinned in-kernel install this version no longer does) and a local Windows CPU pre-flight took 183.7 s with the files "
        "pre-staged (2026-09-19). Building the isolated environment (PyTorch with its CUDA libraries) and downloading the "
        "checkpoint and the photographs come on top and usually take a few minutes (an estimate; no run of this version is "
        "recorded yet)."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4, select that cell and choose "
        "**Runtime → Run after** to supply your own labelled images as a `.zip` holding `labels.csv` (columns `id`, `file`, "
        "`label`) beside the image files — through the upload dialog on Colab, or as a path in `BYOD_PATH` (a `.zip` or a "
        "directory) on any runtime. Images are decoded from the archive, never extracted to disk. They pass through the same "
        "validation, seeded stratified image-disjoint split, floors, frozen-policy probe, unfrozen-policy training and "
        "selection, held-out evaluation, prediction, artifact export and reload-parity cells as the iNaturalist sample, and "
        "each of those cells starts again from the pretrained DINOv2 weights, not from the bird-adapted model of the default "
        "run. The expected schema, the minimum size and the ceilings are stated in the Prerequisites and in Section 4, and "
        "uploaded files stay inside this runtime. BYOD is optional and never part of the default path."
    ),
    "pipeline_class": "DINOv2FeatureExtractionPipeline",
    "weights_key": "vit-small-dinov2",
    "modules": ["pipeline.py", "metrics.py", "samples.py"],
    "entry_module": "pipeline.py",
    "identity_names": {},
    "runtime_imports": ["torch", "timm"],
    "title": "DINOv2 ViT-S/14 — DIMER E2E supervised adaptation tutorial: linear probe vs bounded unfreeze on bird photographs (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/dinov2-feature-extraction-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/dinov2-feature-extraction-pipeline/blob/main/tutorials/dinov2_feature_extraction_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-timm%2Fvit__small__patch14__dinov2.lvd142m-ffcc4d?style=flat",
            "https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-facebookresearch%2Fdinov2-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/facebookresearch/dinov2",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-2304.07193-b31b1b.svg", "https://arxiv.org/abs/2304.07193"),
    ],
    "capability": "self-supervised image feature extraction (one 384-d embedding per image) and bounded supervised adaptation — a linear probe on the frozen features with an optional unfreeze of the last transformer blocks — measured by held-out accuracy and macro-F1, using the pinned DINOv2 ViT-S/14 `lvd142m` weights",
    "intro": (
        "At inference each image is resized so its shorter side is 518 px, centre-cropped to 518 x 518, normalised with "
        "the ImageNet mean/std from the snapshot config, and passed through the ViT-S/14 backbone with no classifier "
        "head; the class token after the final LayerNorm is taken as the image's feature (`POOLING = \"cls\"`) and "
        "L2-normalised (`NORMALIZED = True`), so a dot product between two vectors is their cosine similarity. "
        "**Embeddings are representations, not predictions:** `embed` returns no label, no score and no threshold, and "
        "the carried pipeline module adds snapshot verification, input validation, the class-token pooling choice, L2 "
        "normalisation, a fixed output contract and the `validate_inputs` and `evaluation_report` helpers.\n\n"
        "What this notebook adds to inference is **supervised adaptation under an explicit frozen-vs-unfrozen policy**. "
        "The dataset is real: 180 CC0-licensed, research-grade iNaturalist photographs of six common North American "
        "birds (30 per species, one per observer per species), chosen a priori and pinned by photo id, byte size and "
        "SHA-256, fetched from the iNaturalist open-data bucket at run time and refused on any mismatch; every record "
        "keeps its observation URL and observer login. The carried `metrics.py` scores predictions by **accuracy** and "
        "**macro-F1** with per-class recall and a confusion matrix; a **majority floor** and a **cosine 5-NN vote** over "
        "the frozen features frame the numbers. The **frozen policy** trains only a linear head on the frozen features "
        "(a linear probe); the **unfrozen policy** continues from that probe by training the last transformer blocks with "
        "the head end to end, and the epoch with the lowest validation log-loss — which may be the probe itself — is "
        "kept. Every run of these stages starts from the pretrained DINOv2 weights: `pipe.adapt` restores them before it "
        "trains, and Sections 5 and 6 call `pipe.reset_to_pretrained()`, so a rerun with other settings never builds on "
        "an earlier adaptation. The adaptation question is whether unfreezing buys anything over the probe on 108 "
        "training photographs. Nothing here is a quality claim about your images: it is one seeded split of one small "
        "corpus.\n\n"
        "**Who this is for.** A learner who knows basic Python and NumPy, has met the idea of an embedding vector and of a "
        "classifier, and wants to see how a self-supervised image backbone is measured and adapted honestly: floors "
        "first, a cheap linear probe, a bounded unfreeze, selection on validation and one look at the held-out split. No "
        "prior experience with DINOv2, vision transformers, timm or fine-tuning is assumed; each term is explained where "
        "it is first used and again in the **Glossary** at the end. No GPU is required (CPU works; a GPU is used "
        "automatically).\n\n"
        "**Input → Model → Output.**\n\n"
        "| | Embedding (Section 5) | Adaptation (Sections 6–9) |\n"
        "|---|---|---|\n"
        "| Input | RGB images (any size up to 4,096 px), resized and centre-cropped to 518 × 518 | labelled photographs `id`, `image`, `label`, split into training, validation and test |\n"
        "| Model | DINOv2 ViT-S/14 backbone, class token, L2-normalised; no head | the same backbone + a new linear head (frozen policy), optionally with the last blocks unfrozen (unfrozen policy) |\n"
        "| Output | one unit-length 384-d vector per image; no label, no score | held-out accuracy and macro-F1 beside a majority floor and a k-NN vote, predictions, and a safetensors adapter that reloads with identical probabilities |\n\n"
        "**How to use this notebook.** Choose a runtime (Colab, Kaggle or Linux Jupyter; a GPU is faster, CPU works), then "
        "**Runtime → Run all**. Sections 1–3 are **infrastructure** — the isolated environment, the carried code and the "
        "model verification — and can be run without study; their code is collapsed. The learning path starts in Section 4. "
        "Form fields (`# @param`) are the only values meant to be edited; the defaults reproduce the default path. Each "
        "learner section states what it does and asks you to **predict** before it runs; the next section opens with "
        "**What to notice** and a collapsible **Check your reasoning** block with a worked answer. Each optional experiment "
        "names the field to change and the cell to re-run from (**Runtime → Run after**). Section 10 is a "
        "**change-one-thing activity**. **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. "
        "Your notes are optional.\n\n"
        "**Roadmap:** *core concepts* — 4 the photographs and their split → 5 the embedding contract → 6 floors and the "
        "frozen policy (linear probe); *evaluation practice* — 7 the unfrozen policy and selection on validation → 8 the "
        "held-out comparison → 9 predictions, export and reload (*engineering*) → 10 **change one thing: unfreeze one "
        "block** → conclude."
    ),
    "learning_objectives": (
        "by the end you should be able to (1) say what a DINOv2 embedding is and why a cosine between two embeddings is a "
        "similarity, not a score (Section 5); (2) read accuracy and macro-F1 beside a majority floor and a k-NN vote and "
        "say what each reference point rules out (Section 6); (3) explain the frozen and unfrozen policies, how many "
        "parameters each trains, and why validation log-loss — not test accuracy — chooses between them (Section 7); "
        "(4) judge whether a held-out difference on 48 photographs supports a claim (Section 8); (5) check that an "
        "exported adapter reproduces the evaluated model (Section 9); and (6) predict, measure and explain what changes "
        "when one setting of the unfreeze changes, with every run starting from the same pretrained weights (Section 10)."
    ),
    "exclusions": (
        "object detection, segmentation, captioning, image-text comparison, patch-level (dense) features, attention maps, "
        "data augmentation, full-backbone or patch-embedding training, any similarity search or clustering quality claim, "
        "and any claim that six bird species from one photo site stand in for your images. The repository exposes none of "
        "these."
    ),
    "prerequisites": [
        "- **Learner:** basic Python and NumPy; what an embedding vector is; Colab or Jupyter familiarity. The notebook explains the linear probe, accuracy and macro-F1, the majority floor, the k-NN vote, log-loss, the frozen and unfrozen policies, validation-based selection and reload parity where they are first used; the Glossary repeats them.",
        "- **Runtime:** a fresh **Linux x86_64** runtime — Google Colab, Kaggle or Linux Jupyter. Section 1 builds its own Python 3.12.12 environment from a hash-locked list of manylinux wheels, so the kernel's own Python version does not matter, and a Windows or macOS kernel is not supported (Section 1 stops with that message). The default path runs on CPU and uses CUDA automatically when available; float32 on both. Each 518 px image costs about 46.8 GMACs (upstream card). Time, from the previous notebook version's local Windows CPU pre-flight (2026-09-19, files pre-staged, 183.7 s in all): the 180-image k-NN pass, a linear probe of 18.7 s including feature extraction, and 96.6 s for the four-epoch unfreeze with its validation passes; a Kaggle Tesla T4 run of that version took 293.8 s including its in-kernel install. The locked install (PyTorch 2.14.0 with its CUDA libraries) and the 88 MB checkpoint are the large downloads, then the 19 MB of photographs.",
        "- **Data contract:** a record is `id`, `image`, `label` — a PIL image (or a path to one) with each side 1..4,096 px (the pipeline's own ceiling; images are resized and centre-cropped to 518 px, never rejected for being small), a label of 1..64 plain characters, an id of 1..64 characters from letters, digits and `_ . : -` (the pattern `^[A-Za-z0-9_.:-]{1,64}$`), unique; 2..100 classes. Images are de-duplicated by decoded-pixel digest before splitting so the same photograph never sits in two splits.",
        "- **BYOD dataset (Section 4):** a `.zip` (or a directory, via `BYOD_PATH`) holding `labels.csv` with the columns `id`, `file`, `label` and the image files it names (PNG, JPEG, WebP, BMP and similar). It is split per class into 20 % test (at least one image per class), 15 % validation and the rest training. The **training split** needs 8..20,000 images and every split at least 2 classes, so the practical minimum is about 6 images per class in 2 classes (12 images); the validation and test splits need at least one image each. A refusal names the split, or the `labels.csv` line, id and file (a missing file or a file that is not a decodable image). Validation is structural, not semantic: nothing checks that a label is right for its image.",
        "- **Validation is structural, not semantic:** a mislabelled set is trained on without complaint; observers can contribute to more than one split (the notebook counts them) because the sample is stratified by species, not by observer.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — photographs of people or private places are exactly that. The default path uploads nothing.",
        "- **External access (data):** besides the Hub, the default path fetches 180 pinned objects (`<photo id>/medium.jpg` or `medium.jpeg`, each photo's own served extension recorded in the table, 19,183,071 bytes in total, one SHA-256 each in the carried `SAMPLE_RECORDS` table) from `inaturalist-open-data.s3.amazonaws.com` over HTTPS, each refused on any byte-size or SHA-256 mismatch before it is decoded; every photograph is CC0 by its own iNaturalist licence code and credited to its observer in the records.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Referenced corpus, validation and split\n\n"
                "`fetch_corpus` downloads the 180 pinned photographs (or reads them from the cache), refuses a byte-size "
                "or SHA-256 mismatch per file before it is decoded, and `read_corpus` turns each into a record (`id`, "
                "`image`, `label`) with its species names, observation URL and observer. `build_sample_dataset` draws 18 "
                "training, 4 validation and 8 test photographs per species by a seeded stratified shuffle; "
                "`validate_splits` then checks every record of every split against the contract (the training split needs "
                "at least 8 records, validation and test at least one, every split two classes; a refusal names the split), "
                "`check_split_disjoint` asserts no photograph (by decoded-pixel digest) appears in two splits, "
                "`observer_overlap` counts the observers who contributed to more than one split — an observation, since "
                "the sample is stratified by species — and the training labels table is written to "
                "`outputs/{stem}_train.csv` in the shape BYOD expects. Four refusal probes follow — a duplicate id, an "
                "oversized image, a single-class dataset and a dataset too small to split — each rejected before `torch` "
                "does anything. About a minute on the first run for the downloads.\n\n"
                "With `USE_BYOD = True`, your `.zip` (upload dialog, or `BYOD_PATH`) goes through `load_byod_dataset` and "
                "`split_dataset` instead (20 % test, 15 % validation per class); see the Prerequisites for the minimum "
                "size.\n\n"
                "**Predict before running:** the test split has 48 photographs. By how many percentage points does one "
                "photograph moved from wrong to right change the test accuracy?"
            ),
            "code": _DATA_CODE,
        },
        {
            "md": (
                "**What to notice:** 180 photographs, six classes of 30, splits 108 / 24 / 48 with three digests, the "
                "observer overlap, and four `rejected` probe lines.\n\n"
                "<details><summary>Check your reasoning</summary>About 2.1 points: one photograph is 1/48 of the test split. "
                "Keep that in mind in Section 8 — a difference of one or two photographs between two models is within "
                "what one seeded split can show. In the recorded runs of the previous notebook version (Kaggle T4 and a "
                "local CPU check) the split was 108 / 24 / 48 with digests `1e4cca7f…` / `d176b3ff…` / `0e787f09…`, and 31 "
                "of 117 observers had photographs in more than one split.</details>\n\n"
                "## 5. Embed through the inference contract\n\n"
                "Before any adaptation, the embedding contract is exercised as it always was, on three test photographs. "
                "The cell first calls `pipe.reset_to_pretrained()`: on the first run it changes nothing, but after an "
                "earlier run of Sections 6–10 (or a BYOD rerun) it puts the pretrained DINOv2 weights back, so the vectors "
                "below describe DINOv2 itself. "
                "`validate_inputs` applies exactly the checks `embed` applies — type, batch size 1..`MAX_BATCH`, image side "
                "1..`MAX_IMAGE_SIDE` px — and returns an input manifest; a deliberately oversized image is validated too "
                "and its rejection recorded as a finding. `embed` returns one unit-norm 384-d vector per image, in input "
                "order, with `dim`, `pooling`, `normalized` and `input_size`. Two cosine similarities are printed — a "
                "same-species pair and a cross-species pair — as a qualitative look at the representation before any "
                "metric is read; **cosine is a similarity, not a score**, and a vector carries no label. The frozen "
                "predictions that Section 9 compares against come later, from the probe of Section 6.\n\n"
                "**Predict before running:** will the same-species cosine be higher than the other-species cosine? Would "
                "that one pair prove that the features separate the six species?"
            ),
            "code": _EMBED_CODE,
        },
        {
            "md": (
                "**What to notice:** four sanity checks `True`, one recorded rejection finding, and two cosines.\n\n"
                "<details><summary>Check your reasoning</summary>Usually higher, but one pair proves nothing: two "
                "photographs of the same species can differ in pose, light and background more than two species do. "
                "Whether the features separate the classes is measured on labelled images in Section 6, by the k-NN vote "
                "and the probe — that is why the notebook prints no verdict here.</details>\n\n"
                "## 6. The majority floor, the k-NN baseline and the frozen policy\n\n"
                "Three numbers frame the adaptation, all on the 48 test photographs. The cell starts with "
                "`pipe.reset_to_pretrained()`, so these numbers always describe the pretrained features, however often "
                "the section is re-run. The **majority floor** predicts the most frequent training species for every "
                "image — 1 / 6 here, since the sample is balanced. The **cosine 5-NN vote** labels each test image by its "
                "five nearest training images in the frozen feature space: what the representation gives with no training "
                "at all. The **frozen policy** is `pipe.adapt` with `trainable_blocks=0`: a linear head over the frozen, "
                "L2-normalised class-token features, trained full-batch with AdamW for `PROBE_STEPS` steps — the linear "
                "probe — scored on validation as epoch 0 of its history and then on the test split by `pipe.evaluate` "
                "(accuracy, macro-F1, per-class recall, confusion). The probe's predictions on six test photographs are "
                "kept for Section 9.\n\n"
                "**Predict before running:** the k-NN vote trains nothing; the probe trains 2,310 numbers (384 features × "
                "6 species + 6 biases). Which will score higher on the test split — and by a lot or a little?"
            ),
            "code": _FROZEN_CODE,
        },
        {
            "md": (
                "**What to notice:** the floor near 0.167, the k-NN vote and the probe both far above it, the probe's "
                "validation log-loss, and which species have the lowest recall.\n\n"
                "<details><summary>Check your reasoning</summary>Close to a tie. In the recorded runs of the previous notebook "
                "version (Kaggle T4 and a local CPU check, identical numbers; this version's default computation is the "
                "same) the floor scored 0.1667 accuracy / 0.048 macro-F1, the 5-NN vote 0.8333 / 0.831 and the probe "
                "0.8333 / 0.836, with a validation log-loss of 0.073 at 24 of 24 correct. When a vote with no training "
                "already does as well as a trained head, the features carry the task; the probe mostly re-weights them. "
                "The two sparrows and the junco are where both lose photographs.</details>\n\n"
                "## 7. The unfrozen policy: a bounded unfreeze selected against the probe\n\n"
                "`pipe.adapt` with `TRAINABLE_BLOCKS` > 0 first puts the pretrained weights back (so a rerun with other "
                "settings never continues from an earlier unfreeze), retrains the linear probe on the frozen features "
                "(epoch 0 of the history, the frozen policy — the cell checks it matches the Section 6 probe), then "
                "unfreezes the last `TRAINABLE_BLOCKS` transformer blocks — two by default, 3,550,464 of 22,056,192 "
                "parameters; the patch embedding, the position embedding, the earlier blocks and the final norm stay "
                "frozen — and trains them with the head end to end on the photographs for `EPOCHS` epochs (AdamW at "
                "`LEARNING_RATE`, weight decay 0.01, gradient clipping 1.0, seeded shuffling, no augmentation). Every epoch "
                "is scored on validation, and the epoch with the **lowest validation log-loss** is kept — epoch 0, the "
                "probe, competes on equal terms, so the selected policy can be either. Accuracy and macro-F1 are printed "
                "beside the loss at every epoch.\n\n"
                "**Predict before running:** the probe already gets the 24 validation photographs right. Can an unfrozen "
                "epoch still win the selection — and if so, on what?"
            ),
            "code": _UNFROZEN_CODE,
        },
        {
            "md": (
                "**What to notice:** `starts_from_pretrained_base` with an `abs_difference` of 0 (or a rounding-sized "
                "number on a GPU), the validation log-loss per epoch, `best_epoch` and the selected policy.\n\n"
                "<details><summary>Check your reasoning</summary>Yes — on confidence. With every validation photograph "
                "already right, only log-loss can move, and it rewards a higher probability on the gold species. In the "
                "recorded runs of the previous version validation log-loss went 0.073 (probe) → 0.098 → 0.092 → 0.022 → "
                "0.041 with accuracy 100 / 95.8 / 95.8 / 100 / 100 %, so epoch 3 was kept and the policy is `unfrozen last "
                "2 blocks + linear head`. A lower loss on 24 photographs is a small signal; Section 8 shows what it "
                "bought.</details>\n\n"
                "## 8. Held-out evaluation\n\n"
                "The test split was never used for training or policy selection, and no photograph in it appears in the "
                "training or validation splits. The selected model is scored exactly as the frozen policy was in Section 6, "
                "and the four rows are put side by side: majority floor, k-NN vote, frozen policy, selected policy. Read "
                "the policy first: if validation kept the probe, the last two rows are the same model; if it chose the "
                "unfreeze, the delta is what the unfreeze bought on 48 photographs. The cell asserts the selected model "
                "beats the majority floor; it does **not** assert a gain over the probe, because that is the question, not "
                "the answer. 48 photographs from one seeded split of one corpus give no dispersion estimate — one image "
                "is about two points of accuracy. The cell also adds this run to `run_history`, which Section 10 prints.\n\n"
                "**Predict before running:** validation preferred the unfreeze. Will its test accuracy be higher than the "
                "probe's, equal, or lower? And its test log-loss?"
            ),
            "code": _EVAL_CODE,
        },
        {
            "md": (
                "**What to notice:** the four-way table, `delta_vs_frozen`, the two test log-losses and the per-class "
                "recall.\n\n"
                "<details><summary>Check your reasoning</summary>Equal on accuracy, worse on log-loss. In the recorded runs "
                "of the previous version both policies scored 0.8333 test accuracy (Δ 0.0; macro-F1 0.836 → 0.840), while "
                "the test log-loss rose from 0.498 (probe) to 0.569 (selected). Validation chose the unfreeze on 24 "
                "photographs; on the 48 test photographs it was no better and less well calibrated. That is what selection "
                "noise looks like on a small validation split, and why the notebook asserts no gain.</details>\n\n"
                "## 9. Predict before and after, export the adapter and reload it\n\n"
                "Six test photographs are labelled by `pipe.classify` with the selected model and printed beside the "
                "frozen policy's predictions — the predictions the Section 6 probe made on the same six photographs over "
                "the pretrained features, kept in `frozen_before` — and the gold species with the top probability; read "
                "the probabilities as the head's softmax, not a calibrated confidence. The per-batch `evaluation_report` "
                "helper — the inference-stage helper for `embed` — is written for the three probe images and stays "
                "`not-measurable`, because a batch of vectors has no metric without labels; `pipe.evaluate` is that "
                "labelled evaluation.\n\n"
                "`pipe.save_artifact` writes the head (`head.weight`, `head.bias`) and, when the unfrozen policy was "
                "selected, the trained block tensors — about 14.2 MB — as `adapter.safetensors`, with a `manifest.json` "
                "recording the artifact format, the base model id and revision, the digest of the base `model.safetensors`, "
                "the classes, the selected policy, the tensor names, the file size and SHA-256, the training configuration "
                "and the epoch history (OUT8). `DINOv2FeatureExtractionPipeline.from_artifact` re-verifies the base "
                "snapshot, checks the artifact manifest and digest **before** deserialising, rebuilds the head from the "
                "manifest's classes, refuses any tensor that is not a transformer-block tensor of the base, and overlays "
                "the tensors onto a freshly loaded base — a new object from files, not the in-memory model (VER2). Because "
                "every `adapt` starts from the pretrained weights, the in-memory model is exactly base + saved tensors. The "
                "cell asserts identical probabilities on the six images and an identical test accuracy (VER4).\n\n"
                "**Predict before running:** how many of the six predictions will change between the frozen and the "
                "selected model?"
            ),
            "code": _EXPORT_CODE,
        },
        {
            "md": (
                "**What to notice:** the six rows (gold, frozen, selected), `predictions_changed`, 30 tensors in the "
                "artifact, `reload_parity` all true, and six entries in `outputs/`.\n\n"
                "<details><summary>Check your reasoning</summary>Few or none. In the recorded runs of the previous version "
                "one of six changed (`test-002`, white-throated sparrow → song sparrow, which is the gold species), the "
                "artifact held 30 tensors (the head and the two blocks, 14,213,808 bytes) and the reload reproduced the "
                "probabilities exactly. Loading a file is not reproducing a result; the parity check is what shows the "
                "export is the model that was evaluated.</details>\n\n"
                "## 10. Your turn — change one thing: unfreeze one block\n\n"
                "**Predict → Change one thing → Run → Observe → Explain.**\n\n"
                "1. **Predict:** with `TRAINABLE_BLOCKS = 1` (the last block only, about 1.8 M trainable parameters instead "
                "of 3.6 M), will validation still choose the unfreeze over the probe, and will the test accuracy move? "
                "Write your guess down.\n"
                "2. **Change one thing:** in Section 7 set `TRAINABLE_BLOCKS = 1` and nothing else.\n"
                "3. **Run:** select the Section 7 cell and choose **Runtime → Run after** (it re-runs Sections 7–10; the "
                "artifact and result files are overwritten with this run). `adapt` starts again from the pretrained "
                "weights, so epoch 0 equals the Section 6 probe — the cell prints that check.\n"
                "4. **Observe:** this cell prints one row per Section 7 run in this session — blocks, learning rate, "
                "epochs, the probe's validation log-loss, the selected epoch and policy, and the test accuracy beside the "
                "frozen policy's.\n"
                "5. **Explain:** did fewer unfrozen parameters change the selection or the test result, and can 48 test "
                "photographs tell the two settings apart?\n\n"
                "<details><summary>Check your reasoning</summary>Read your two rows. Both start from the same probe "
                "(identical `probe_val_log_loss`), so any difference comes from the unfreeze alone. With one block there "
                "is less to train, and the selection can fall either way on 24 validation photographs; a test accuracy "
                "that moves by one or two photographs (2.1 or 4.2 points) is within what one split can show. If validation "
                "keeps epoch 0, the policy is the frozen one, the artifact holds only the head, and the reload parity still "
                "holds. Other experiments in the same pattern: `LEARNING_RATE = 1e-4`, `EPOCHS = 8`, `TRAINABLE_BLOCKS = 4` "
                "— each is a new row.</details>"
            ),
            "code": _ACTIVITY_CODE,
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The frozen DINOv2 features already separate six bird species well enough that a cosine 5-NN vote and a linear "
        "probe both reach the low eighties on 48 held-out photographs against a majority floor of one in six, and a "
        "bounded unfreeze of the last two blocks on 108 training photographs — selected against the probe by "
        "validation log-loss — was selected on validation but changed held-out accuracy by nothing (and raised the test "
        "log-loss) in the recorded runs. That is the claim and the finding: the adaptation contract runs both policies "
        "end to end on a real labelled corpus, chooses between them on validation rather than by assumption, and reports "
        "the answer against a floor and a no-training baseline rather than in isolation.\n\n"
        "The test split is 48 photographs from one seeded split of one small corpus with no dispersion estimate — one "
        "image is about two points of accuracy, so a two-point delta is noise. Observers contribute to more than one "
        "split (the notebook counts them), so some of what the head learns may be a photographer's style rather than a "
        "bird. Accuracy and macro-F1 say whether the gold species is predicted, not whether the features are good for "
        "any other task; the head's softmax is not a calibrated confidence. The unfreeze changes the last blocks, which "
        "every input shares, so `embed` returns different vectors while the adapted model is loaded — cosines are not "
        "comparable across the frozen and adapted models, which is why Sections 5 and 6 reset to the pretrained weights "
        "first.\n\n"
        "Three things to carry to real data. **Floors first:** the majority floor and the k-NN vote on *your* images are "
        "the numbers to read before any trained head's — if the probe barely beats k-NN, the features already carry the "
        "task. **Leakage:** split by photographer, session or device when your images come from one (the contract "
        "de-duplicates by pixels, not by source). **Policy:** unfreezing is a hypothesis to test on validation, not a "
        "default; with a few hundred images the probe is usually the honest choice, and the artifact records which "
        "policy won.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone "
        "notebook, can acquire and digest-verify the pinned model snapshot, fetch and digest-verify a real labelled image "
        "corpus, validate the demonstrated dataset contract without leakage, execute the embedding contract, a linear "
        "probe and a bounded unfreeze with validation-based policy selection, evaluate by accuracy and macro-F1 against a "
        "floor and a k-NN baseline on an independent split, and emit the shown machine-readable artifacts — without the "
        "repository being reachable. It does **not** establish benchmark superiority, feature quality on any other task, a "
        "usable acceptance threshold, or production fitness.\n\n"
        "**Next experiments** (each starts after the default Run all; every one starts again from the pretrained weights):\n\n"
        "1. **Unfreeze settings:** in Section 7 change one of `TRAINABLE_BLOCKS` (1 or 4), `LEARNING_RATE` (e.g. 1e-4) or "
        "`EPOCHS` (e.g. 8), select the Section 7 cell and choose **Runtime → Run after**; Section 10 shows the new row "
        "beside the earlier ones.\n"
        "2. **Another split:** in Section 4 change `SPLIT_SEED`, then **Run after** from Section 4; the floors, the probe "
        "and the unfreeze are all recomputed on the new split.\n"
        "3. **Your own labelled images:** in Section 4 set `USE_BYOD = True` (and upload, or set `BYOD_PATH`), then "
        "**Run after** from Section 4; read the k-NN baseline before either policy.\n\n"
        "## Troubleshooting\n\n"
        "- **Section 1 stops with \"needs a Linux x86_64 runtime\".** The locked environment is built from manylinux "
        "wheels; use Colab, Kaggle or a Linux Jupyter server.\n"
        "- **Section 1 fails to download `uv`, Python or a package.** The runtime needs `pypi.org`, "
        "`files.pythonhosted.org` and the python-build-standalone release host. Re-run the cell; a size or SHA-256 "
        "mismatch is refused on purpose.\n"
        "- **\"The isolated environment's Python process exited\".** Usually out of memory. Restart the session and choose "
        "**Run all** again; on a small CPU runtime lower `BATCH_SIZE` in Section 7.\n"
        "- **Section 3 reports a size or SHA-256 mismatch.** A snapshot file was altered or truncated; delete "
        "`weights/vit-small-dinov2/model.safetensors` and run Section 3 again.\n"
        "- **Section 4 fails to download a photograph, or reports a size or SHA-256 mismatch.** "
        "`inaturalist-open-data.s3.amazonaws.com` was unreachable or served different bytes. Re-run Section 4; files "
        "already fetched are read from `weights/inat-birds/`. A persistent mismatch means the served file changed and "
        "the notebook refuses it on purpose.\n"
        "- **Section 4 refuses a BYOD dataset.** The message names the split (too few training images, or a split with "
        "one class), or the `labels.csv` line, id and file (missing, or not a decodable image), or the rule (duplicate "
        "id, id or label pattern, image side). Fix the archive and run Section 4 again.\n"
        "- **Section 7 stops with \"Epoch 0 is not the Section 6 probe\".** Sections 6 and 7 were run on different "
        "splits or settings; run Section 4 onward (**Run after**) in order.\n"
        "- **Section 9's parity assertion fails.** The export or reload is broken; run Sections 7–9 again. Do not use "
        "that artifact.\n"
        "- **Slow on CPU.** The k-NN pass embeds all 156 training and test photographs at 518 px and each unfreeze epoch "
        "trains on 108; a GPU runtime is several times faster. Lower `EPOCHS` for a quicker experiment.\n"
        "- **CUDA out of memory.** Lower `BATCH_SIZE` in Section 7, or switch the runtime to CPU.\n\n"
        "## Glossary\n\n"
        "- **DINOv2 / ViT-S/14:** a vision transformer (small size, 14 px patches) trained by self-supervision — no "
        "labels — on 142 M curated images (Oquab et al., 2023). It turns an image into 1,369 patch tokens plus a class "
        "token.\n"
        "- **Class token / embedding:** the extra token whose final state summarises the image; here it is the 384-d "
        "embedding. **L2 normalisation:** scaling a vector to length 1, so a dot product is a cosine similarity.\n"
        "- **Linear probe:** a single linear layer trained on frozen features; the cheapest honest test of a "
        "representation.\n"
        "- **k-NN vote:** label a test image by the most common label among its k most similar training images; no "
        "training.\n"
        "- **Majority floor:** the score of always answering the most frequent training class.\n"
        "- **Accuracy / macro-F1:** the share of correct labels / the unweighted mean of per-class F1, which punishes a "
        "forgotten class. **Log-loss:** the mean negative log-probability of the gold label; it rewards confident "
        "correct answers and punishes confident wrong ones.\n"
        "- **Frozen policy / unfrozen policy:** train only a head on the frozen backbone / also train the last few "
        "transformer blocks end to end.\n"
        "- **Validation-based selection:** keep the epoch (and so the policy) with the best validation score; the test "
        "split is used once, afterwards.\n"
        "- **Pretrained base / reset:** the verified DINOv2 weights; `reset_to_pretrained()` (and every `adapt`) puts "
        "them back before a new run.\n"
        "- **Adapter / reload parity:** the saved head plus any trained blocks / the check that base + adapter "
        "reproduces the in-memory model's probabilities.\n\n"
        "## Conclusion (your notes)\n\n"
        "Fill in from the numbers this run printed; keep each claim to what the evidence shows.\n\n"
        "- **Task:** which images, which classes, which split sizes?\n"
        "- **Principal result:** the selected policy and its test accuracy and macro-F1, with `n`.\n"
        "- **Baselines / reference:** the majority floor, the k-NN vote and the frozen policy on the same split, and "
        "`delta_vs_frozen`.\n"
        "- **Uncertainty or failure mode:** how many photographs is the delta? Which species are confused? What did "
        "your Section 10 run change?\n"
        "- **Limitations:** what does this run *not* show (see Interpretation and limits)?\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/dinov2-feature-extraction-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/dinov2-feature-extraction-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/dinov2-feature-extraction-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/facebookresearch/dinov2\n"
        "- DINOv2: Learning Robust Visual Features without Supervision (Oquab et al., 2023): https://arxiv.org/abs/2304.07193\n"
        "- iNaturalist open data (CC0 photographs credited to their observers in the carried records): https://www.inaturalist.org/pages/developers\n"
        "- timm documentation: https://huggingface.co/docs/timm\n"
        "- DIMER Notebook Specification 2.2 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)"
    ),
}
