"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (pipeline.py, metrics.py, samples.py), and the model pin/stage/verify cells are produced by
the generator from repository sources so they cannot drift from the package.

This template configures an E2E supervised-adaptation workflow: the pinned DINOv2 ViT-S/14 snapshot is
digest-verified and loaded, a digest-pinned real image corpus (CC0 iNaturalist bird photographs) is fetched,
validated and split, the embedding contract is exercised, the frozen features are scored by k-NN and by a linear
probe (the frozen policy) beside a majority floor, a bounded unfreeze of the last blocks is trained and selected
against the probe on validation (the unfrozen policy), the held-out split is scored, and the adapter is exported
and reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "dinov2_feature_extraction_pipeline",
    "repo_name": "dinov2-feature-extraction-pipeline",
    "stem": "dinov2_feature_extraction",
    "notebook_name": "dinov2_feature_extraction_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, stages and digest-verifies the "
        "pinned DINOv2 ViT-S/14 snapshot (safetensors, 88 MB), fetches 180 digest-pinned CC0 iNaturalist photographs of six "
        "bird species from the iNaturalist open-data bucket (19 MB, no credential), validates them and draws 108 / 24 / 48 "
        "training, validation and test images by a seeded stratified split, embeds three test images through the "
        "inference contract with an input manifest and a rejection probe, scores the frozen features on the test split by "
        "a majority floor, a cosine 5-NN vote and a linear probe (the **frozen policy**), trains a bounded unfreeze of the "
        "last two transformer blocks with the head and selects between it and the probe by validation log-loss (the "
        "**unfrozen policy**), scores the held-out split with the selected model, prints predictions before and after, "
        "exports the head and any trained blocks as safetensors with a manifest, and reloads that artifact into a fresh "
        "pipeline to verify parity. The default path needs no repository clone, no DIMER worker or service, no credential, "
        "no upload dialog and no configuration edit (NOTEBOOK_SPEC 2.0 §5). On CPU the whole path takes about four "
        "minutes of model time after the downloads; a CUDA runtime is used automatically when present."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to supply your own "
        "labelled images as a `.zip` holding `labels.csv` (columns `id`, `file`, `label`) beside the image files — images are "
        "decoded from the archive, never extracted to disk. They pass through the same validation, seeded stratified "
        "image-disjoint split, floors, frozen-policy probe, unfrozen-policy training and selection, held-out evaluation, "
        "prediction, artifact export and reload-parity cells as the iNaturalist sample. The expected schema and the ceilings "
        "are stated in the Prerequisites and in Section 4, and uploaded files stay inside this runtime. BYOD is optional and "
        "never part of the default path."
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
        "kept. The adaptation question is whether unfreezing buys anything over the probe on 108 training photographs. "
        "Nothing here is a quality claim about your images: it is one seeded split of one small corpus."
    ),
    "learning_objectives": (
        "install the pinned runtime; read what the carried pipeline, metrics and dataset modules guarantee; stage and "
        "digest-verify the immutable upstream snapshot; fetch a digest-pinned real image corpus and validate and split it "
        "without leakage; embed images through the public API and read the vector contract correctly; read accuracy and "
        "macro-F1 beside a majority floor and a k-NN baseline; train a linear probe on frozen features and a bounded "
        "unfreeze with explicit hyperparameters and validation-based selection between the two policies; evaluate on an "
        "independent test split; compare predictions before and after; and export a safetensors adapter (head plus any "
        "trained blocks) that reloads against the pinned base with verified parity."
    ),
    "exclusions": (
        "object detection, segmentation, captioning, image-text comparison, patch-level (dense) features, attention maps, "
        "data augmentation, full-backbone or patch-embedding training, any similarity search or clustering quality claim, "
        "and any claim that six bird species from one photo site stand in for your images. The repository exposes none of "
        "these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU and uses CUDA automatically when available; float32 on both. Each 518 px image costs about 46.8 GMACs (upstream card): the build record measured about 0.12 s per image to embed on CPU (21 s for the 180-image k-NN pass), a 16 s linear probe including feature extraction, and about 19 s per unfreeze epoch over 108 images plus a 24-image validation pass. The pinned `torch==2.14.0` install and the 88 MB checkpoint are the large downloads of the run, then the 19 MB of photographs.",
        "- **Knowledge:** basic Python and NumPy; what an embedding vector is; what a linear probe is and why it is the cheapest honest test of a representation; what accuracy and macro-F1 measure and why macro-F1 punishes a forgotten class; what validation-based selection between two policies means.",
        "- **Data contract:** records are `{{id, image, label}}` — a PIL image (or a path to one) with each side 1..4,096 px (the pipeline's own ceiling; images are resized and centre-cropped to 518 px, never rejected for being small), a label of 1..64 plain characters, ids matching `[A-Za-z0-9_.:-]{{1,64}}` and unique; a dataset needs 8..20,000 records and 2..100 classes; images are de-duplicated by decoded-pixel digest before splitting so the same photograph never sits in two splits. BYOD accepts a `.zip` (or a directory) holding `labels.csv` and the image files.",
        "- **Validation is structural, not semantic:** nothing checks that a label is right for its image — a mislabelled set is trained on without complaint; observers can contribute to more than one split (the notebook counts them) because the sample is stratified by species, not by observer.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — photographs of people or private places are exactly that. The default path uploads nothing.",
        "- **External access (data):** besides the Hub, the default path fetches 180 pinned objects (`<photo id>/medium.jpg` or `medium.jpeg`, each photo's own served extension recorded in the table, 19,183,071 bytes in total, one SHA-256 each in the carried `SAMPLE_RECORDS` table) from `inaturalist-open-data.s3.amazonaws.com` over HTTPS, each refused on any byte-size or SHA-256 mismatch before it is decoded; every photograph is CC0 by its own iNaturalist licence code and credited to its observer in the records.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Referenced corpus, validation and split\n\n"
                "`fetch_corpus` downloads the 180 pinned photographs (or reads them from the cache), refuses a byte-size "
                "or SHA-256 mismatch per file before it is decoded, and `read_corpus` turns each into a `{{id, image, "
                "label}}` record with its species names, observation URL and observer. `build_sample_dataset` draws 18 "
                "training, 4 validation and 8 test photographs per species by a seeded stratified shuffle; "
                "`validate_dataset` then checks every record against the contract, `check_split_disjoint` asserts no "
                "photograph (by decoded-pixel digest) appears in two splits, `observer_overlap` counts the observers who "
                "contributed to more than one split — an observation, since the sample is stratified by species — and "
                "the training labels table is written to `outputs/{stem}_train.csv` in the shape BYOD expects.\n\n"
                "Look for: 180 photographs, six classes of 30, splits 108 / 24 / 48, three digests, and four refusal "
                "probes — a duplicate id, an oversized image, a single-class dataset and a dataset too small to split — "
                "each rejected before `torch` does anything. About a minute on the first run for the downloads."
            ),
            "code": (
                "import hashlib\n"
                "import io\n"
                "import json\n"
                "import time\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "t0 = time.perf_counter()\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    file_name, payload = next(iter(uploaded.items()))\n"
                "    byod_path = Path('work') / file_name\n"
                "    byod_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "    byod_path.write_bytes(payload)\n"
                "    records = load_byod_dataset(byod_path)\n"
                "    splits = split_dataset(records, seed=SPLIT_SEED)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "    raw_count = {{'byod': len(records)}}\n"
                "else:\n"
                "    corpus = read_corpus(fetch_corpus(cache_dir='weights/inat-birds'))\n"
                "    raw_count = {{'photographs': len(corpus), 'species': len(SPECIES)}}\n"
                "    splits = build_sample_dataset(corpus, seed=SPLIT_SEED)\n"
                "    data_source = f'{{CORPUS_NAME}} ({{CORPUS_RELEASE}}; {{CORPUS_LICENSE}})'\n"
                "fetch_seconds = round(time.perf_counter() - t0, 1)\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "dataset_manifests = {{name: validate_dataset(part) for name, part in splits.items()}}\n"
                "classes = class_names(train_records)\n"
                "disjoint = check_split_disjoint(splits)\n"
                "overlap = observer_overlap(splits)\n"
                "write_dataset_csv(train_records, 'outputs/{stem}_train.csv')\n"
                "print({{'data_source': data_source, 'raw': raw_count, 'splits': disjoint, 'classes': classes, 'observer_overlap': overlap, 'fetch_seconds': fetch_seconds, 'corpus_bytes': CORPUS_BYTES}})\n"
                "for name, manifest in dataset_manifests.items():\n"
                "    print({{name: {{'n': manifest['n_records'], 'label_counts': manifest['label_counts'], 'image_side': manifest['image_side'], 'digest': manifest['digest'][:16] + '...'}}}})\n"
                "example = train_records[0]\n"
                "print({{'example': {{k: example[k] for k in ('id', 'label', 'observer', 'inat_observation_url') if k in example}}, 'size': example['image'].size}})\n\n"
                "probes = {{\n"
                "    'duplicate id': [{{**r, 'id': 'same'}} for r in train_records[:8]],\n"
                "    'oversized image': [{{**train_records[0], 'image': Image.new('RGB', (MAX_IMAGE_SIDE + 1, 8))}}, *train_records[1:8]],\n"
                "    'single class': [{{**r, 'label': 'bird'}} for r in train_records[:8]],\n"
                "    'too small': train_records[:3],\n"
                "}}\n"
                "for name, probe in probes.items():\n"
                "    try:\n"
                "        validate_dataset(probe)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                "## 5. Embed through the inference contract\n\n"
                "Before any adaptation, the embedding contract is exercised as it always was, on three test photographs. "
                "`validate_inputs` applies exactly the checks `embed` applies — type, batch size 1..`MAX_BATCH`, image side "
                "1..`MAX_IMAGE_SIDE` px — and returns an input manifest; a deliberately oversized image is validated too "
                "and its rejection recorded as a finding. `embed` returns one unit-norm 384-d vector per image, in input "
                "order, with `dim`, `pooling`, `normalized` and `input_size`. Two cosine similarities are printed — a "
                "same-species pair and a cross-species pair — as a qualitative look at the representation before any "
                "metric is read; **cosine is a similarity, not a score**, and a vector carries no label. The frozen "
                "predictions that Section 9 compares against come later, from the probe."
            ),
            "code": (
                "probe_records = test_records[:3]\n"
                "print({{'ceilings': {{'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MAX_BATCH': MAX_BATCH}}, 'contract': {{'EMBED_DIM': EMBED_DIM, 'POOLING': POOLING, 'NORMALIZED': NORMALIZED, 'TRANSFORMER_BLOCKS': TRANSFORMER_BLOCKS, 'PARAMETER_COUNT': PARAMETER_COUNT}}}})\n"
                "input_manifest = validate_inputs([r['image'] for r in probe_records], names=[r['id'] for r in probe_records])\n"
                "try:\n"
                "    validate_inputs(Image.new('RGB', (MAX_IMAGE_SIDE + 1, 8)))\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'oversized-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "started = time.perf_counter()\n"
                "embedding_result = pipe.embed([r['image'] for r in probe_records])\n"
                "embed_seconds = round(time.perf_counter() - started, 3)\n"
                "vectors = np.asarray(embedding_result['embeddings'], dtype=np.float32)\n"
                "checks = {{\n"
                "    'one_vector_per_image': vectors.shape == (3, EMBED_DIM) and embedding_result['dim'] == EMBED_DIM,\n"
                "    'unit_norm': bool(embedding_result['normalized']) and bool(np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)),\n"
                "    'contract_fields': embedding_result['pooling'] == POOLING and embedding_result['input_size'] == [518, 518],\n"
                "    'all_values_finite': bool(np.isfinite(vectors).all()),\n"
                "}}\n"
                "if not all(checks.values()):\n"
                "    raise RuntimeError(f'embed output failed a sanity check: {{checks}}')\n"
                "same = next((r for r in test_records[3:] if r['label'] == probe_records[0]['label']), test_records[3])\n"
                "other = next((r for r in test_records[3:] if r['label'] != probe_records[0]['label']), test_records[4])\n"
                "pair_vectors = np.asarray(pipe.embed([same['image'], other['image']])['embeddings'], dtype=np.float32)\n"
                "print({{'probe_ids': [r['id'] for r in probe_records], 'probe_labels': [r['label'] for r in probe_records], 'seconds': embed_seconds, 'device': pipe.device, 'checks': checks, 'findings': len(input_manifest['findings'])}})\n"
                "print({{'cosine_same_species': round(float(vectors[0] @ pair_vectors[0]), 4), 'cosine_other_species': round(float(vectors[0] @ pair_vectors[1]), 4), 'note': 'one pair each; a similarity, not a score'}})"
            ),
        },
        {
            "md": (
                "## 6. The majority floor, the k-NN baseline and the frozen policy\n\n"
                "Three numbers frame the adaptation, all on the 48 test photographs. The **majority floor** predicts the "
                "most frequent training species for every image — 1 / 6 here, since the sample is balanced. The **cosine "
                "5-NN vote** labels each test image by its five nearest training images in the frozen feature space: what "
                "the representation gives with no training at all. The **frozen policy** is `pipe.adapt` with "
                "`trainable_blocks=0`: a linear head over the frozen, L2-normalised class-token features, trained "
                "full-batch with AdamW for `PROBE_STEPS` steps — the linear probe — scored on validation as epoch 0 of its "
                "history and then on the test split by `pipe.evaluate` (accuracy, macro-F1, per-class recall, "
                "confusion). The build record saw the k-NN vote and the probe both in the low eighties; read the "
                "per-class recall to see which species the features confuse. About forty seconds on CPU."
            ),
            "code": (
                "PROBE_STEPS = 300  # @param {{type:\"integer\"}}\n"
                "PROBE_LR = 0.01  # @param {{type:\"number\"}}\n\n"
                "def brief(m):\n"
                "    return {{'accuracy': round(m['accuracy'], 4), 'macro_f1': round(m['macro_f1'], 4), 'n': m['n']}}\n\n"
                "floor = majority_baseline([r['label'] for r in train_records], [r['label'] for r in test_records], classes)\n"
                "print({{'majority_floor': brief(floor), 'baseline': floor['baseline']}})\n"
                "t0 = time.perf_counter()\n"
                "baseline_knn = pipe.knn_baseline(train_records, test_records, k=5)\n"
                "print({{'knn_baseline': brief(baseline_knn), 'baseline': baseline_knn['baseline'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "t0 = time.perf_counter()\n"
                "probe_result = pipe.adapt(train_records, val_records, probe_steps=PROBE_STEPS, probe_lr=PROBE_LR, trainable_blocks=0)\n"
                "frozen_test = pipe.evaluate(test_records)\n"
                "print({{'frozen_policy': probe_result['policy'], 'probe_final_loss': round(probe_result['probe_final_loss'], 4), 'validation': {{k: round(v, 4) for k, v in probe_result['history'][0]['val'].items()}}, 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "print({{'frozen_policy_test': brief(frozen_test), 'log_loss': round(frozen_test['log_loss'], 4), 'verdict': frozen_test['verdict'], 'per_class_recall': {{c: round(v['recall'], 2) for c, v in frozen_test['per_class'].items()}}}})\n"
                "print({{'definitions': frozen_test['definitions']}})\n"
                "assert frozen_test['accuracy'] > floor['accuracy'] and probe_result['policy'].startswith('frozen')"
            ),
        },
        {
            "md": (
                "## 7. The unfrozen policy: a bounded unfreeze selected against the probe\n\n"
                "`pipe.adapt` with `TRAINABLE_BLOCKS` > 0 first retrains the linear probe on the frozen features (epoch 0 "
                "of the history, the frozen policy), then unfreezes the last `TRAINABLE_BLOCKS` transformer blocks — "
                "two by default, 3,550,464 of 22,056,192 parameters; the patch embedding, the position embedding, "
                "the earlier blocks and the final norm stay frozen — and trains them with the head end to end on the "
                "photographs for `EPOCHS` epochs (AdamW at `LEARNING_RATE`, weight decay 0.01, gradient clipping 1.0, "
                "seeded shuffling, no augmentation). Every epoch is scored on validation, and the epoch with the "
                "**lowest validation log-loss** is kept — epoch 0, the probe, competes on equal terms, so the selected "
                "policy can be either. Accuracy and macro-F1 are printed beside the loss at every epoch.\n\n"
                "Watch the validation loss: with 24 validation photographs the probe is already near-perfect, so the "
                "unfreeze must beat it on confidence, not just on the label. The build record's sweep on this sample: two blocks at 3e-5 for four epochs lowered validation log-loss below the probe's at epoch 3 and was selected, for the same held-out accuracy as the probe; two blocks at 1e-4 was also selected (epoch 3) and lost eight points on the held-out split; one block at 1e-4 was not selected."
            ),
            "code": (
                "EPOCHS = 4  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 3e-5  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 8  # @param {{type:\"integer\"}}\n"
                "TRAINABLE_BLOCKS = 2  # @param {{type:\"integer\"}}\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'stage': entry['stage'], 'train_loss': round(entry['train_loss'], 4)}}\n"
                "    if entry.get('val'):\n"
                "        row['val_log_loss'] = round(entry['val']['log_loss'], 4)\n"
                "        row['val_accuracy'] = round(entry['val']['accuracy'], 4)\n"
                "        row['val_macro_f1'] = round(entry['val']['macro_f1'], 4)\n"
                "    print(row)\n\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, probe_steps=PROBE_STEPS, probe_lr=PROBE_LR, trainable_blocks=TRAINABLE_BLOCKS, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'selected_policy': adapt_result['policy'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'trainable_head': adapt_result['n_trainable_head'], 'trainable_blocks': adapt_result['n_trainable_blocks'], 'total_parameters': adapt_result['n_total'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation\n\n"
                "The test split was never used for training or policy selection, and no photograph in it appears in the "
                "training or validation splits. The selected model is scored exactly as the frozen policy was in Section 6, "
                "and the four rows are put side by side: majority floor, k-NN vote, frozen policy, selected policy. Read "
                "the policy first: if validation kept the probe, the last two rows are the same model; if it chose the "
                "unfreeze, the delta is what the unfreeze bought on 48 photographs — the build record saw it buy nothing "
                "on accuracy at the default settings and lose accuracy at a higher learning rate. The cell asserts the "
                "selected model beats the majority floor; it does **not** assert a gain over the probe, because that is "
                "the question, not the answer. 48 photographs from one seeded split of one corpus give no dispersion "
                "estimate — one image is about two points of accuracy."
            ),
            "code": (
                "adapted_test = pipe.evaluate(test_records)\n"
                "adapted_val = pipe.evaluate(val_records)\n"
                "comparison = {{\n"
                "    metric: {{'majority_floor': round(floor[metric], 4), 'knn5': round(baseline_knn[metric], 4), 'frozen_policy': round(frozen_test[metric], 4), 'selected_policy': round(adapted_test[metric], 4)}}\n"
                "    for metric in ('accuracy', 'macro_f1')\n"
                "}}\n"
                "comparison['log_loss'] = {{'frozen_policy': round(frozen_test['log_loss'], 4), 'selected_policy': round(adapted_test['log_loss'], 4)}}\n"
                "comparison['per_class_recall'] = {{c: {{'knn5': round(baseline_knn['per_class'][c]['recall'], 2), 'frozen': round(frozen_test['per_class'][c]['recall'], 2), 'selected': round(adapted_test['per_class'][c]['recall'], 2)}} for c in classes}}\n"
                "comparison['delta_vs_frozen'] = {{metric: round(adapted_test[metric] - frozen_test[metric], 4) for metric in ('accuracy', 'macro_f1')}}\n"
                "comparison['selected_policy'] = adapt_result['policy']\n"
                "for metric, row in comparison.items():\n"
                "    print({{metric: row}})\n"
                "print({{'confusion_selected': adapted_test['confusion'], 'classes': classes}})\n"
                "evaluation_report_payload = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'dataset_digests': {{name: manifest['digest'] for name, manifest in dataset_manifests.items()}},\n"
                "    'splits': disjoint,\n"
                "    'observer_overlap': overlap,\n"
                "    'classes': classes,\n"
                "    'baselines': {{'majority_floor': floor, 'knn5': baseline_knn}},\n"
                "    'frozen_policy': {{'adaptation': {{k: v for k, v in probe_result.items() if k not in ('history', 'trainable_names')}}, 'history': probe_result['history'], 'test': frozen_test}},\n"
                "    'validation_metrics': adapted_val,\n"
                "    'test_metrics': adapted_test,\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False)\n"
                "assert adapted_test['accuracy'] > floor['accuracy']\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
        },
        {
            "md": (
                "## 9. Predict before and after, export the adapter and reload it\n\n"
                "Six test photographs are labelled by `pipe.classify` with the selected model and printed beside the "
                "frozen policy's predictions (recomputed from the probe's stored head on the frozen features — the head "
                "and blocks of the frozen policy were saved before the unfreeze) and the gold species with the top "
                "probability; read the probabilities as the head's softmax, not a calibrated confidence. The per-batch "
                "`evaluation_report` helper — the inference-stage helper for `embed` — is written for the three probe "
                "images and stays `not-measurable`, because a batch of vectors has no metric without labels; "
                "`pipe.evaluate` is that labelled evaluation.\n\n"
                "`pipe.save_artifact` writes the head (`head.weight`, `head.bias`) and, when the unfrozen policy was "
                "selected, the trained block tensors — about 14.2 MB — as `adapter.safetensors`, with a `manifest.json` "
                "recording the artifact format, the base model id and revision, the digest of the base `model.safetensors`, "
                "the classes, the selected policy, the tensor names, the file size and SHA-256, the training configuration "
                "and the epoch history (OUT8). `DINOv2FeatureExtractionPipeline.from_artifact` re-verifies the base "
                "snapshot, checks the artifact manifest and digest **before** deserialising, rebuilds the head from the "
                "manifest's classes, refuses any tensor that is not a transformer-block tensor of the base, and overlays "
                "the tensors onto a freshly loaded base — a new object from files, not the in-memory model (VER2). The "
                "cell asserts identical probabilities on the six images and an identical test accuracy (VER4)."
            ),
            "code": (
                "import csv\n"
                "import shutil\n\n"
                "show = test_records[:6]\n"
                "after = pipe.classify([r['image'] for r in show])\n"
                "frozen_pipe = DINOv2FeatureExtractionPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "frozen_pipe.adapt(train_records, val_records, probe_steps=PROBE_STEPS, probe_lr=PROBE_LR, trainable_blocks=0)\n"
                "before = frozen_pipe.classify([r['image'] for r in show])\n"
                "rows = []\n"
                "for i, record in enumerate(show):\n"
                "    rows.append({{'id': record['id'], 'gold': record['label'], 'frozen_label': before['labels'][i], 'frozen_top_probability': round(max(before['probabilities'][i]), 4), 'selected_label': after['labels'][i], 'selected_top_probability': round(max(after['probabilities'][i]), 4), 'observer': record.get('observer', ''), 'inat_observation_url': record.get('inat_observation_url', '')}})\n"
                "    print({{k: rows[-1][k] for k in ('id', 'gold', 'frozen_label', 'frozen_top_probability', 'selected_label', 'selected_top_probability')}})\n"
                "single_report = evaluation_report(embedding_result, sample_kind='three iNaturalist test photographs' if not USE_BYOD else 'three BYOD test images')\n"
                "print({{'batch_report_verdict': single_report['verdict'], 'selected_policy': after['policy'], 'predictions_changed': sum(r['frozen_label'] != r['selected_label'] for r in rows), 'of': len(rows)}})\n"
                "with open('outputs/{stem}_predictions.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))\n"
                "    writer.writeheader()\n"
                "    writer.writerows(rows)\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'policy': artifact_manifest['adapter']['policy'], 'classes': artifact_manifest['adapter']['classes'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...'}})\n\n"
                "reloaded = DINOv2FeatureExtractionPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "reloaded_after = reloaded.classify([r['image'] for r in show])\n"
                "reloaded_test = reloaded.evaluate(test_records)\n"
                "parity = {{'probabilities_identical': reloaded_after['probabilities'] == after['probabilities'], 'accuracy_in_memory': round(adapted_test['accuracy'], 6), 'accuracy_reloaded': round(reloaded_test['accuracy'], 6), 'classes_identical': reloaded.classes == pipe.classes}}\n"
                "print({{'reload_parity': parity, 'reloaded_policy': reloaded.adapter['policy'], 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['probabilities_identical'] and parity['classes_identical'] and abs(adapted_test['accuracy'] - reloaded_test['accuracy']) < 1e-9\n\n"
                "weight_entry = next(entry for entry in snapshot['files'] if entry['path'] == WEIGHTS_FILE)\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': str(WEIGHTS_DIR), 'files': len(snapshot['files']), 'total_bytes': snapshot.get('totalBytes'), 'fetched_this_run': fetched, 'weight_file': WEIGHTS_FILE, 'weight_format': 'safetensors, digest-verified', 'weight_sha256': weight_entry['sha256']}},\n"
                "    'data_source': data_source,\n"
                "    'corpus': {{'name': CORPUS_NAME, 'release': CORPUS_RELEASE, 'base_url': CORPUS_BASE_URL, 'photographs': len(SAMPLE_RECORDS), 'bytes': CORPUS_BYTES, 'license': CORPUS_LICENSE, 'species': SPECIES}},\n"
                "    'inference_contract': {{'input_manifest': input_manifest, 'sanity_checks': checks, 'probe_ids': [r['id'] for r in probe_records], 'seconds': embed_seconds}},\n"
                "    'comparison': comparison,\n"
                "    'predictions_before_after': rows,\n"
                "    'batch_report': single_report,\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes'], 'tensors': len(artifact_manifest['tensors']), 'policy': artifact_manifest['adapter']['policy']}},\n"
                "    'reload_parity': parity,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'timm': timm.__version__, 'device': pipe.device, 'dtype': 'float32', 'source': pipe.source}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(result_payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The frozen DINOv2 features already separate six bird species well enough that a cosine 5-NN vote and a linear "
        "probe both reach the low eighties on 48 held-out photographs against a majority floor of one in six, and a "
        "bounded unfreeze of the last two blocks on 108 training photographs — selected against the probe by "
        "validation log-loss — was selected on validation but changed held-out accuracy by nothing. That is the claim and the finding: the adaptation contract runs both policies "
        "end to end on a real labelled corpus, chooses between them on validation rather than by assumption, and reports "
        "the answer against a floor and a no-training baseline rather than in isolation.\n\n"
        "The test split is 48 photographs from one seeded split of one small corpus with no dispersion estimate — one "
        "image is about two points of accuracy, so a two-point delta is noise. Observers contribute to more than one "
        "split (the notebook counts them), so some of what the head learns may be a photographer's style rather than a "
        "bird. Accuracy and macro-F1 say whether the gold species is predicted, not whether the features are good for "
        "any other task; the head's softmax is not a calibrated confidence. The unfreeze changes the last blocks, which "
        "every input shares, so `embed` returns different vectors after it — cosines are not comparable across the frozen "
        "and adapted models.\n\n"
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
        "**Optional experiments (they do not affect the default path):** set `TRAINABLE_BLOCKS = 1` or `4` and watch the "
        "selection; set `LEARNING_RATE = 1e-4` and read what a hotter unfreeze does to the held-out accuracy (the build "
        "record saw it lose eight points); set `EPOCHS = 8` and watch whether validation log-loss keeps falling or turns; "
        "or bring your own labelled images through BYOD and read the k-NN baseline before either policy.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/dinov2-feature-extraction-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/dinov2-feature-extraction-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/dinov2-feature-extraction-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/facebookresearch/dinov2\n"
        "- DINOv2: Learning Robust Visual Features without Supervision (Oquab et al., 2023): https://arxiv.org/abs/2304.07193\n"
        "- iNaturalist open data (CC0 photographs credited to their observers in the carried records): https://www.inaturalist.org/pages/developers\n"
        "- timm documentation: https://huggingface.co/docs/timm\n"
        "- DIMER Notebook Specification 2.0 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)"
    ),
}
