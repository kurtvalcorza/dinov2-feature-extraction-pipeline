"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 1.1 §3.6 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
module, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "dinov2_feature_extraction_pipeline",
    "repo_name": "dinov2-feature-extraction-pipeline",
    "stem": "dinov2_feature_extraction",
    "notebook_name": "dinov2_feature_extraction_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "pipeline_class": "DINOv2FeatureExtractionPipeline",
    "weights_key": "vit-small-dinov2",
    "runtime_imports": ["torch", "timm"],
    "title": "DINOv2 ViT-S/14 — DIMER image feature extraction tutorial (standalone)",
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
    "capability": "self-supervised image feature extraction (one 384-d embedding per image) using the pinned DINOv2 ViT-S/14 `lvd142m` weights",
    "intro": (
        "At inference each image is resized so its shorter side is 518 px, centre-cropped to 518 x 518, normalised with "
        "the ImageNet mean/std from the snapshot config, and passed through the ViT-S/14 backbone with no classifier "
        "head; the class token after the final LayerNorm is taken as the image's feature (`POOLING = \"cls\"`) and "
        "L2-normalised (`NORMALIZED = True`), so a dot product between two vectors is their cosine similarity. "
        "**Embeddings are representations, not predictions:** the pipeline returns no label, no score, no class and no "
        "threshold, and no intrinsic accuracy exists for a vector on its own. **No adaptation occurs:** no training, "
        "fine-tuning, in-context conditioning, or preprocessing fitting happens in this notebook — the upstream "
        "checkpoint supplies the architecture, weights and preprocessing configuration, and the carried pipeline module "
        "adds snapshot verification, input validation, the class-token pooling choice, L2 normalisation, a fixed output "
        "contract and the `validate_inputs` and `evaluation_report` helpers."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried pipeline module guarantees, resolve and digest-verify the "
        "immutable upstream model revision, generate a small synthetic input set and validate it into an input manifest, "
        "embed a batch through the public API, read the embedding output (shape, unit, pooling, normalisation) "
        "correctly, run a qualitative cosine-similarity check and understand why it is not a metric, produce an "
        "evaluation report that is honestly `not-measurable` and says what downstream task would make the features "
        "measurable, and export the vectors with their identifiers plus provenance."
    ),
    "exclusions": (
        "image classification, object detection, segmentation, captioning, image-text comparison, patch-level (dense) "
        "features, attention maps, or any similarity search, clustering or probe — the carried module exposes only the "
        "pooled per-image vector; everything downstream is the caller's code."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU and uses CUDA automatically when available; inference is float32 on both. Each 518 px image costs about 46.8 GMACs (upstream card), so a three-image CPU batch takes seconds to tens of seconds on a hosted CPU runtime; on the model card's GPU (RTX 5070 Ti) the verified snapshot loaded in 3.4 s and a two-image batch took 1.2 s. The pinned `torch==2.14.0` install and the 88 MB checkpoint are the largest downloads of the run.",
        "- **Knowledge:** basic Python and NumPy; what an embedding vector is and why cosine similarity between two vectors is not an accuracy.",
        "- **Data:** the default sample is a set of three synthetic images generated in code, so nothing is downloaded and no private data is needed. Optional BYOD upload is gated off by default so the sample path can run top-to-bottom without interaction. Expected BYOD input: one or more image files decodable by Pillow (PNG/JPEG/WebP and similar), any colour mode, each side at most 4096 px, at most `MAX_BATCH` files per run. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Generate the synthetic sample set or optional BYOD\n\n"
                "The default sample is **synthetic**: three 256 x 256 RGB images built in code — a deterministic gradient "
                "(red ramps left to right, green top to bottom, blue is their mean), the same gradient rotated by 180 "
                "degrees, and a flat mid-grey block. They need no download, contain no personal data, and each one's pixel "
                "SHA-256 is printed and exported alongside its identifier; no randomness is involved, so no seed is needed. "
                "None of them is a photograph, so the embeddings they produce are sanity evidence that the code path works: "
                "the rotated copy is there only so that Section 7 can show a *qualitative* similarity comparison (a "
                "near-duplicate versus an unrelated image), which is not a benchmark and not a metric. BYOD is optional and "
                "disabled by default; when enabled, upload one or more image files and they replace the synthetic set."
            ),
            "code": (
                "import hashlib\n"
                "import io\n\n"
                "import numpy as np\n"
                "from PIL import Image\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "SAMPLE_SIDE = 256\n\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    images = {{}}\n"
                "    for name, data in uploaded.items():\n"
                "        image = Image.open(io.BytesIO(data))\n"
                "        image.load()\n"
                "        images[name] = image\n"
                "    sample_kind = 'BYOD upload'\n"
                "else:\n"
                "    # Deterministic synthetic set: no randomness, so no seed is needed and the digests are stable.\n"
                "    ramp = np.linspace(0.0, 255.0, SAMPLE_SIDE)\n"
                "    red = np.tile(ramp, (SAMPLE_SIDE, 1))\n"
                "    green = red.T\n"
                "    blue = (red + green) / 2.0\n"
                "    gradient = Image.fromarray(np.rint(np.stack([red, green, blue], axis=-1)).astype(np.uint8), mode='RGB')\n"
                "    images = {{\n"
                "        f'synthetic_gradient_{{SAMPLE_SIDE}}': gradient,\n"
                "        f'synthetic_gradient_{{SAMPLE_SIDE}}_rot180': gradient.rotate(180),\n"
                "        f'synthetic_flat_grey_{{SAMPLE_SIDE}}': Image.new('RGB', (SAMPLE_SIDE, SAMPLE_SIDE), (128, 128, 128)),\n"
                "    }}\n"
                "    sample_kind = 'synthetic'\n"
                "image_ids = list(images)\n"
                "digests = {{name: hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest() for name, image in images.items()}}\n"
                "for name in image_ids:\n"
                "    print({{'id': name, 'mode': images[name].mode, 'size': images[name].size, 'pixel_sha256': digests[name]}})\n"
                "print({{'sample_kind': sample_kind, 'count': len(image_ids)}})"
            ),
        },
        {
            "md": (
                "## 5. Validate the inputs → input manifest\n\n"
                "`validate_inputs` is the pipeline's public validation stage: it applies exactly the checks `embed` applies "
                "— type, batch size 1..`MAX_BATCH`, image side 1..`MAX_IMAGE_SIDE` px — and returns an **input manifest** "
                "naming the schema and ceilings, each input's identifier, observed mode and size, and the verdict. The "
                "manifest is written to `outputs/{stem}_input_manifest.json`. The cell first prints the ceilings and the "
                "embedding contract — `EMBED_DIM` (vector length, 384), `POOLING` (`\"cls\"`: the class token after the "
                "final LayerNorm — one vector **per image**, not per patch or per token) and `NORMALIZED` (`True`: every "
                "vector has unit L2 norm) — so the values shown are the ones in force before any model work. To show what "
                "rejection looks like, it also validates a deliberately oversized image and records the pipeline's own "
                "error message as a finding. **What the pipeline changes about your images:** each is converted to RGB, "
                "resized so its shorter side is 518 px and centre-cropped to 518 x 518 (`crop_pct: 1.0`, "
                "`crop_mode: \"center\"` in the snapshot config) — for a non-square image the outer strips along the longer "
                "side never reach the encoder; nothing else is dropped, and there is no missing-data concept: every pixel "
                "of the cropped square is visible to the encoder. The notebook itself does not resize, crop, or subsample."
            ),
            "code": (
                "import json\n"
                "import os\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "print({{'ceilings': {{'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MAX_BATCH': MAX_BATCH}}, 'contract': {{'EMBED_DIM': EMBED_DIM, 'POOLING': POOLING, 'NORMALIZED': NORMALIZED}}}})\n"
                "input_manifest = validate_inputs([images[name] for name in image_ids], names=image_ids)\n"
                "# Demonstrate rejection on an input that breaks a ceiling; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs(Image.new('RGB', (MAX_IMAGE_SIDE + 1, 8)))\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'oversized-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest, indent=2))"
            ),
        },
        {
            "md": (
                "## 6. Embed the batch\n\n"
                "`embed` returns `embeddings` (a list with one 384-float list per input, **in input order**, so position "
                "*i* belongs to `image_ids[i]`), plus `dim`, `pooling`, `normalized`, `input_size`, `device`, `source`, "
                "`model_id` and `model_revision`. The checks below are falsifiable plumbing checks (one vector per input, "
                "length `EMBED_DIM`, finite, unit norm within float tolerance) and the cell raises if any fails; they are "
                "not a quality measure. Small numeric differences between CPU and CUDA kernels move each vector slightly, "
                "so values are repeatable on fixed hardware but not bitwise-identical across devices. The timing is "
                "measured for this batch on the runtime identified in Section 1 and includes the first-call warm-up."
            ),
            "code": (
                "import time\n\n"
                "started = time.perf_counter()\n"
                "result = pipe.embed([images[name] for name in image_ids])\n"
                "elapsed = time.perf_counter() - started\n"
                "vectors = np.asarray(result['embeddings'], dtype=np.float32)\n"
                "norms = np.linalg.norm(vectors, axis=1)\n"
                "checks = {{\n"
                "    'one_vector_per_input': vectors.shape[0] == len(image_ids),\n"
                "    'dim_matches_contract': vectors.shape[1] == result['dim'] == EMBED_DIM,\n"
                "    'all_finite': bool(np.isfinite(vectors).all()),\n"
                "    'unit_norm': bool(np.allclose(norms, 1.0, atol=1e-4)),\n"
                "}}\n"
                "if not all(checks.values()):\n"
                "    raise RuntimeError(f'embedding output failed a sanity check: {{checks}}')\n"
                "print({{key: value for key, value in result.items() if key != 'embeddings'}})\n"
                "print({{'embeddings_shape': list(vectors.shape), 'norms': [round(float(n), 6) for n in norms], 'seconds': round(elapsed, 3), 'checks': checks}})"
            ),
        },
        {
            "md": (
                "## 7. Evaluate → evaluation report\n\n"
                "The repository ships **no metric helper and reports no performance measure**, because an embedding is a "
                "representation, not a prediction — there is nothing to score it against on its own. `evaluation_report` "
                "still produces a report, and its verdict is always `not-measurable`: it names the score semantics (a "
                "cosine similarity is not a probability, an accuracy, or a calibrated score) and states what would make "
                "the features measurable. Evaluating these features requires a downstream labelled task: a retrieval set "
                "with relevance labels (mean average precision), a labelled image set for a linear probe or k-NN "
                "classifier (accuracy), or human-judged duplicate pairs to calibrate a similarity threshold; the DINOv2 "
                "paper evaluates exactly such tasks, and those upstream numbers are not measured here. The pairwise cosine "
                "similarities printed below (a dot product, because the vectors are unit-normalised) are a **qualitative "
                "check only**: the rotated copy is expected to score higher against the original than the flat block does, "
                "which shows that the vector encodes image content, but the absolute values mean nothing without a "
                "threshold calibrated on your own labelled pairs, and the pipeline deliberately ships none. The report is "
                "written to `outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                "report = evaluation_report(result, sample_kind=sample_kind)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(report, indent=2))\n"
                "cosine = vectors @ vectors.T\n"
                "print('pairwise cosine similarity (qualitative check only; no threshold is shipped):')\n"
                "for i, name in enumerate(image_ids):\n"
                "    print(f\"{{name:>36}}  \" + '  '.join(f'{{cosine[i, j]:.4f}}' for j in range(len(image_ids))))"
            ),
        },
        {
            "md": (
                "## 8. Export vectors, identifiers, and provenance\n\n"
                "One JSON record is written under `outputs/`: an `items` list with, per image, its identifier, pixel "
                "digest, size and 384-float vector (so every vector maps back to its input), the embedding contract "
                "(`dim`, `pooling`, `normalized`, `input_size`), the pairwise cosine matrix keyed by the same identifiers, "
                "the sanity checks, the input manifest, the evaluation report, the notebook's source (repository, "
                "revision, embedded module digest, generator), the model identifier, the immutable model revision, the "
                "model licence, and the runtime identity (Python, `torch`, `timm`, device). No credentials are involved in "
                "any step, so none can reach the export."
            ),
            "code": (
                "payload = {{\n"
                "    'items': [\n"
                "        {{'id': name, 'pixel_sha256': digests[name], 'width': images[name].width, 'height': images[name].height, 'vector': result['embeddings'][index]}}\n"
                "        for index, name in enumerate(image_ids)\n"
                "    ],\n"
                "    'dim': result['dim'],\n"
                "    'pooling': result['pooling'],\n"
                "    'normalized': result['normalized'],\n"
                "    'input_size': result['input_size'],\n"
                "    'cosine_similarity': {{'ids': image_ids, 'matrix': [[round(float(value), 6) for value in row] for row in cosine]}},\n"
                "    'sanity_checks': checks,\n"
                "    'input_manifest': input_manifest,\n"
                "    'evaluation_report': report,\n"
                "    'sample_kind': sample_kind,\n"
                "    'seconds': round(elapsed, 3),\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'timm': timm.__version__,\n"
                "        'device': pipe.device,\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "Each output is one L2-normalised 384-float vector per image — a representation, not a prediction. There is no "
        "label, no class, no score and no threshold, and the evaluation report says `not-measurable` by construction: a "
        "cosine similarity between two vectors is not an accuracy, and the numbers printed on the synthetic set are "
        "plumbing evidence only. Every image is resized to shorter side 518 px and centre-cropped to 518 x 518, so for a "
        "non-square input the outer strips along the longer side never reach the encoder. The pipeline does not detect "
        "out-of-distribution inputs (drawings, scans, satellite tiles), blur, or capture-device drift, and it exposes no "
        "patch-level features, no attention maps, and no downstream search, clustering or probe.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, "
        "can acquire and digest-verify the pinned model, validate the demonstrated inputs against the enforced ceilings, "
        "execute the public pipeline path, and emit the shown machine-readable outputs in the tested runtime — without "
        "the repository being reachable. It does **not** establish benchmark superiority, reproduction of the upstream "
        "DINOv2 evaluations, retrieval or probe quality on your data, safety for high-consequence decisions, or "
        "production fitness on an unseen domain.\n\n"
        "**Next experiments:** enable `USE_BYOD` with a handful of your own images, including a near-duplicate pair, and "
        "look at whether the cosine ordering matches your judgement; build a small labelled set and fit a linear probe or "
        "k-NN classifier on the exported vectors to get a number that actually means something; compare the CUDA and CPU "
        "cosine matrices on the same batch to see the size of kernel-level variability.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/dinov2-feature-extraction-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/dinov2-feature-extraction-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/dinov2-feature-extraction-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/facebookresearch/dinov2\n"
        "- DINOv2 paper: https://arxiv.org/abs/2304.07193\n"
        "- timm documentation: https://huggingface.co/docs/timm"
    ),
}
