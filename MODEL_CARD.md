---
license: apache-2.0
model_card_spec: "1.1"
pipeline_tag: image-feature-extraction
base_model: timm/vit_small_patch14_dinov2.lvd142m
---

# DINOv2 ViT-S/14 lvd142m (DIMER package v0.1.0) — Visual Feature Extraction

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-timm%2Fvit__small__patch14__dinov2.lvd142m-ffcc4d?style=flat)](https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-facebookresearch%2Fdinov2-181717?style=flat&logo=github&logoColor=white)](https://github.com/facebookresearch/dinov2)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2304.07193-b31b1b.svg)](https://arxiv.org/abs/2304.07193)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Pipeline](https://img.shields.io/badge/Pipeline-dinov2--feature--extraction--pipeline-2ea44f?style=flat&logo=github)](https://github.com/kurtvalcorza/dinov2-feature-extraction-pipeline)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This pipeline provides a ready-to-run interactive Google Colab notebook that exercises the repository's public API end to end — bootstrap a fresh runtime, stage and verify the pinned upstream revision, validate an input, run the task, and inspect and export the outputs:

- **Task Inference Tutorial**:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/dinov2-feature-extraction-pipeline/blob/main/tutorials/dinov2_feature_extraction_colab.ipynb) [`dinov2_feature_extraction_colab.ipynb`](https://github.com/kurtvalcorza/dinov2-feature-extraction-pipeline/blob/main/tutorials/dinov2_feature_extraction_colab.ipynb)  
  *DINOv2 ViT-S/14 image feature extraction: one L2-normalised 384-d class-token vector per image on a synthetic sample set with a qualitative pairwise cosine check; embeddings are representations, not predictions, and no metric is reported.*

---

###### Description

`timm/vit_small_patch14_dinov2.lvd142m` is the small (ViT-S/14) DINOv2 vision transformer, pre-trained by Meta AI on the curated LVD-142M image collection with the self-supervised DINOv2 objective — no labels — and published in `timm` as a feature backbone (upstream README; Oquab et al., arXiv:2304.07193), pinned here to revision `4610ca143709d58a633b6397a74412c2c3842454`. At the fixed 518×518 input the image becomes 37×37 = 1369 patches of 14 px plus a class token, 1370 tokens of width 384 (upstream README `forward_features` example), processed by 12 transformer blocks; upstream reports 22.1 M parameters and 46.8 GMACs. The snapshot has `num_classes = 0` and `global_pool = "token"`, so a forward pass returns the class token after the final LayerNorm — a 384-d vector, no logits and no label. This repository fixes that as `POOLING = "cls"`, L2-normalises the vector (`NORMALIZED = True`) so a dot product between two outputs is a cosine similarity, and adds packaging: the `DINOv2FeatureExtractionPipeline` class in `src/dinov2_feature_extraction_pipeline/pipeline.py`, digest verification of the local snapshot (`verify_snapshot`), input validation and a fixed output contract. Nothing is fine-tuned, adapted or conditioned here; there is no metric helper because a representation has no intrinsic accuracy.

#### Intended Use and Limitations

###### Primary Intended Uses

The task is image feature extraction: input one PIL image or a batch of up to `MAX_BATCH = 32` images; output one L2-normalised float32 vector of length `EMBED_DIM = 384` per image plus `dim`, `pooling` and identity fields. Envisioned applications are the ones the DINOv2 paper targets with frozen features: nearest-neighbour image retrieval and de-duplication, clustering of unlabelled photo collections, and linear probes or small heads trained on a user's own labelled data on top of the frozen vector. In a larger system the pipeline is the encoder stage; every decision — what counts as "similar", which cluster means what, which label a probe predicts — belongs to a downstream component the caller builds and evaluates. The four sibling timm classifiers cover the case where a fixed ImageNet label is wanted instead.

###### Primary Intended Users

The intended users are machine-learning engineers, data scientists and application developers building retrieval, clustering or few-shot classification systems in research prototypes, internal enterprise tooling, or the DIMER model workbench. The pipeline assumes its users understand that an embedding carries no label and no score, that cosine similarity between two vectors is meaningful only relative to a threshold they calibrate on their own data, that the vector encodes whatever the self-supervised objective found salient (including background, style and layout, not only the subject), and that any downstream use needs its own labelled evaluation. It is not designed for hobbyist "point and trust" use.

###### Out-of-scope use cases

1. **Capability boundary:** the pipeline does not classify, detect, segment, caption, or compare images to text; it returns a vector and nothing else. Patch-level (dense) features, attention maps and the register-token variants of DINOv2 are not exposed; only the pooled class token is. Similarity search, clustering and probes are the caller's code, not this repository's.
2. **Input boundary:** only PIL images are accepted (`TypeError` otherwise); any side above `MAX_IMAGE_SIDE = 4096` px or below 1 px is rejected; batches above 32 are rejected because each image costs 46.8 GMACs at 518 px. Every image is resized so its shorter side is 518 px and centre-cropped to 518×518 (`crop_pct = 1.0`, `crop_mode = "center"` from the snapshot `config.json`), so content outside the central square of a wide or tall image is discarded. Non-RGB modes are converted to RGB; depth, multispectral and video inputs are unsupported.
3. **Decision boundary:** not for autonomous or high-impact decisions — identity matching, moderation takedowns, medical or forensic similarity — without a human reviewing the match and a locally measured error rate for the downstream task. In particular the vector is not a face or person embedding and must not be used as one.

#### Factors

###### Groups

The pipeline is not human-centric by design — it is a general-purpose image encoder with no person, age, gender or skin-type output — but its embedding space is shaped by LVD-142M, which the DINOv2 paper describes as 142 M images curated from web crawls by retrieval against ImageNet-style seed sets, and which therefore contains photographs of people whose demographic composition is not disclosed. Neither the upstream `timm` card nor this repository reports how similarity or retrieval quality varies across groups; the DINOv2 paper reports fairness probes for its larger models that this card does not restate. The fairness audit transfers to the operator: for any downstream task, measure that task's metric on a labelled sample of your own data stratified by the groups that matter to your application, and treat any material gap as a blocker.

###### Instrumentation

LVD-142M images are web photographs of varied, undocumented provenance — many makes of camera, lens and post-processing, mostly JPEG-encoded. The pipeline consumes decoded pixel arrays, so the instrument sits behind PIL: resolution, JPEG compression level, colour profile, white balance and sensor noise all reach the model as changed pixel statistics after the 518-px resize, and because the output is a similarity-bearing vector, a systematic change of capture device can move a whole batch of images in embedding space together — a retrieval index built from one camera and queried from another is a known failure mode. The pipeline does not detect drift, blur, over-exposure or a change of capture device; it only rejects non-image types and images outside the 1–4096 px side range.

###### Environment

Operating environment: Python 3.12 with `torch==2.14.0`, `torchvision==0.29.0`, `timm==1.0.29`, `pillow==11.3.0` (exact pins in `pyproject.toml`). CUDA is optional; `from_pretrained` picks `cuda:0` when available, else CPU, and runs in float32 on both. On this repository's smoke run (claude-science WSL venv, RTX 5070 Ti 16 GB, two synthetic 256×256 images through `DINOv2FeatureExtractionPipeline.from_pretrained().embed`) loading the verified snapshot took 3.36 s and embedding the pair 1.22 s including transform and first-call CUDA warm-up; both vectors had unit norm and their cosine similarity was 0.9667 (a gradient and its 180° rotation). The CPU path is exercised only by the unit tests with an injected runner. Data environment: inputs are assumed to be natural photographs; the paper reports that DINOv2 features transfer to many domains without fine-tuning, but this pipeline measures nothing about that, and synthetic, medical or satellite imagery should be treated as untested.

#### Metrics

###### Performance Measures

The pipeline reports **no performance measure** and ships no metric helper: its output is a representation, not a prediction, so there is no ground truth to score it against inside this repository. The paper evaluates DINOv2 features through downstream tasks — ImageNet linear probe and k-NN accuracy, retrieval mean average precision, segmentation mIoU — and that is what a caller must do too: choose a downstream task, obtain labels for it, train the probe or set the retrieval threshold, and report that task's metric. Numbers from the paper for ViT-S/14 are not restated here because this pipeline has not reproduced them and the paper's evaluation used its own heads and protocols. What the unit tests do check is structural: 384 floats per image, unit L2 norm, and that different inputs yield different vectors.

###### Decision thresholds

The pipeline applies no threshold and emits no decision: `embed` returns vectors, and the only fixed choices are `POOLING = "cls"` and L2 normalisation, both stated as constants. Any threshold — a cosine cut-off for "duplicate", a distance radius for "same cluster", a probe's class boundary — is deliberately not shipped, because it depends on the downstream task and data, and setting one here would be arbitrary. Calibrating it belongs to the operator: pick the cut-off on a labelled sample of your own pairs, trading false matches (two different images judged the same) against missed matches for your application, and re-check it whenever the data source changes.

###### Approaches to uncertainty and variability

The pipeline reports no accuracy number, so there is no estimation procedure or dispersion to state. Inference is deterministic given the same weights, device and library versions: there is no sampling, dropout is disabled by `model.eval()`, and no seed is required; small numeric differences between CPU, GPU and attention-kernel choices move each vector slightly, so cosine similarities computed on one device are not bit-identical on another — the fourth decimal place should be treated as noise. The output carries no confidence: a cosine similarity is not a probability and is not calibrated; a caller who needs a probability of "same" must fit one on their own labelled pairs.

#### Ethical considerations and biases

###### Data

Upstream states the checkpoint was pre-trained on LVD-142M (upstream README "Pretrain Dataset"); the DINOv2 paper describes it as 142 M images assembled from uncurated web crawls by de-duplication and retrieval against curated seed datasets, and the disclosure stops there — no per-image licensing, consent status or demographic composition is given, and the source is public web imagery, so photographs of identifiable people are present and personal data is not ruled out. License: Apache-2.0. The snapshot README declares `apache-2.0`; the `cc-by-nc-4.0` string in the snapshot `config.json` is a stale timm import field — upstream re-licensed DINOv2 to Apache-2.0 on 2023-08-31, as recorded with the commit reference in `docs/WEIGHTS.md`. This repository distributes code, tests and documentation; the 88 MB `model.safetensors` snapshot is git-ignored and staged locally under `weights/vit-small-dinov2/` with a manifest, and no sample data is shipped. The operator must audit the images they submit for personal, confidential or proprietary content; the pipeline performs no such check.

###### Human Life

The pipeline is not intended for decisions in health, safety, criminal justice, employment, credit, housing or any other domain central to human life, and it has not been validated or certified for any of them by anyone. Its only validation is the offline unit suite and the smoke run in this repository. Image similarity is the building block of person re-identification and face search, which are foreseeable misuses; the vector is not designed or evaluated for them, and any such use, or any other use touching human life, is admissible only with a human reviewer on every consequential outcome, an independent domain evaluation on representative data, and whatever legal clearance the domain requires.

###### Mitigations

Implemented and inspectable in `src/dinov2_feature_extraction_pipeline/pipeline.py`: (1) supply chain — `MODEL_REVISION` is a 40-hex commit; `verify_snapshot` re-hashes every file in `weights/vit-small-dinov2/dimer-base-manifest.json` and raises on the first size or SHA-256 mismatch before any weight is loaded; the Hub path is taken only with `allow_download=True` and then through timm's `hf-hub:<id>@<revision>` form; `trust_remote_code` is never enabled (timm executes no remote code); `from_pretrained` also checks the loaded model's `num_features` equals `EMBED_DIM`. (2) Input integrity — `_validate` rejects non-PIL inputs, empty or over-size batches, and images outside 1–4096 px before the model runs. (3) Reproducibility — exact `==` dependency pins, `model.eval()`, deterministic preprocessing from the snapshot's `pretrained_cfg`, fixed pooling and normalisation, and `model_id`/`model_revision` in every result. (4) Refusals — no dense-feature or training API is exposed; a missing snapshot with `allow_download=False` raises `FileNotFoundError`. No statistical mitigation is applied because the pipeline does not train.

###### Risks and harms

Misuse as a biometric: general image embeddings can be repurposed for person re-identification or face search; data subjects bear the harm, and nothing in the pipeline prevents it beyond the prohibition in this card. Spurious similarity: the vector encodes style, background and layout as well as subject — the smoke run scored a gradient and its 180° rotation at 0.9667 — so near-duplicates by appearance are not duplicates by content, and a retrieval or de-duplication system that trusts the score removes or surfaces the wrong images; operators and end users bear that harm. Bias in the embedding space: LVD-142M's web origin skews toward Western imagery, so retrieval quality is likely uneven across cultures and regions, unmeasured here. Automation bias: a "0.97 similar" reads as certainty. Likelihood is moderate for spurious similarity under normal use; magnitude ranges from a wrong search result to a wrongly linked person.

###### Use cases

The pipeline must not be used for surveillance, biometric identification, person re-identification, demographic profiling, or social scoring; its vectors are not designed or evaluated for any of these, and adapting them to try is a misuse. It must not support unlawful discrimination in employment, housing, credit, insurance, education or healthcare access, nor deceptive or manipulative applications such as fabricating evidence that two images depict the same thing. Any use that violates the license terms of the upstream weights (Apache-2.0 per the snapshot README; see the `cc-by-nc-4.0` discrepancy noted under Data) or the DIMER deployment terms is prohibited. The developers identify no further prohibited use beyond these.

## Immutable provenance

- Model: `timm/vit_small_patch14_dinov2.lvd142m`
- Revision: `4610ca143709d58a633b6397a74412c2c3842454`
- Snapshot manifest: `weights/vit-small-dinov2/dimer-base-manifest.json`, `totalBytes` 88245132
- `model.safetensors` SHA-256: `04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081` (88240510 bytes)
- `config.json` SHA-256: `b651fc1b08edf7d1c15a121cbb3270c802bd87901cfd4860d7ecb286e7d52a05` (615 bytes)
- Weight format: SafeTensors; loader `timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, pretrained_cfg_overlay={"file": ...}, num_classes=0)`

## Input/output contract

- `DINOv2FeatureExtractionPipeline.from_pretrained(device=None, weights_dir=None, allow_download=False)`
- `embed(images)` — `images`: one `PIL.Image.Image` or a sequence of 1–32; sides 1–4096 px; any mode (converted to RGB); resized to shorter side 518 and centre-cropped to 518×518. Returns `{"embeddings": [[float × 384], ...], "dim": 384, "pooling": "cls", "normalized": true, "input_size": [518, 518], "device", "source", "model_id", "model_revision"}`; every vector has unit L2 norm, so `a · b` is the cosine similarity.
- `verify_snapshot(path=None)` — returns the manifest dict with `path`; raises `FileNotFoundError` / `ValueError`.
- No metric helper: there is no intrinsic metric for an embedding; a downstream labelled task is needed.

## Runtime

- Pins: `torch==2.14.0`, `torchvision==0.29.0`, `timm==1.0.29`, `huggingface-hub==0.36.2`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`; Python 3.12.
- Precision: float32 on both CPU and CUDA; preprocessing resize (shorter side) 518 → centre-crop 518×518, bicubic, ImageNet mean/std from the snapshot `config.json`; pooling = class token after the final norm; L2 normalisation applied in `embed`.
- Measured (claude-science WSL venv, RTX 5070 Ti, `HF_HUB_OFFLINE=1`): device `cuda:0`, source `local-snapshot`, load 3.36 s, embed (2 images) 1.22 s, total 4.58 s; output 2 × 384, norms 1.000000 / 1.000000, cosine between a synthetic 256×256 gradient and its 180° rotation 0.9667. CPU not measured.
- Tests: `pytest -q -o addopts= tests` — 11 passed, offline, no weights required; `ruff check src tests` clean.

## References

- Oquab et al. DINOv2: Learning Robust Visual Features without Supervision. 2023. https://arxiv.org/abs/2304.07193
- Dosovitskiy et al. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR 2021. https://arxiv.org/abs/2010.11929
- Original code and weights: https://github.com/facebookresearch/dinov2
- Wightman. PyTorch Image Models. https://github.com/huggingface/pytorch-image-models (doi:10.5281/zenodo.4414861)
- Upstream card: https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m
