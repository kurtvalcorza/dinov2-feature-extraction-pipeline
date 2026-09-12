# Tutorials

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/kurtvalcorza/dinov2-feature-extraction-pipeline)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/dinov2-feature-extraction-pipeline/blob/main/tutorials/dinov2_feature_extraction_colab.ipynb)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-timm%2Fvit__small__patch14__dinov2.lvd142m-ffcc4d?style=flat)](https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m)
[![Upstream](https://img.shields.io/badge/Upstream-facebookresearch%2Fdinov2-181717?style=flat&logo=github&logoColor=white)](https://github.com/facebookresearch/dinov2)
[![arXiv](https://img.shields.io/badge/arXiv-2304.07193-b31b1b.svg)](https://arxiv.org/abs/2304.07193)

Notebook specification: **DIMER Notebook Specification 1.0**

| Notebook | Profile | Capability | Default runtime | BYOD | Release status |
|---|---|---|---|---|---|
| `dinov2_feature_extraction_colab.ipynb` | `TASK-INFERENCE` (embedding obligations, §20.6) | DINOv2 ViT-S/14 image feature extraction: one L2-normalised 384-d class-token vector per image on a synthetic in-code sample set (a gradient, its 180-degree rotation, a flat block) with a qualitative pairwise cosine check; no metric exists or is reported | CPU (CUDA used automatically when available) | one or more image files, gated off by default | **Candidate** — static checks pass; the clean-runtime execution row in `../docs/release-verification.md` is pending and must be recorded for the exact notebook revision before promotion |

## Conformance notes

- The notebook exercises `DINOv2FeatureExtractionPipeline` from the repository public API rather than reimplementing model loading; the pipeline pins the immutable upstream revision, stages missing snapshot files through the package's `stage_missing_files(..., allow_download=True)`, loads only from a digest-verified local snapshot (`verify_snapshot`), and executes no remote code. The notebook never calls `timm` or `huggingface_hub` directly.
- §20.6 obligations: the embedding shape (`EMBED_DIM` = 384) and pooling policy (`POOLING` = `"cls"`, the class token after the final LayerNorm, L2-normalised), the per-image unit, the absence of any missing-data concept (every pixel of the 518 x 518 centre crop is visible to the encoder), the statement that embeddings are representations rather than predictions, and the export of identifiers (image id, pixel digest, size) alongside every vector are all stated and implemented.
- No intrinsic metric exists (EVAL9): the repository ships no metric helper; the notebook says so, names the downstream labelled tasks a real evaluation needs (retrieval mAP, linear-probe or k-NN accuracy, human-judged duplicate pairs), and presents the pairwise cosine similarities as a qualitative check only. Recorded `SHOULD` deviation: EVAL11 (no baseline — none is meaningful without a downstream task).
- The default sample set is synthetic and generated in code; its embeddings are plumbing/sanity evidence only.
- `USE_BYOD` defaults to `False` so the sample path never opens an upload dialog.
- The upstream weight license is Apache-2.0; the notebook cites `../docs/WEIGHTS.md` for the resolved provenance record rather than restating the stale `config.json` field.
- `tools/validate_release_assets.py` performs source validation only. It does not satisfy the
  clean-runtime execution requirement; a release review must confirm that a recorded clean run in
  `docs/release-verification.md` matches the notebook revision under review before the status is
  promoted to `Release-grade`.
