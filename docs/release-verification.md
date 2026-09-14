# Release verification

`tutorials/dinov2_feature_extraction_colab.ipynb` (`TASK-INFERENCE`) is a **release candidate** until
the exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests,
JSON validation, code-cell compilation, and `tools/validate_release_assets.py` are necessary
checks but are **not** runtime evidence under DIMER Notebook Specification 1.1. This file is
the durable release-gate record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no
  persisted outputs or execution counts; no unresolved placeholder markers; every code cell
  is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `TASK-INFERENCE`
  profile, the notebook-spec version and the standalone carrier; `metadata.dimer` declares that profile, spec `1.1`, `standalone: true` and `generated_from` (repository, generating revision, module SHA-256, generator);
- the standalone carrier (ST1–ST6, PAR1–PAR3): no clone, repository install or repository import on the
  primary path; exactly one cell tagged `embedded_module` equal to `src/dinov2_feature_extraction_pipeline/pipeline.py`
  after the generator's documented rewrites; the inline `MANIFEST` equal to the committed snapshot manifest and the
  inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical to `tools/build_notebook.py`
  output; the pinned-install cell with its restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in the export;
- `MODEL_ID`/`MODEL_REVISION` are bound only in the carried module cell (and repeated in the inline manifest,
  which the notebook asserts against the module before fetching), the revision is a 40-hex immutable commit, and the
  same identity string appears in `README.md`, `MODEL_CARD.md`, and `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `DINOv2FeatureExtractionPipeline.from_pretrained(weights_dir=...)`, `validate_inputs`, `embed`, `evaluation_report`),
  the ceiling and contract print (`MAX_IMAGE_SIDE`, `MAX_BATCH`, `EMBED_DIM`, `POOLING`, `NORMALIZED`), the embedding
  sanity checks (one vector per input, width equals `EMBED_DIM`, unit norm), the export, the learner-facing embedding
  statements (representations not predictions, class-token pooling, one vector per image, no missing-data concept, no
  metric helper, cosine check is qualitative only, no threshold shipped) and the gated-off BYOD default listed in
  the validator; forbidden patterns (credential-in-URL, any `git clone` / `github.com` / repository import on the
  primary path, a mutable `revision='main'`, direct `timm.create_model` / `from timm import` / `from torchvision import` /
  `from transformers import` / `from huggingface_hub import` use **outside the carried module cell**, `trust_remote_code=True`,
  `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no
  document makes an unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter, single H1, required heading order, and immutable provenance.

CI also installs the pinned CPU-only `torch`/`torchvision` wheels plus `timm`, runs `ruff`, `tools/build_notebook.py --check`, and the
offline unit suite (`tests/test_pipeline.py`, `tests/test_role_helpers.py`, `tests/test_notebook_parity.py`; injected runner, no weights). These are
source/provenance and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel | Kaggle CPU kernel, Python 3.12 image | Reproducible clean-room executor of the same class; the notebook is pushed verbatim plus one leading shim cell that provides `google.colab` and chdirs to a scratch directory (no repository checkout is needed — the notebook is standalone) |
| Local WSL harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and not promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU (or CUDA) runtime (Colab, or the Kaggle
   executor above) with **no repository checkout** and a clean model cache;
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their
   defaults for the sample path: `USE_BYOD = False`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS` (= `pyproject.toml`);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the carried module cell executes (defines the pipeline class and both role helpers) with no import of the
     repository package;
   - the three synthetic 256 x 256 images generated in code with their pixel SHA-256 values printed, and the
     ceilings (`MAX_IMAGE_SIDE` 4096, `MAX_BATCH` 32) plus the contract (`EMBED_DIM` 384, `POOLING` `cls`,
     `NORMALIZED` true) surfaced before the model runs;
   - pinned `timm/vit_small_patch14_dinov2.lvd142m` acquisition at the immutable revision through the package:
     the inline `MANIFEST` is asserted against the module identity and written to `weights/vit-small-dinov2/`,
     `stage_missing_files(WEIGHTS_DIR, allow_download=True)` reports all three manifest entries
     (`README.md`, `config.json`, `model.safetensors`) on a clean runtime, `verify_snapshot` returns the manifest dict,
     and `from_pretrained(weights_dir=WEIGHTS_DIR)` reports `source == 'local-snapshot'` with the resolved 518 x 518
     input size;
   - `validate_inputs` writes `outputs/dinov2_feature_extraction_input_manifest.json` (verdict `accepted`, one
     recorded rejection finding from the oversized probe);
   - embedding through `embed(...)` with one 384-float unit-norm vector per input, in input order, and all four
     sanity checks true;
   - `evaluation_report` writes `outputs/dinov2_feature_extraction_evaluation_report.json` with verdict
     `not-measurable` and the downstream-task `needs` text, and the pairwise cosine matrix printed as a qualitative
     check (the rotated copy expected to score higher against the original than the flat block);
   - `outputs/dinov2_feature_extraction_result.json` written with every vector keyed by its identifier, the contract,
     the cosine matrix, `NOTEBOOK_SOURCE`, model revision, model licence, runtime versions and device;
6. verify the export exists and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, timm, device),
   model identifier and immutable revision, whether the model cache was clean, outcome, produced
   outputs, and any warning or applicable `SHOULD` deviation in the table below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of `tutorials/dinov2_feature_extraction_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/dinov2_feature_extraction_colab.ipynb`). Wall times, when recorded,
are the sum of per-cell times reported by the executor and include installs and the model download;
they are measurements for the stated runtime, not general estimates.

### Manual clean-runtime evidence

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-14 | `347e21d` / `46d155ab2f9a` | Kaggle CPU (`kurtvalcorza/dimer-nb2-dinov2-feature-extraction` v1) | Default sample path | 194.1 s | **PASSED** — 8/8 ok code cells executed cleanly, 8 files, 88 MB staged |

## Current status

No clean-runtime execution of the notebook has been recorded yet; the run is **pending** and queued
to the GPU lane. Static validation (`tools/validate_release_assets.py`), nbformat validation, a
`compile()` sweep over every code cell, and the offline unit suite passed on the tutorial source at
the candidate revision, which is necessary but not sufficient. The registry status remains
**Candidate** until a reviewer confirms a recorded run against the notebook blob under review and
an integrator promotes it; promotion is not performed by the builder. Two facts a reviewer should
weigh: `stage_missing_files` was exercised only with an injected downloader in the unit suite (the
real `hf_hub_download` fetch of all three manifest entries into a fresh `weights/vit-small-dinov2/` has not been
executed), and the standalone carrier itself — executing the carried module cell in a runtime that has no
repository checkout — has been validated statically only (parity PASS), never run; the earlier local GPU smoke run
used the verified snapshot on CUDA through the installed package, so the clean run will be the first execution of
the standalone path, of the staging path, and of the CPU embedding path against the real weights.
