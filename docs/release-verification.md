# Release verification

`tutorials/dinov2_feature_extraction_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until
the exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation,
code-cell compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but
are **not** runtime evidence under DIMER Notebook Specification 2.2 (REL8). This file is the durable release-gate
record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.2`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `metrics.py`, `samples.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed 3-entry snapshot manifest and the
  inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; exactly two kernel cells — the isolated install
  (pinned `uv` wheel by size and SHA-256, managed CPython 3.12.12, the carried hash lock installed with
  `--require-hashes --only-binary :all:`, a Linux x86_64 check) and the router to the persistent worker, whose
  `google.colab` stubs carry a module spec; the Infrastructure titles on the seven setup cells; the guided-layer
  markers; no stale text (the stored-head description, untraceable sweep claims, "re-run from that cell");
  `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions (the 180 photo
  digests live in the carried `samples.py`, not in prose);
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `DINOv2FeatureExtractionPipeline.from_pretrained(weights_dir=...)`, `fetch_corpus` from the pinned cache path,
  `read_corpus` + `build_sample_dataset(seed=SPLIT_SEED)` / `load_byod_dataset`, `validate_splits`, `BYOD_PATH`,
  `class_names`, `check_split_disjoint`, `observer_overlap`, `write_dataset_csv`, `validate_inputs` with the
  oversized-image refusal probe, `pipe.embed` with the sanity checks and the two cosines, `majority_baseline`,
  `pipe.reset_to_pretrained()` before Sections 5 and 6, `pipe.knn_baseline`, `pipe.adapt(trainable_blocks=0)` with its frozen-policy assertion, `pipe.evaluate` on the
  frozen policy and on the validation and test splits after the unfrozen policy with the floor assertion,
  `pipe.adapt` with `trainable_blocks=TRAINABLE_BLOCKS` and `lr=LEARNING_RATE` and its epoch-0 check against the Section 6 probe, the
  `run_history` table, `pipe.classify` after with the Section 6 probe's predictions as the frozen column, the
  per-batch `evaluation_report`, `pipe.save_artifact`, `DINOv2FeatureExtractionPipeline.from_artifact` and the
  reload-parity assertion, and the provenance fields `weight_format`, `weight_sha256` and the `corpus` block), the
  six expected `outputs/` paths, the learner-facing statements (representations not predictions, supervised
  adaptation under an explicit frozen-vs-unfrozen policy, the majority floor, the cosine 5-NN vote, the frozen and
  unfrozen policies, lowest validation log-loss, cosine is a similarity not a score, no dispersion estimate, named
  exclusions, the CC0 licence) and the gated-off BYOD default; forbidden patterns (credential-in-URL, any `git
  clone` / `github.com` / repository import on the primary path, a mutable `revision='main'`, direct
  `timm.create_model(` / `from timm` / `from huggingface_hub import` / `urllib.request` / `safetensors` /
  `torch.optim` / `.backward(` / `pipe._model` / `cross_entropy(` / `torch.nn.Linear(` use **outside the carried
  module cells**, `trust_remote_code=True`, `pickle.load`, `torch.load(` without `weights_only=True`,
  `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the 19 required headings in order, and the
  immutable provenance section.

CI also installs the pinned CPU-only torch wheel plus `timm`, `huggingface-hub`, `safetensors`, `numpy` and
`pillow`, runs `ruff check src tests tools`, `tools/build_notebook.py --check`, and the offline unit suite
(`tests/test_pipeline.py`, `tests/test_adaptation.py`, `tests/test_role_helpers.py`, `tests/test_import_boundary.py`,
`tests/test_notebook_parity.py`; injected mean-colour runner and corpus fetcher, synthetic JPEG swatches, temporary
manifests, no weights — `tests/test_model_backed.py` is skipped without the snapshot). These are source/provenance and
unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present; float32 either way) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle kernel or Colab CLI on a fresh VM | Fresh Linux x86_64 CPU or GPU runtime; the committed notebook executed verbatim, every code cell in order in one kernel, **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; the isolated environment leaves the kernel's preloaded NumPy and torch alone, so no restart is expected |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, pre-staged pins, `CUDA_VISIBLE_DEVICES=-1` | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/vit-small-dinov2/` or the photo cache `weights/inat-birds/` (the standalone path writes the
   manifest itself, stages the missing file from the Hub, and fetches the 180 pinned photographs from the
   iNaturalist open-data bucket, so neither directory may be seeded);
3. run the notebook top-to-bottom **once**, without a restart and without editing implementation cells (form
   parameters at their defaults: `USE_BYOD = False`, `BYOD_PATH = ''`, `SPLIT_SEED = 42`, `PROBE_STEPS = 300`, `PROBE_LR = 0.01`, `EPOCHS = 4`,
   `LEARNING_RATE = 3e-5`, `BATCH_SIZE = 8`, `TRAINABLE_BLOCKS = 2`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `torchvision==0.29.0`, `timm==1.0.29`, `huggingface-hub==0.36.2`,
   `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`, imported in the isolated Python 3.12.12; a run that
   needed a restart, or any error, is recorded as such and is not promotion evidence (`restarted: false` is
   required);
5. verify every default-path stage completes:
   - the isolated environment built from the carried hash lock (uv wheel and every package digest-checked) with no
     GitHub access, and every later cell routed to it;
   - the three carried module cells execute (defining `DINOv2FeatureExtractionPipeline`, `verify_snapshot`,
     `stage_missing_files`, `validate_inputs`, `evaluation_report`, `classification_metrics`, `majority_baseline`,
     `knn_predict`, `SAMPLE_RECORDS`, `SPECIES`, `fetch_corpus`, `read_corpus`, `build_sample_dataset`,
     `validate_dataset`, `class_names`, `check_split_disjoint`, `observer_overlap`, `split_dataset`,
     `load_byod_dataset`, `write_dataset_csv` and the ceilings) with no import of the repository package;
   - the inline manifest asserted against the module's constants, then `stage_missing_files(WEIGHTS_DIR,
     allow_download=True)` reporting `['model.safetensors']` (and any other absent entry) fetched from
     `timm/vit_small_patch14_dinov2.lvd142m` at the immutable revision, and `verify_snapshot` returning its dict
     (3 files); `from_pretrained(weights_dir=WEIGHTS_DIR)` loading from the verified directory with `source`
     `local-snapshot` and `input_size` 518 × 518;
   - Section 4: `fetch_corpus` fetching the 180 pinned photographs (19,183,071 bytes) from
     `inaturalist-open-data.s3.amazonaws.com` into `weights/inat-birds/`, six species of 30 read, and the seeded
     stratified draw of 108 / 24 / 48 records with `check_split_disjoint` reporting no shared photograph, the observer
     overlap counted (31 of 117 observers in more than one split in the recorded run) and the three dataset digests
     `1e4cca7f…` / `d176b3ff…` / `0e787f09…`; `outputs/…_train.csv` written; the four dataset refusal probes each
     raising `ValueError`;
   - Section 5: the ceilings (`MAX_IMAGE_SIDE` 4096, `MAX_BATCH` 32) and the contract (`EMBED_DIM` 384, `POOLING`
     `cls`, `NORMALIZED` `True`, `TRANSFORMER_BLOCKS` 12, `PARAMETER_COUNT` 22,056,192) surfaced; `validate_inputs`
     writing `outputs/…_input_manifest.json` (verdict `accepted`, one recorded rejection finding from the oversized
     probe); `embed` on three test photographs with all four sanity checks `True` and the same-species / other-species
     cosines printed;
   - Section 6: the majority floor (16.7 % accuracy), the cosine 5-NN vote (≈ 83.3 % on the sample), and the
     frozen policy — `adapt(trainable_blocks=0)` reporting `frozen backbone + linear probe` with its validation
     metrics — scored on the test split (≈ 83.3 % accuracy, macro-F1 ≈ 0.836) on CPU float32, with the
     cell's assertion that the probe beats the floor and the policy is the frozen one;
   - Section 7: `pipe.adapt` printing epoch 0 as the linear probe (validation log-loss ≈ 0.073, equal to the
     Section 6 probe's: `starts_from_pretrained_base` difference 0), then
     4 unfreeze epochs of the last two blocks (3,550,464 trainable of 22,056,192 parameters plus
     the 2,310-parameter head) with validation log-loss / accuracy each epoch (0.073 (100.0 %) → 0.098 (95.8 %) → 0.092 (95.8 %) → 0.022 (100.0 %) → 0.041 (100.0 %) in the recorded run) and
     the selected policy `unfrozen last 2 blocks + linear head` (`best_epoch` 3);
   - Section 8: `pipe.evaluate` on the validation and test splits with the four-way comparison, per-class recall and
     the confusion matrix, and `outputs/…_evaluation_report.json` written (the cell asserts the selected model beats
     the majority floor — on the sample ≈ 83.3 % versus 16.7 %; the delta over the probe, +0.0
     points, is reported, not asserted);
   - Section 9: six test photographs labelled by the selected model and by the Section 6 frozen-policy probe, printed with
     the gold species and top probabilities, the per-batch `evaluation_report` verdict `not-measurable`,
     `outputs/…_predictions.csv` written; `pipe.save_artifact` writing `outputs/…_adapter/{adapter.safetensors,
     manifest.json}` (30 tensors, about 14.2 MB, `classes` and `policy` recorded) and
     `DINOv2FeatureExtractionPipeline.from_artifact` reloading it with identical probabilities on the six images and
     an identical test accuracy (the cell asserts both); `outputs/…_result.json` written with `NOTEBOOK_SOURCE`, the
     model identity and licence, the snapshot block (`weight_format`, `weight_sha256`), the `corpus` block, the
     inference-contract items, the comparison, the before/after predictions, the artifact digest and policy, the
     reload parity, the runtime versions and device;
6. verify the exports exist and the interpretation section matches the observed path; for the REL12 BYOD gate,
   run one compatible `.zip` through Section 4 onward (export and reload parity included) and one incompatible
   archive that is refused with a message naming the split or the `labels.csv` line; for the active-learning
   journey, after the default run set `TRAINABLE_BLOCKS = 1` and **Run after** from Section 7, and confirm the
   epoch-0 check and the reload parity pass;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, timm, device), the model identifier
   and immutable revision, whether the model cache, the weights directory and the photo cache were clean, outcome,
   produced outputs, the observed metrics and the selected policy (as observations, not a benchmark) and any warning
   or applicable `SHOULD` deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `dinov2_feature_extraction_colab.ipynb` (`E2E`) | `f7f0d3b` / `fa03142c` | 2026-09-19 | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-dinov2-feature-extraction` v4; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers ?` after, Python 3.12.13, `cuda:0`) | **Passed only after a manual restart** — not a one-pass Run all, not promotion evidence. Pass 1 stopped at the install cell's stale-import guard (`cuda-bindings` 12.9.4 → 13.4.2, `numpy` 2.0.2 → 2.5.3); pass 2 after the restart: 11/11 code cells ok; 188 files, 107 MB staged from the Hub into a clean cache; comparison {accuracy: {majority_floor: 0.1667, knn5: 0.8333, frozen_policy: 0.8333, selected_policy: 0.8333}, macro_f1: {majority_floor: 0.0476, knn5: 0.8308, frozen_policy: 0.8361, selected_policy: 0.8403}, log_loss: {frozen_policy: 0.4984, selected_policy: 0.5688}, per_class_recall: {american_goldfinch: {knn5: 1, frozen: 0.88, selected: 0.88}, chipping_sparrow: {knn5: 0.75, frozen: 1, selected: 0.88}, dark_eyed_junco: {knn5: 0.62, frozen: 0.75, selected: 0.75}, house_finch: {knn5: 0.88, frozen: 0.88, selected: 0.88}, song_sparrow: {knn5: 0.75, frozen: 0.62, selected: 0.75}, white_throated_sparrow: {knn5: 1, frozen: 0.88, selected: 0.88}}, delta_vs_frozen: {accuracy: 0, macro_f1: 0.0042}, selected_policy: unfrozen last 2 blocks + linear head}; reload parity {probabilities_identical: True, accuracy_in_memory: 0.8333, accuracy_reloaded: 0.8333, classes_identical: True}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-dinov2-feature-extraction/v4/evidence/` in the workspace |
| `dinov2_feature_extraction_colab.ipynb` (`E2E`) | `530212d` / `7c38c2ee` | 2026-09-19 | Local pre-flight harness (Windows, CPython 3.12.10, CPU, `google.colab` shim, pins pre-installed) | PASS — pre-flight only, **not** promotion evidence |
| `dinov2_feature_extraction_colab.ipynb` (`TASK-INFERENCE`, superseded) | `347e21d` / `46d155ab2f9a` | 2026-09-14 | Kaggle CPU (`kurtvalcorza/dimer-nb2-dinov2-feature-extraction` v1) | PASSED — 8/8 code cells, 194.1 s, 88 MB staged; evidence for the earlier inference-only notebook, not for the `E2E` blob |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/dinov2_feature_extraction_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/dinov2_feature_extraction_colab.ipynb`). Wall times are the sum of per-cell times
reported by the executor and include the model download where it occurred; they are measurements for the stated
runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-10-09 | `a4cecc7` / `c2333f2a` | Colab CLI 0.7.4 sequential execution, fresh Colab VM, Tesla T4 (session `suite-dinov2-a4cecc7-3cb9`; isolated Python 3.12.12 beside kernel Python 3.13.15, `torch 2.14.0+cu130`, `timm 1.0.29`, `cuda:0`). Not a browser Run all: the CLI runs every code cell in order in one kernel and sets no execution counts, so the order is taken from `exec.log` (`Executing cell 1/14` … `14/14`) | Default settings only (sample path; notebook blob SHA-1 verified against GitHub at the commit before the VM was allocated), after the fleet-sweep SWP-A change (quality asserts against the majority floor became recorded verdicts; the frozen-policy contract and reload parity stay hard checks): uv isolated environment (45 locked packages, 56 s) → three carried modules → pinned snapshot digest-verified → 180 photographs, 108 / 24 / 48, 31 of 117 observers in more than one split → embed checks all `True` → floor, 5-NN, probe → unfreeze → comparison with verdicts → predictions before/after → adapter export → reload parity → Section 10 runs table | 220.6 s (session wall, including the isolated install and all downloads) | **PASSED in one pass — no restart, 0 error outputs**, 14/14 code cells (cells 4–6 are the carried module definitions and print nothing). Every printed metric equals the 2026-10-04 run of `638a644`: floor 0.1667 / 0.0476; 5-NN 0.8333 / 0.8308; frozen policy 0.8333 / 0.8361, log-loss 0.4984, validation log-loss 0.0732; unfreeze validation log-loss 0.0732 → 0.098 → 0.0919 → 0.022 → 0.0413, epoch-0 check exact, `best_epoch` 3; selected policy 0.8333 / 0.8403, log-loss 0.5688; one prediction changed (`test-002` → song sparrow, the gold); reload parity exact. New SWP-A lines: Section 6 `frozen_vs_majority_floor: above the majority floor`; Section 8 `verdicts` = above the majority floor / above the majority floor / `no gain`. Adapter 30 tensors / 14,213,808 B, SHA-256 `6683ec0ad0ab8485…` (differs from the 10-04 run's `3de8038e…`; the printed probabilities and metrics agree to the printed precision). Evidence (byte-for-byte): `docs/verification/2026-10-09-colab-t4/dinov2_feature_extraction_colab_a4cecc7_colab-cli-t4_output.ipynb` (SHA-256 `e45519f38061ca464ae1820c983aa71f18ed77e9c3a264c2fcbb4b11969b80ab`), `run_summary.json` (`9e71a95563a190194933c54b4822d171235110c4cb33c9f4e5fb796e10cfc7a2`), `exec.log` (`b64c91ad062eb37d322c9cd319967ed274460adce7a8438b38c6ff95c5bb13ac`). Not exercised: the BYOD journey (REL12; the case SWP-A changes), the Section 10 change-one-thing rerun, browser-only interactions |
| 2026-10-04 | `638a644` / `0796c23c` | Colab CLI 0.7.4 sequential execution, fresh Colab VM, Tesla T4 (session `suite-dinov2-638a644-3cb9`; isolated Python 3.12.12 beside kernel Python 3.13.15, `torch 2.14.0+cu130`, `timm 1.0.29`, `cuda:0`). Not a browser Run all: the CLI runs every code cell in order in one kernel and sets no execution counts, so the order is taken from `exec.log` (`Executing cell 1/14` … `14/14`) | Default settings only (sample path; notebook blob SHA-1 verified against GitHub at the commit before the VM was allocated): uv isolated environment (45 locked packages, 56 s) → three carried modules → snapshot pinned at `4610ca14…`, 3 files fetched and digest-verified → 180 photographs, six species of 30, 108 / 24 / 48 with digests `1e4cca7f…` / `d176b3ff…` / `0e787f09…`, 31 of 117 observers in more than one split, four dataset refusals → embed of three test photographs with all four checks `True` → floor, 5-NN, probe → unfreeze → comparison → predictions before/after → adapter export → reload parity → Section 10 runs table | 224.4 s (session wall, including the isolated install and all downloads) | **PASSED in one pass — no restart, 0 error outputs**, 14/14 code cells (cells 4–6 are the carried module definitions and print nothing); majority floor 0.1667 / macro-F1 0.0476; cosine 5-NN 0.8333 / 0.8308; frozen policy (linear probe) 0.8333 / 0.8361, test log-loss 0.4984, validation 24/24 at log-loss 0.0732; unfreeze history validation log-loss 0.0732 → 0.098 → 0.0919 → 0.022 → 0.0413 with accuracy 100 / 95.83 / 95.83 / 100 / 100 %, epoch-0 check exact (abs difference 0.0), selected `unfrozen last 2 blocks + linear head` at `best_epoch` 3; **selected policy on the test split 0.8333 / macro-F1 0.8403, log-loss 0.5688 (Δ 0.0 accuracy, +0.0042 macro-F1 vs the frozen policy)**; six predictions, one changed (`test-002`, white-throated → song sparrow, the gold); per-batch report `not-measurable`; adapter 30 tensors / 14,213,808 B, SHA-256 `3de8038e96d49c96…`; reload parity exact (probabilities identical, accuracy 0.833333 both ways); six exports written. All printed metrics equal the 2026-09-19 runs of the previous version. Evidence (byte-for-byte): `docs/verification/2026-10-04-colab-t4/dinov2_feature_extraction_colab_638a644_colab-cli-t4_output.ipynb` (SHA-256 `53309e6b385a52b6fd51897b3cd98590da7bc311e08847a7b471ca94d29f848e`), `run_summary.json` (`bea152c48402141bf928b58a7c3b91dce79a61c2e0f2cf1303798b931079bb2a`), `exec.log` (`6289a8760e837947b8d67b632b5b719944bc3023e85114f255ca3bdf6b270bb6`). Not exercised: the BYOD journey (REL12), the Section 10 change-one-thing rerun (`TRAINABLE_BLOCKS = 1`, Run after), browser-only interactions (forms, upload dialog) |
| 2026-09-19 | `f7f0d3b` / `fa03142c` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-dinov2-feature-extraction` v4; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers ?` after, Python 3.12.13, `cuda:0`) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout (blob SHA-1 verified against GitHub before execution) | 293.8 s | **Passed only after a manual restart** — not a one-pass Run all, not promotion evidence. Pass 1 stopped at the install cell's stale-import guard (`cuda-bindings` 12.9.4 → 13.4.2, `numpy` 2.0.2 → 2.5.3); pass 2 after the restart: 11/11 code cells ok; 188 files, 107 MB staged from the Hub into a clean cache; comparison {accuracy: {majority_floor: 0.1667, knn5: 0.8333, frozen_policy: 0.8333, selected_policy: 0.8333}, macro_f1: {majority_floor: 0.0476, knn5: 0.8308, frozen_policy: 0.8361, selected_policy: 0.8403}, log_loss: {frozen_policy: 0.4984, selected_policy: 0.5688}, per_class_recall: {american_goldfinch: {knn5: 1, frozen: 0.88, selected: 0.88}, chipping_sparrow: {knn5: 0.75, frozen: 1, selected: 0.88}, dark_eyed_junco: {knn5: 0.62, frozen: 0.75, selected: 0.75}, house_finch: {knn5: 0.88, frozen: 0.88, selected: 0.88}, song_sparrow: {knn5: 0.75, frozen: 0.62, selected: 0.75}, white_throated_sparrow: {knn5: 1, frozen: 0.88, selected: 0.88}}, delta_vs_frozen: {accuracy: 0, macro_f1: 0.0042}, selected_policy: unfrozen last 2 blocks + linear head}; reload parity {probabilities_identical: True, accuracy_in_memory: 0.8333, accuracy_reloaded: 0.8333, classes_identical: True}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-dinov2-feature-extraction/v4/evidence/` in the workspace |
| 2026-09-19 | `530212d` / `7c38c2ee` | Local pre-flight harness (Windows, CPython 3.12.10, CPU float32, `torch 2.14.0+cu130` with `CUDA_VISIBLE_DEVICES=-1`, `timm 1.0.29`) | Default sample path (install skipped, pins pre-installed → three carried modules → inline manifest assert → `stage_missing_files` fetched 0 of 3 entries because the snapshot was pre-staged → `verify_snapshot` 3 files → `from_pretrained` on CPU at 518 × 518 → `fetch_corpus` served from the pre-staged cache after its 180 digest checks → six species of 30 read, 108 / 24 / 48 drawn with `check_split_disjoint` clean, 31 of 117 observers in more than one split, digests `1e4cca7f…` / `d176b3ff…` / `0e787f09…` → four dataset refusals → input manifest with the oversized-image refusal → `embed` of three test photographs (0.48 s) with all four sanity checks `True` → majority floor → 5-NN vote → frozen-policy probe → unfrozen-policy `adapt` → validation + test evaluation → predictions before/after (a fresh probe pipeline for the frozen column) → adapter export → reload parity) | 183.7 s | **PASSED** — 11/11 code cells; majority floor 16.7 % / macro-F1 0.048; cosine 5-NN 83.3 % / 0.831; frozen policy (linear probe, 300 steps, 18.7 s incl. features) 83.3 % / 0.836, log-loss 0.498, validation 100 %; `adapt` with the unfreeze: 3,550,464 block + 2,310 head parameters, 4 epochs, 96.6 s, validation log-loss 0.073 (probe) → 0.098 → 0.092 → 0.022 → 0.041 with accuracy 100 / 95.8 / 95.8 / 100 / 100 %, selected `unfrozen last 2 blocks + linear head` at `best_epoch` 3; **selected policy on the test split 83.3 % / macro-F1 0.840, log-loss 0.569 (Δ +0.0 accuracy, +0.004 macro-F1 vs the probe)**; per-class recall 0.88 / 0.88 / 0.75 / 0.88 / 0.75 / 0.88 (goldfinch, chipping sparrow, junco, house finch, song sparrow, white-throated sparrow); six predictions printed — one changed (`test-002`, white-throated → song sparrow, the gold); per-batch report `not-measurable`; adapter 14,213,808 B / 30 tensors, SHA-256 `7030cbd1…`; reload parity exact (probabilities identical, test accuracy 0.833333 both ways); six exports written. Pre-flight; hosted clean-runtime run still required |
| 2026-09-14 | `347e21d` / `46d155ab2f9a` (`TASK-INFERENCE`, superseded) | Kaggle CPU (`kurtvalcorza/dimer-nb2-dinov2-feature-extraction` v1) | Default sample path of the inference-only notebook: three synthetic images, `stage_missing_files` fetching `model.safetensors` from the Hub, `verify_snapshot` over 3 files, `embed` with its sanity checks and a qualitative cosine table, `not-measurable` report, CSV + JSON exports | 194.1 s | **PASSED** — 8/8 code cells, 88 MB staged; history only |

## Current status

**Candidate.** The notebook was revised after the Notebook Review Framework v1 review (PR #9: DV2-M1..M3,
DV2-m1..m4): a uv isolated environment replaces the in-kernel install, every adaptation starts from the
pretrained base, BYOD splits are validated with the minimums the pipeline applies, and the guided layer is added.
A hosted one-pass run of the new blob is recorded: the 2026-10-04 Colab CLI sequential execution on a fresh
Tesla T4 of `638a644` / `0796c23c` passed 14/14 code cells with no restart and 0 errors (default settings only; not
a browser Run all). After the 2026-10-05 fleet-sweep change SWP-A (quality asserts became recorded verdicts), the
current blob `c2333f2a` (`a4cecc7`) passed the same Colab CLI sequential execution on a fresh Tesla T4 on 2026-10-09:
14/14 code cells, no restart, 0 errors, every printed metric unchanged. It stays **Candidate**. The 2026-09-19 Kaggle Tesla T4 run of blob `fa03142c` (committed at
`f7f0d3b`) passed only after a manual restart after the install cell, so it is not a one-pass Run all and not
promotion evidence; the earlier rows remain history. Promotion still needs the REL12 BYOD exercise and the active-learning rerun of step 6, plus
any remaining steps of the procedure above (for example a Kaggle run of the current blob).
