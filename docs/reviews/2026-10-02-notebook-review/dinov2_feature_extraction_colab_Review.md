# DINOv2 Feature Extraction E2E Notebook — Review

**Verdict: Needs revision**  
**Review date:** 3 October 2026 (relay batch of 2 October 2026)  
**Repository:** `kurtvalcorza/dinov2-feature-extraction-pipeline`  
**Notebook:** `tutorials/dinov2_feature_extraction_colab.ipynb`  
**Reviewed commit:** `7b430969252f7f70712db646a93f2dfe88ca524f` (`main`, confirmed with `gh api repos/kurtvalcorza/dinov2-feature-extraction-pipeline/commits/main`)  
**Notebook Git blob:** `fa03142cca23d67fdec34ab99c0685f9007cd748`. This is the blob executed in the recorded Kaggle Tesla T4 run of 2026-09-19 (commit `f7f0d3b`, `fetched_blob_verified: true`); the notebook last changed in `a7f5488`. At the reviewed commit `tools/build_notebook.py --check` and `tools/validate_release_assets.py` both exit 0.  
**Finding prefix:** `DV2`  
**Framework:** Notebook Review Framework v1. **Requirements baseline:** NOTEBOOK_SPEC 2.2 (2026-09-26), `ml-worker` `origin/main`.

## Executive assessment

The default path is careful and it reproduces exactly. The notebook carries the package's three modules (126 + 801 + 2,166 lines, one documented rewrite), asserts the inline manifest against the module identity, stages and re-hashes the pinned `timm/vit_small_patch14_dinov2.lvd142m` snapshot, fetches 180 digest-pinned CC0 iNaturalist photographs, validates and splits them 108 / 24 / 48 with a pixel-digest disjointness check and an observer-overlap count, exercises the embedding contract with an input manifest and a rejection probe, frames adaptation with a majority floor and a cosine 5-NN vote, trains a linear probe (frozen policy) and a bounded unfreeze of the last two blocks selected against it by validation log-loss, evaluates once on the held-out split, and exports a safetensors adapter that reloads onto a fresh base with exact parity. Score semantics (cosine is a similarity; softmax is not a calibrated confidence), the small-sample limit (one image ≈ 2.1 points) and the observer leakage caveat are stated correctly.

A direct CPU run of every code cell at the documented defaults reproduced the Kaggle record to four decimals:

| Measure | This review (CPU, pins, defaults) | Kaggle T4 record (blob `fa03142c`) |
|---|---|---|
| Code cells completed | 11/11 (1,102 s of cell time on a contended CPU) | 11/11 on pass 2 (pass 1 stopped at the install guard) |
| Split / observers in > 1 split | 108 / 24 / 48; 31 of 117 | identical |
| Dataset digests | `1e4cca7f…` / `d176b3ff…` / `0e787f09…` | identical |
| Accuracy: floor / 5-NN / frozen / selected | 0.1667 / 0.8333 / 0.8333 / 0.8333 | identical |
| Macro-F1: floor / 5-NN / frozen / selected | 0.0476 / 0.8308 / 0.8361 / 0.8403 | identical |
| Test log-loss: frozen / selected | 0.4984 / 0.5688 | identical |
| Validation log-loss, epochs 0–4 | 0.073 / 0.098 / 0.092 / 0.022 / 0.041; selected epoch 3 | identical |
| Reload parity | probabilities identical, accuracy 0.833333 both ways | identical |

Three problems stand in the way of `Ready for intended use`:

1. **No one-pass `Run all` (DV2-M1).** The recorded run stopped at the install cell's stale-module guard (`cuda-bindings` 12.9.4 → 13.4.2, `numpy` 2.0.2 → 2.5.3) and passed only after a restart; the release record reports it as PASSED and the procedure calls the restart "expected".
2. **Every documented rerun starts from the adapted backbone (DV2-M2).** `adapt` writes the selected block weights into the pipeline's model in place. The optional experiments ("set `TRAINABLE_BLOCKS = 1` or `4` … set `LEARNING_RATE = 1e-4` … set `EPOCHS = 8`") and the BYOD instruction ("re-run from that cell") all reuse that model, so the "frozen" probe, the k-NN baseline and the unfreeze no longer start from DINOv2, and for some settings the reload-parity assertion cannot hold.
3. **Guided layer largely absent (DV2-M3).** Declared `GUIDED`, but there is no audience statement, how-to-use, roadmap, task contract, glossary, prediction prompt, checkpoint, troubleshooting or conclusion template, and the 3,093 carried lines are not labelled as infrastructure.

The REL12 BYOD release gate has not been recorded on a hosted runtime; this review ran six incompatible BYOD archives through the cell-13 code path locally (§4). The positive BYOD run and the active-learning rerun were not executed within the probe time cap (§1).

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `E2E` / `GUIDED` (metadata `dimer.notebook_profile` / `notebook_mode`, opening cell) |
| Declared spec | DIMER Notebook Specification **2.0** (metadata, opening cell, `NOTEBOOK_SOURCE`) |
| Spec baseline applied | NOTEBOOK_SPEC **2.2** |
| Intended audience | Not stated. Prerequisites: "basic Python and NumPy; what an embedding vector is; what a linear probe is …; what accuracy and macro-F1 measure …; what validation-based selection between two policies means" |
| Supported runtime | "Google Colab or Jupyter, Python 3.12"; CPU float32 default, CUDA used automatically |
| Promised outcomes | Pinned install; carried modules; digest-verified snapshot; 180 pinned photographs validated and split without leakage; embedding contract with input manifest and rejection probe; majority floor, 5-NN vote and frozen policy; bounded unfreeze selected on validation; held-out accuracy / macro-F1 with per-class recall; predictions before and after; safetensors adapter with verified reload parity; six `outputs/` files; BYOD through "the same validation, … split, floors, frozen-policy probe, unfrozen-policy training and selection, held-out evaluation, prediction, artifact export and reload-parity cells"; optional experiments |
| Generator | `tools/build_notebook.py` (`build_notebook.py/2`) + `tools/notebook_template.py`; recorded generating revision `fd60557` |
| Release status | `Release-grade` (`tutorials/README.md`, `STATUS.md`, `README.md`, `docs/release-verification.md`) |

### Evidence actually obtained

- **Source inspection.** All 25 cells (11 code; cells 5, 7, 9 are the carried `metrics.py`, `pipeline.py`, `samples.py`). Also read: the generator and template, `pipeline.py` (`adapt`, `save_artifact`, `load_artifact`, `features`), `samples.py` (`fetch_corpus`, `validate_dataset`, `split_dataset`, `load_byod_dataset`), `README.md`, `STATUS.md`, `MODEL_CARD.md`, `tutorials/README.md`, `docs/release-verification.md`. The repository has no `AGENTS.md` and no `docs/execution-evidence/`.
- **Documented execution evidence.** `docs/release-verification.md` plus the archived executor output for Kaggle kernel `kurtvalcorza/dimer-nb2-dinov2-feature-extraction` v4 (`run_summary.json`, `executed-pass1.ipynb`, `executed.ipynb`, `outputs/`). Kaggle Tesla T4, 2026-09-19, **the reviewed blob**, clean HF cache, image torch 2.10.0+cu128 / numpy 2.0.2. Pass 1 failed in cell 3 with the restart `RuntimeError` (163.7 s); pass 2 ran 11/11 (130.1 s); 293.8 s total. No Colab run, no BYOD run and no optional-experiment run is recorded.
- **Direct execution (this review).**
  - **Environment:** `run_probes.py`, Windows 11, CPU only (`CUDA_VISIBLE_DEVICES=-1`), Python 3.12.12, torch 2.14.0+cpu, torchvision 0.29.0+cpu, timm 1.0.29, huggingface-hub 0.36.2, safetensors 0.8.0, numpy 2.5.3, pillow 11.3.0 — the notebook's pins (CPU wheel), taken read-only from another pipeline repository's `.venv` because this repository has none. Nothing was installed.
  - **Install skipped:** cell 3 ran with `DIMER_NOTEBOOK_CI_PREINSTALLED=1`, the notebook's executor hook.
  - **Clean assets:** empty scratch `HF_HOME` and working directory; cell 11 fetched `model.safetensors` from the Hub at the pinned revision and `verify_snapshot` passed; cell 13 fetched all 180 photographs from the iNaturalist bucket (249 s including the downloads).
  - **Executed:** every code cell at defaults (P1, 1,102 s of cell time; the host was shared with other sessions, so the 5-NN pass took 430 s — timings are not a runtime claim). A hook after cell 17 recorded the probe's own predictions on the six Section 9 images (P1b). Six incompatible BYOD archives through cell 13 with `USE_BYOD = True` and a `google.colab.files.upload` shim, carried modules only, no model (P4).
  - **Not executed (probe cap):** the 1,200 s cap expired during the active-learning rerun (`TRAINABLE_BLOCKS = 1`, cells 19 → 21 → 23), which was killed before it wrote a result; the positive BYOD run (60 images, 4 classes) queued after it never started. Both journeys are **not verified** by execution; DV2-M2 rests on source inspection.
- **Learner observation:** none. No claim here is about measured learning effectiveness.

## 2. Separate judgments

- **Technical correctness:** strong on the default path (P1 11/11, identical to the hosted record; snapshot and every photograph digest-checked; split disjointness asserted; reload parity exact). Two defects: the install pattern forces a restart (DV2-M1), and in-place adaptation makes every documented rerun start from a modified backbone (DV2-M2).
- **Promise fulfilment:** the default promises are met and the interpretation section reports the observed result honestly ("changed held-out accuracy by nothing"). The optional experiments and the BYOD rerun do not deliver what they promise from a default session (DV2-M2); the BYOD contract states a minimum the code does not accept (DV2-m1).
- **Learner experience:** accurate, unusually candid prose (floors first, leakage by observer, one image ≈ 2 points, policy as a hypothesis), but the guided layer is missing (DV2-M3), a template escape leaks into the data contract (DV2-m2), and Section 9 describes a computation the code does not perform (DV2-m3).
- **Spec conformance:** unresolved applicable MUSTs — RUN1, RUN10, ENV6, REL2 (DV2-M1); UX12 for unlabelled measurement claims (DV2-m4); REL12 BYOD evidence absent from the release record. SHOULD deviations: GDL1–GDL7, GDL9–GDL14, UX8 (DV2-M3); GDL10/UX5 experiment validity (DV2-M2); EXE2 (DV2-S3).

## 3. Promise and objective tracing

| Claim / objective | Implementation | Observable result | Learner interpretation | Status |
|---|---|---|---|---|
| One-pass `Run all` | cell 3 in-kernel `pip install` + stale-module guard | Kaggle pass 1 `RuntimeError`, restart, pass 2 11/11 | Section 1 says the cell "stops with a restart instruction" | **Not met** (DV2-M1) |
| Digest-verified pinned snapshot | cell 11 | 3/3 files verified at `4610ca14…`; `model.safetensors` fetched | clear | Met |
| 180 pinned photographs, validated, split without leakage | cell 13 | 180 read; 108/24/48; disjoint; 31 of 117 observers in > 1 split; four refusal probes rejected | observer leakage explained | Met |
| Embedding contract | cell 15 | four sanity checks `True`; oversized probe rejected and recorded | "cosine is a similarity, not a score" | Met |
| Floor, 5-NN, frozen policy | cell 17 | 0.1667 / 0.8333 / 0.8333 accuracy | per-class recall pointed to | Met |
| Bounded unfreeze selected by validation log-loss | cell 19 | epoch 3 selected (0.022 vs probe 0.073) | prose warns the probe is already near-perfect | Met |
| Independent held-out evaluation, four-way comparison | cell 21 | Δ accuracy 0.0, Δ macro-F1 +0.004; test log-loss worse (0.569 vs 0.498) | "it does not assert a gain over the probe"; test log-loss not discussed (DV2-S2) | Met |
| Predictions before and after | cell 23 | 6 rows; frozen column identical to cell 17's probe | prose says "recomputed from the probe's stored head"; code retrains a probe on a second base (DV2-m3) | Met, explanation inaccurate |
| Adapter export + fresh reload parity | cell 23 | 30 tensors; probabilities identical; accuracy equal | "loading is not reproducing" made explicit | Met |
| Optional experiments: change blocks / LR / epochs "and watch the selection" | cell 19 fields | not executed (cap); source: rerun continues from adapted blocks | none | **Not met** (DV2-M2, source) |
| BYOD through the same stages | cell 13 `USE_BYOD`, then cells 15–23 | negatives refused (5 of 6 with a named rule); positive not executed | contract stated first; minimum misstated | Partly verified (DV2-M2, DV2-m1) |

| Learning objective (opening cell) | Learner activity | Evidence exercised |
|---|---|---|
| Install the pinned runtime; read what the carried modules guarantee; stage and verify the snapshot | run cells | printed identity and verified-file count; no activity |
| Fetch, validate and split the corpus without leakage | run cell, read probes | outputs readable; no prediction or interpretation prompt |
| Embed images and read the vector contract | run cell | sanity checks and two cosines printed |
| Read accuracy / macro-F1 beside floor and k-NN; train probe and unfreeze; evaluate; compare predictions | run cells, read tables | outputs readable; no checkpoint asks the learner to interpret them |
| Export and reload the adapter with verified parity | run cell | parity printed and asserted |

The objectives are phrased as operations the code performs (GDL5), and the only learner-controlled activity — the optional experiments — is invalid after a default run (DV2-M2).

## 4. Journeys

| Journey | Basis | Result |
|---|---|---|
| **First-time learner** | Source inspection, all 25 cells | Stage prose is accurate and explains why each number matters (floors, selection rule, small-sample limit, observer leakage, softmax semantics). Missing: audience, how-to-use, roadmap, task contract, glossary, prediction prompts, checkpoints, troubleshooting, conclusion template; carried cells (3,093 lines) unlabelled and uncollapsed (DV2-M3). The data contract shows `{{id, image, label}}` and `[A-Za-z0-9_.:-]{{1,64}}` (DV2-m2). |
| **Clean default** | Documented (Kaggle T4, reviewed blob) + direct (CPU, install skipped) | Kaggle: pass 1 failed at the install guard, pass 2 11/11 after a restart (DV2-M1). Direct: 11/11 at defaults with a clean model cache and a fresh photo download; every comparison value, digest, validation-loss history and the reload parity equal the Kaggle record (table above); six outputs written. No Colab run. |
| **Active learning** | **Not verified** by execution (probe cap); source inspection | Interpretation: "set `TRAINABLE_BLOCKS = 1` or `4` and watch the selection; set `LEARNING_RATE = 1e-4` …; set `EPOCHS = 8` …" with no rerun scope. `adapt` merges the selected block weights into `self._model` (`pipeline.py` lines 590–594) and starts every call from that model, so a rerun of cell 19 trains its epoch-0 probe on the adapted features and continues the unfreeze from the adapted blocks; `save_artifact` writes only the blocks named in the latest run, so with `TRAINABLE_BLOCKS = 1` the in-memory block 10 differs from the base the reload uses and cell 23's parity assertion is expected to fail (DV2-M2). |
| **Reuse and recovery** | Direct (P4, no model) + source; Colab upload dialog not verified | Positive BYOD not executed (cap). Incompatible archives through cell 13: no `labels.csv` → "BYOD zip must contain labels.csv"; missing `label` column → "labels.csv is missing columns ['label']"; single class → "1 distinct labels; 2..100 are required"; a row naming a file not in the zip → raw `KeyError: 'ghost.jpg'`; a text file named `.jpg` → `UnidentifiedImageError: cannot identify image file <_io.BytesIO …>` with no file name; an **in-contract** 40-image, 2-class set (the Prerequisites say 8..20,000 records) → "6 records; 8..20000 are required", which is the validation split, not the dataset (DV2-m1). A seventh probe (basename collision) was mis-built and is not counted. Source: following the BYOD instruction after the default run reuses the adapted backbone (DV2-M2). |

## 5. Findings

### Major

#### DV2-M1 — `Run all` needs a manual restart after the install cell, and the release record counts the restarted run

- **Cell/section:** cell 3, Section 1 (generator `tools/build_notebook.py`, `_INSTALL_GUARD` lines 47–70 and the install-cell assembly around line 470; Section 1 prose); `docs/release-verification.md` procedure step 4 ("an interpreter restart after the install is expected") and the evidence tables; `STATUS.md`, `README.md`, `tutorials/README.md` release lines.
- **Observed issue:** the cell `pip install`s eight pins into the running kernel, then raises `RuntimeError: Core dependencies changed while older modules were loaded … Restart the runtime, then rerun from the top.` when a loaded distribution changed. Section 1 and the release procedure present this as designed.
- **Consequence:** a learner selecting **Run all** on a stock Kaggle or Colab image hits an error in the first code cell and must restart and run again. RUN1, RUN10 and ENV6 forbid this, and the registry status `Release-grade` rests on a restart-dependent run.
- **Evidence:** documented — Kaggle T4 run of blob `fa03142c`, pass 1 `ok: false` with `cuda-bindings: loaded=12.9.4, installed=13.4.2; numpy: loaded=2.0.2, installed=2.5.3`, `restarted_after_install_cell: true`, pass 2 11/11; the record reports "**PASSED** — 11/11 code cells ok (1 restart after install cell)". Source — `pip_install_in_kernel: true`, `uses_uv: false` (probe static block).
- **Recommended correction:** adopt the fleet's **uv isolated-environment pattern**, which is how the capstone and newer workshop notebooks already run in one pass: the setup cell bootstraps uv, creates an isolated managed interpreter (`uv venv --managed-python --python 3.12.12 <ROOT>/env`), installs a hash-locked `requirements.txt` compiled with `uv pip compile` (`uv pip install --require-hashes --only-binary :all:`), and runs the pinned stages in that environment, so the kernel's preloaded NumPy/torch are never replaced and no restart can be required. Reference implementations on `main`: `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` and `bioclip2-biodiversity-pipeline/tutorials/DIMER_Philippine_Biodiversity_Field_Survey_Capstone.ipynb`. Do not add another in-kernel install guard or loosen pins to dodge the restart. Implement it in `tools/build_notebook.py`, regenerate, re-qualify with a one-pass hosted Run all, and correct the release record so a restart-dependent run is not reported as a `Run all` PASS.
- **Acceptance check:** a fresh Kaggle or Colab runtime completes every code cell in a single **Run all** with no restart and no error, recorded in `docs/release-verification.md` with the notebook blob id and `restarted: false`; `grep -n "Restart the runtime" tutorials/dinov2_feature_extraction_colab.ipynb` returns nothing; the release procedure no longer calls a restart expected.
- **Spec:** RUN1, RUN10, ENV6, REL2.

#### DV2-M2 — Adaptation modifies the backbone in place, so the optional experiments and the BYOD rerun start from the sample-adapted model

- **Cell/section:** cells 17, 19, 23 and the BYOD path of cell 13; Interpretation "Optional experiments"; opening cell BYOD paragraph. Code: `pipeline.py` `adapt` (lines 479, 590–594), `save_artifact` (trainable_names only), `features`. Generator: `tools/notebook_template.py` line 38 (BYOD "re-run from that cell") and line 442 (optional experiments).
- **Observed issue:** `pipe` is loaded once in cell 11. When the unfrozen policy is selected (the default outcome, epoch 3), `adapt` writes the selected block weights into `pipe`'s model and leaves them there; the notebook acknowledges that `embed` changes but nothing resets the model. Every documented follow-up reuses `pipe`:
  - "set `TRAINABLE_BLOCKS = 1` or `4` … `LEARNING_RATE = 1e-4` … `EPOCHS = 8` and watch the selection" — re-running cell 19 trains its "epoch 0, the probe" on adapted features and continues the unfreeze from adapted blocks; cell 21 then compares the result with the pristine `frozen_test` from cell 17;
  - with `TRAINABLE_BLOCKS = 1`, `save_artifact` writes only block 11 and the head, while the in-memory block 10 is still the sample-adapted one, so `from_artifact` (pristine base + saved tensors) cannot reproduce the in-memory probabilities and cell 23's parity assertion should fail;
  - BYOD "set `USE_BYOD = True` in Section 4 and re-run from that cell": the "cosine 5-NN vote over the frozen features" and the "frozen policy" are computed on the bird-adapted backbone, not on DINOv2; if the BYOD selection keeps the probe, the artifact holds only the head and the parity assertion should fail for the same reason.
  No rerun scope ("re-run cells 11–23") is given anywhere.
- **Consequence:** the notebook's only learner-controlled experiment, and the BYOD branch it promises uses "the same … cells", give contaminated comparisons or an `AssertionError` after a default run; a learner is likely to draw the wrong conclusion about how many blocks to unfreeze.
- **Evidence:** source inspection (`adapt` reuses `self._model`; lines 590–594 merge `best_state["blocks"]` into it; `save_artifact` saves `adapter["trainable_names"]` only; `load_artifact` overlays onto a freshly verified base). **Not verified by execution:** the `TRAINABLE_BLOCKS = 1` rerun probe was started and killed by the 1,200 s cap before it reported; the BYOD positive run did not start. The default run confirms the precondition: the unfrozen policy is selected (`best_epoch` 3).
- **Recommended correction:** make `adapt` start from the verified base on every call (restore the initial block weights captured at load, or reload `from_pretrained(weights_dir=WEIGHTS_DIR)` at the top of cell 17 and cell 19), keep the frozen-policy pipeline separate from the unfrozen one, and in the BYOD and experiment text state the exact rerun range ("re-run cells 17–23"). Have cell 21 print the default-run comparison next to the changed one.
- **Acceptance check:** after a default Run all, setting `TRAINABLE_BLOCKS = 1` and re-running the documented cells gives an epoch-0 validation log-loss equal (to 1e-6) to cell 17's probe validation log-loss of the same session, and cell 23's parity assertion passes; the same holds for the BYOD rerun with a dataset where the probe is selected.
- **Spec:** GDL10, UX5, UX7, SRC2, DAT13, VER4.

#### DV2-M3 — Declared `GUIDED`, but the guided layer is largely absent

- **Cell/section:** opening cells 0–1, every section boundary, cells 3, 5, 7, 9, end of notebook. Generator: `tools/notebook_template.py` and `tools/build_notebook.py` (section assembly).
- **Observed issue:** no intended-learner statement, no **How to use this notebook**, no roadmap, no Input → Model → Output contract, no glossary (class token, L2 normalisation, linear probe, k-NN vote, macro-F1, log-loss, frozen/unfrozen policy), no prediction before the floors, the unfreeze or the held-out comparison, no interpretation checkpoint with a sample answer, no troubleshooting section (download failure from the Hub or the iNaturalist bucket, digest mismatch, CPU time, out-of-memory on GPU, BYOD errors), no evidence-based conclusion template; three "Look for"/"Watch" notes only. The four code cells holding infrastructure (install and three carried modules, 3,093 lines) carry no **Infrastructure** label and no `cellView: form`.
- **Consequence:** a self-paced learner new to representation probing gets an accurate script with good prose but little help deciding what matters, what to expect before each result, or how to state a conclusion; the 2,166-line dataset cell dominates the scroll.
- **Evidence:** source inspection; probe `guided_markers` (How to use / Roadmap / Glossary / audience / Check your reasoning / Troubleshooting / conclusion template / Infrastructure / What to notice all absent; the one "predict … before" match is the Section 9 heading "Predict before and after", not a learner prompt), `cellView_form_cells: []`.
- **Recommended correction:** add the GDL layer in the template following NOTEBOOK_SPEC's guided-mode requirements and its 2.2 reference notebook: audience and how-to-use, roadmap, task contract, glossary, a prediction before Sections 6, 7 and 8, "What to notice" after each principal stage, collapsible checkpoint answers, a Predict → Change one thing → Run → Observe → Explain activity built on DV2-M2's fix, troubleshooting, and a conclusion scaffold; title the install and carried cells `# @title Infrastructure: …` with `cellView: form`.
- **Acceptance check:** each of GDL1–GDL7 and GDL9–GDL14 maps to a named cell in a checklist added to `tutorials/README.md`, and cells 3, 5, 7 and 9 carry `cellView: form` with an Infrastructure title.
- **Spec:** GDL1–GDL7, GDL9–GDL14, UX8.

### Minor

#### DV2-m1 — The BYOD contract states a minimum the code rejects, and two failures surface as raw exceptions

- **Cell/section:** cell 1 Prerequisites ("a dataset needs 8..20,000 records"), cell 13 (`validate_dataset(part)` for every split), `samples.py` `split_dataset` and `load_byod_dataset`. Generator: `tools/notebook_template.py` line 115 and the cell-13 template.
- **Observed issue:** cell 13 validates each split with the dataset default `min_records = 8` and ≥ 2 classes. `split_dataset` sends 15 % per class to validation, so a BYOD set needs roughly 50+ images before the validation split reaches 8; a 40-image, 2-class set inside the stated contract is refused with "6 records; 8..20000 are required", which does not name the split. A `labels.csv` row whose file is missing raises `KeyError: 'ghost.jpg'`, and a non-image member raises `UnidentifiedImageError` naming only a `BytesIO`. BYOD has no location field, only the upload dialog.
- **Consequence:** users with a modest labelled set cannot tell why it was refused or how many images they need; two common packaging mistakes give errors without the corrective action.
- **Evidence:** direct (P4, cell-13 code path, no model): the six results quoted in §4.
- **Recommended correction:** state the effective minimum (per class and total) in the Prerequisites and the BYOD paragraph, or validate the validation split with `min_records=1` as `adapt` already does; name the split in the error; wrap the missing-file and undecodable-file cases with messages naming the `labels.csv` row and file; add a `BYOD_PATH` form field that bypasses the upload dialog (EXE2).
- **Acceptance check:** a 40-image, 2-class BYOD zip is either accepted or refused with a message that names the validation split and the minimum; a missing file and a text file named `.jpg` each produce a `ValueError` naming the row id and file name.
- **Spec:** DAT12, DAT19, UX10.

#### DV2-m2 — Template escapes leak into the data contract

- **Cell/section:** cell 1 Prerequisites, "Data contract" bullet (`tools/notebook_template.py` line 115).
- **Observed issue:** the bullet renders as `{{id, image, label}}` and ids matching `[A-Za-z0-9_.:-]{{1,64}}`; the string is not passed through `str.format`, so the doubled braces survive. The second is a different regular expression from the one the code enforces (`{1,64}`).
- **Consequence:** the one place that states the record schema and id rule shows a malformed schema and a wrong pattern.
- **Evidence:** source inspection; probe `double_brace_leak_in_markdown` lists both strings.
- **Recommended correction:** use single braces in the template string (or format it), regenerate.
- **Acceptance check:** the regenerated notebook's markdown contains no `{{`; the id pattern shown equals `_ID_RE.pattern` in `samples.py`.
- **Spec:** DAT12.

#### DV2-m3 — Section 9 describes a frozen-column computation the code does not perform

- **Cell/section:** cell 22 prose; cell 23 (`frozen_pipe`). Generator: `tools/notebook_template.py` line 345.
- **Observed issue:** the prose says the frozen predictions are "recomputed from the probe's stored head on the frozen features — the head and blocks of the frozen policy were saved before the unfreeze". Cell 23 instead loads a second base model and trains a new probe with the same seed and settings.
- **Consequence:** the learner is told about a stored-head mechanism that does not exist, and the cell's extra model load and probe training (the most expensive part of Section 9) is unexplained. The numbers are right.
- **Evidence:** source inspection; direct (P1b): the retrained probe's labels and probabilities on the six images equal the cell-17 probe's exactly (max absolute difference 0.0, CPU).
- **Recommended correction:** describe what the cell does (a fresh base and a re-trained probe with the same seed reproduce the frozen policy), or keep the cell-17 probe pipeline and reuse it, which also helps DV2-M2.
- **Acceptance check:** the Section 9 text matches the code path that produces the frozen column.
- **Spec:** UX2.

#### DV2-m4 — Measurement claims without an environment, and a stale README sentence

- **Cell/section:** cell 1 Prerequisites (per-image and per-epoch CPU times "the build record measured"); cell 0 ("about four minutes of model time"); cells 18 and 24 (the 1e-4 sweep "lost eight points", "one block at 1e-4 was not selected"); `README.md` Quick start ("There is no label, no score and no metric helper").
- **Observed issue:** the timings and the sweep results cite "the build record" without naming the runtime, date or revision, and the sweep appears in no release record (`docs/release-verification.md` records only the default settings). The README sentence contradicts the `classify` / `evaluate` / `classification_metrics` API documented a few lines above it.
- **Consequence:** the learner cannot tell what hardware the timings describe (this review's contended CPU took 1,102 s), and the optional experiments are framed by results nobody can trace.
- **Evidence:** source inspection; `grep` of the repository finds the 1e-4 sweep only in the notebook template and `MODEL_CARD.md`.
- **Recommended correction:** label timings with environment and revision (or as estimates), record the sweep as a row in `docs/release-verification.md` or drop the numbers, and rewrite the README sentence to say `embed` itself carries no label or score.
- **Acceptance check:** every measured time or experiment result in the notebook names its runtime and matches a row in `docs/release-verification.md`; the README sentence no longer denies the metric helpers.
- **Spec:** UX12, SRC3.

### Suggestions

- **DV2-S1** — Declare `notebook_spec` 2.2 instead of 2.0 once the guided layer lands.
- **DV2-S2** — In Section 8, point out that the selected policy's test log-loss (0.569) is worse than the probe's (0.498) although validation log-loss chose it on 24 images; it is a good illustration of selection noise.
- **DV2-S3** — Add a BYOD location field so executors and Jupyter users can supply a path without the upload dialog (EXE2).
- **DV2-S4** — Record per-stage wall times in `result.json`.

## 6. Readiness

**Needs revision.** Open Majors DV2-M1 to DV2-M3. Remaining gates after the fixes: a one-pass hosted Run all of the regenerated blob (RUN1/RUN10/REL2), the REL12 BYOD exercise recorded in `docs/release-verification.md` (one compatible archive through export and reload, one incompatible archive), and an executed check of the optional-experiment rerun (DV2-M2's acceptance check). The registry status `Release-grade` should return to `Candidate` until then.

## 7. Verified versus inferred

- **Verified by direct execution (CPU, install skipped, labelled above):** default path 11/11 with every number equal to the hosted record; the frozen column of Section 9 equals the cell-17 probe; five of six BYOD refusals and their messages, including the in-contract 40-image refusal.
- **Verified from documented evidence:** the restart on Kaggle pass 1 and the pass-2 numbers.
- **Inferred from source (not executed within the probe cap):** the contaminated rerun of cell 19, the expected parity failure with `TRAINABLE_BLOCKS = 1`, and the BYOD rerun on the sample-adapted backbone (DV2-M2); the Colab upload dialog behaviour; the basename-collision behaviour of `load_byod_dataset`.
- **Only Kurt can confirm:** whether the intended rerun model for experiments is "re-run from Section 3" (a documentation fix) or an `adapt` that always starts from the base (a code fix).
- **Most likely to be wrong:** DV2-M2's predicted parity failure — it follows from `save_artifact` saving only the latest run's blocks, but it was not executed; if a future `adapt` restores the base before training, only the stale-comparison half of the finding would remain.
