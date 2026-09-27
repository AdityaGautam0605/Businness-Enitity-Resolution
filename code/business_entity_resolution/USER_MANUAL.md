# Business Entity Resolution — User Manual

Match each Source 1 business against Sources 2 and 3. The pipeline supports a
conservative baseline, supervised gradient-boosting training, saved-model
prediction, evaluation, debugging artifacts, and submission packaging.

## 1. Install

Use Python 3.11 (the verified version). Run all commands below from the repository
root, the directory containing `code/` and `task_split_roles.md`.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r code/business_entity_resolution/requirements.txt
```

Activation is optional: commands use the interpreter directly so PowerShell's
activation-script execution policy does not interfere.

Linux/macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r code/business_entity_resolution/requirements.txt
```

Replace `.\.venv\Scripts\python.exe` in subsequent commands with
`.venv/bin/python` on Linux/macOS. Shell commands are single lines for easy copying.
`requirements-tested.txt` pins the versions used during verification; install it
instead of `requirements.txt` to reproduce that dependency set. Model loading
requires the same scikit-learn version used during training.

## 2. Supply datasets

The repository does not contain the competition datasets or their download URLs.
Obtain the files from the organizers and extract them yourself. For example, if
they provided a local ZIP, PowerShell supports:

```powershell
Expand-Archive -LiteralPath "C:\path\to\provided-dataset.zip" -DestinationPath dataset
```

This is an example local path, not a download link. Inspect the extracted folder
and pass the actual paths to the CLI; the filenames below are conventions only.

```text
dataset/
  source1.tsv
  source2.tsv
  source3.tsv
  train_ground_truth.tsv
```

Files must be UTF-8 tab-separated text, with headers. A UTF-8 BOM is accepted.

| Source column | Contract |
|---|---|
| `entity_id` | Required, unique within each source, nonempty string |
| `business_name` | Recommended; missing values/column become empty strings |
| `business_address` | Recommended; missing values/column become empty strings |
| `country` | Recommended; available conflicting countries prevent candidate generation |
| `city`, `state`, `postal_code` | Optional extra similarity signals |

IDs are preserved as text (`001` and `NA` remain unchanged). IDs cannot contain
commas, tabs, newlines, or surrounding whitespace. Source 2 and Source 3 IDs must
be globally unique: the output format cannot distinguish two different target
records with the same ID. The pipeline rejects such collisions instead of
silently overwriting records. Rename IDs only consistently with the official
dataset/ground truth requirements. No source-prefix convention is assumed;
the validator checks exact membership in the supplied files.

Ground truth must have exactly these columns:

```text
source1_entity_id    matched_entity_ids
S1_001              S2_004,S3_018
S1_002
```

The gap above illustrates a tab, not spaces. For a singleton, include its Source 1
ID, a tab, and an empty second field. Target lists have no spaces around commas
and contain no duplicates. Partial labeling is supported: a missing Source 1 row
means **unlabeled**, while a present row with an empty list means **known singleton**.
For labeled entities, the match list must be complete; omitted matches otherwise
become false negative labels during training. Unknown source or target IDs fail
validation. Entirely empty source files must still have headers.

## 3. Try a synthetic demo without the competition data

```powershell
.\.venv\Scripts\python.exe code/business_entity_resolution/src/make_demo.py --out dataset/demo
.\.venv\Scripts\python.exe code/business_entity_resolution/src/run_pipeline.py --s1 dataset/demo/source1.tsv --s2 dataset/demo/source2.tsv --s3 dataset/demo/source3.tsv --truth dataset/demo/train_ground_truth.tsv --out output/demo-baseline --save-intermediates
```

The generated records include true matches, singletons, multiple matches, name
suffix variations, and similar businesses at different street numbers. They are
synthetic smoke-test data, not evidence of accuracy on the competition data.
The demo generator requires a new or empty destination.

## 4. Run the baseline

```powershell
.\.venv\Scripts\python.exe code/business_entity_resolution/src/run_pipeline.py --mode baseline --s1 dataset/source1.tsv --s2 dataset/source2.tsv --s3 dataset/source3.tsv --truth dataset/train_ground_truth.tsv --out output/baseline-01 --save-intermediates
```

Omit `--truth` if labels are unavailable. The default mode is `baseline`, preserving
the original `--s1 --s2 --s3 --out` invocation. The baseline needs no trained model.
Every output directory must be new or empty; use a distinct directory per
experiment to preserve prior results and avoid stale diagnostic files.

The baseline score is:

```text
clip(0.65 * name_token_sort
   + 0.35 * address_token_sort
   - 0.20 * street_number_conflict, 0, 1)
```

The default acceptance threshold is 0.86. Missing fields contribute zero
similarity, including when both fields are missing. Consequently this baseline
is conservative for records missing names or addresses. Pass `--threshold 0.90`
to override the acceptance threshold during baseline or prediction runs.

## 5. Train the supervised matcher (ML-3)

```powershell
.\.venv\Scripts\python.exe code/business_entity_resolution/src/run_pipeline.py --mode train --s1 dataset/source1.tsv --s2 dataset/source2.tsv --s3 dataset/source3.tsv --truth dataset/train_ground_truth.tsv --config code/business_entity_resolution/config.example.json --out output/train-01 --save-intermediates
```

This command prepares candidates and features, trains grouped cross-validation
models, tunes the threshold on out-of-fold predictions, fits the final model on
all labeled candidate pairs, and saves `output/train-01/model.joblib`.
An optional `--model path/to/new-model.joblib` changes the artifact destination;
existing model files are not overwritten.

Training requires positive and negative labeled candidate pairs and at least
two independent labeled entity groups. Source entities sharing a known true
target are grouped transitively so their positive examples cannot cross folds.
All pairs for a Source 1 entity stay together. Fold count is limited by the
available groups. A single-class training fold uses a constant classifier and is
flagged in the report; a fold with no training candidates fails with guidance.
Unlabeled entities are excluded from training, not treated as negative examples.

Threshold selection maximizes entity-level macro F0.5, includes singletons and
true matches lost during blocking, and breaks ties in favor of a higher threshold.
A saved threshold slightly greater than 1 represents rejecting every candidate
when that wins validation. F0.5 weights precision more strongly than recall; it
is not a fixed “two times the cost” penalty per false positive.

Read `run_report.json` → `training.cross_validation.macro_f05` for the out-of-fold
tuning result, and compare it with `training.baseline` and
`training.predict_empty_baseline`. A trained model is not automatically promoted
over the baseline: choose based on held-out results.

**Score interpretation:** the threshold was selected using the out-of-fold
predictions, so this is a tuning score, not an untouched test score. Training
mode also writes final-model predictions on the supplied records; their
`evaluation` score is a training-overlap score. For final accuracy measurement,
use a separate set of labeled businesses not used to train or tune. Keep known
linked businesses on the same side of that split. Prediction reports flag known
overlap with saved training source IDs or true target IDs; unknown duplicate
businesses cannot be detected by this check.

## 6. Predict with the saved model

```powershell
.\.venv\Scripts\python.exe code/business_entity_resolution/src/run_pipeline.py --mode predict --s1 dataset/source1.tsv --s2 dataset/source2.tsv --s3 dataset/source3.tsv --model output/train-01/model.joblib --out output/predict-01
```

Add `--truth path/to/heldout_truth.tsv` for integrated evaluation. Predict uses
the model's saved configuration and threshold by default. An explicit `--config`
replaces that configuration and is recorded in the report; omitted JSON fields
take program defaults. Changing blocking settings can change recall, so remeasure
when overriding them. Feature-version and scikit-learn-version checks reject
incompatible artifacts. Load only trusted `.joblib` files; they use pickle.

## 7. Understand outputs and debug by stage

| File | Purpose |
|---|---|
| `candidate_pairs.tsv` | One row per Source 1 ID, comma-separated `candidate_entity_ids` |
| `matching_results.tsv` | One row per Source 1 ID, comma-separated `matched_entity_ids` |
| `pair_scores.tsv` | Pair IDs, score, and `is_match`; available in every mode |
| `run_report.json` | Configuration, versions, input paths/sizes, data profile, timings, counts, metrics |
| `pipeline.log` | Stage progress and tracebacks |
| `failure.json` | Failure reason and completed stage timings, if a started run fails |
| `blocking_misses.tsv` | Known true pairs not shortlisted; requires ground truth |
| `errors.tsv` | Final-prediction false positives and false negatives; requires ground truth |
| `normalized_source1.tsv`, `normalized_targets.tsv` | Normalization inspection; `--save-intermediates` |
| `pair_features.tsv` | Pair IDs plus 16 features; `--save-intermediates` |
| `model.joblib` | Trained classifier, schema, threshold, configuration, and training IDs |
| `oof_scores.tsv`, `oof_errors.tsv` | Training validation scores/folds and mismatch details |
| `fold_assignments.tsv` | Every labeled Source 1 entity, including those with no candidates, and its fold/group |
| `threshold_curve.tsv` | Macro F0.5 at each distinct out-of-fold score threshold |
| `feature_summary.tsv` | Mean feature values for positive versus negative labeled pairs |

Outputs are deterministically sorted within each Source 1 match list; empty
lists remain empty TSV fields. There is no one-to-one assignment restriction:
multiple target records may represent the same business.

The report records missing fields, duplicate raw name/address combinations,
country counts, candidate caps, skipped dense buckets, and blocking recall.
Blocking recall is null when there are no positive truth pairs, not a claimed
100%. Very large intermediate files can be avoided by omitting
`--save-intermediates`; pair scores are always saved for debugging.

## 8. Iterate and test

1. Run the baseline and inspect the data profile and blocking recall first.
2. If recall is low, inspect `blocking_misses.tsv` with normalized records.
   Adjust `normalize.py`, `blocking.py`, or candidate limits, then rerun into a
   new directory. A scorer cannot recover a pair that blocking dropped.
3. Inspect positive/negative feature summaries and error examples. Add features
   in `features.py`, update its ordered schema/version, and retrain models.
4. Train and compare out-of-fold results to the fixed and predict-empty baselines.
5. Evaluate on untouched labeled entities before selecting a submission model.
6. Run the regression suite after code changes:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s code/business_entity_resolution/tests -v
```

The suite generates its own temporary datasets and checks input failures,
normalization, missing fields, blocking/ranking, score math, threshold selection
against brute force, singleton handling, partial labels, model roundtrips,
baseline/train/predict integration, CLI exit codes, and a packaged-code run from
a fresh directory. Expected failure tests print `[FAIL]`/tracebacks; the final
unittest summary determines whether tests passed.

Copy `config.example.json` to a separate JSON file for experiments. Available
settings are `max_candidates`, `max_block_size`, `baseline_threshold`, `cv_folds`,
`random_state`, `n_estimators`, `max_depth`, and `learning_rate`. Unknown keys and
invalid values fail early. Increasing candidate limits raises memory/time cost;
lowering them may lose true matches.

## 9. Evaluate, validate, and package independently

```powershell
.\.venv\Scripts\python.exe code/business_entity_resolution/src/evaluate.py output/predict-01/matching_results.tsv dataset/train_ground_truth.tsv output/predict-01/rechecked_errors.tsv --report output/predict-01/rechecked_metrics.json
.\.venv\Scripts\python.exe code/business_entity_resolution/src/validate.py output/predict-01/matching_results.tsv output/predict-01/candidate_pairs.tsv dataset/source1.tsv dataset/source2.tsv dataset/source3.tsv
.\.venv\Scripts\python.exe code/business_entity_resolution/src/package_submission.py --run output/predict-01 --s1 dataset/source1.tsv --s2 dataset/source2.tsv --s3 dataset/source3.tsv --out output/submission-01.zip
```

The standalone evaluator scores only labeled truth entities but requires a
prediction row for each of them. A header-only error dump is written when there
are no errors. The validator enforces schemas, unique rows/IDs, full Source 1
coverage, valid target membership, and matches being a subset of candidates.

Packaging revalidates outputs and includes the two submission TSVs at archive
root, the run report, code, tests, config, dependency files, documentation, and
the saved model when used. It excludes raw datasets and large debug tables.
Model files must still exist at their recorded paths when packaging. Review
the official submission rules for any additional naming or size constraints;
the repository does not include that specification. Existing ZIPs are not
overwritten. After extraction, run commands from the extracted archive root;
the bundled model is `model.joblib`.

## 10. Module ownership and extension points

| Owner | Modules | Contract |
|---|---|---|
| ML-1 | `data_io.py`, `normalize.py`, `blocking.py` | Source frames → normalized frames → unique pair IDs |
| ML-2 | `features.py`, baseline functions in `model.py` | Pair IDs + normalized frames → ordered numeric feature table |
| ML-3 | `training.py`, `model.py`, `evaluate.py`, `aggregate.py` | Labeled features → artifact; scored pairs → output rows and metrics |
| Backend | `config.py`, `pipeline.py`, `run_pipeline.py`, `validate.py`, `package_submission.py` | Configuration, orchestration, logs, validation, packaging |

The backend is a local CLI/Python orchestrator. No web server, database, AWS
service, credentials, or deployment is required. `pipeline.execute_pipeline`
is the reusable Python entry point when `src/` is on the module search path.
Stage functions can be called directly for focused debugging. Runs load source
records and candidate features into memory; full-data scalability still needs
measurement before choosing production limits.

Normalization currently supports Unicode text, accent folding, selected English
street abbreviations, common legal suffixes (including Pvt/Ltd), and US/UK/IN
country aliases. Leading street numbers are recognized; address parsing is not
internationally complete. `St` is interpreted as Street, so Saint names need
dataset-specific refinement. Soundex only applies to Latin letters; Unicode
names can still use token/address passes. Blocking unions name-prefix, name-token,
phonetic, exact-address, and street-number/street-prefix keys. Known unequal
countries are excluded; missing country permits compatible key queries across
countries. Oversized buckets are skipped per pass and candidates are ranked
before capping. These recall tradeoffs must be checked on real labeled data.

Passing tests establishes the covered behavior; it cannot establish that all
bugs are absent or that the matching rules are accurate on unseen data.
