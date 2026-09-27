# Business Entity Resolution — Implementation Notes

## Objective

Resolve Source 1 businesses to zero or more matching records in Sources 2 and 3.
Return one output row per Source 1 entity, including known or predicted singletons.

## Implemented approach

1. Read UTF-8 TSVs with strict schema/ID checks; profile missing fields and duplicates.
2. Normalize names, legal suffixes, selected address abbreviations, and country aliases.
3. Union prefix, token, phonetic, exact-address, and street-number blocking passes.
   Rank candidates deterministically before applying configured limits.
4. Compute 16 name/address/geographic similarity and missing-evidence features.
5. Use the weighted baseline or train a gradient-boosted classifier. Training
   groups known linked source entities, produces out-of-fold predictions, and
   selects a threshold maximizing macro F0.5. Fit and save the final model.
6. Aggregate match lists, validate source/target coverage and candidate membership,
   and optionally evaluate against ground truth and export error tables.

## Reproduction and ownership

See [USER_MANUAL.md](code/business_entity_resolution/USER_MANUAL.md) for commands,
schemas, module boundaries, configuration, diagnostics, and packaging. ML-1 owns
input/normalization/blocking, ML-2 owns features and baseline, ML-3 owns training,
inference/evaluation/aggregation, and the backend orchestrates the stages.

## Evaluation methodology

The evaluator averages per-source F0.5 scores. A correctly empty singleton scores
1; any false singleton match scores 0. Out-of-fold threshold tuning accounts for
true matches omitted by blocking and keeps all labeled entities in the score.
The report includes baseline and predict-empty comparisons. The threshold-selected
OOF score is a tuning result; final performance requires untouched labeled entities.
Final-model evaluation on training records is explicitly marked as overlapping.

## Verification and results

Automated tests use generated fixtures to cover edge cases and complete CLI flows,
including persisted-model predictions and execution of a packaged copy. Exact
verification results are recorded in `code/business_entity_resolution/VERIFICATION.md`.
No official dataset, measured competition score, or full-scale benchmark is
available in this repository. Do not substitute synthetic scores for those results.

## Limitations

Blocking caps and skipped dense buckets can lose true matches. Address/country
normalization needs region-specific tuning. Missing fields reduce baseline evidence.
Model selection may require more labels; a trained classifier is not guaranteed to
beat the baseline. Records/features are processed in memory. Official submission
prefix/naming rules were not supplied; validation uses exact input membership.
There is no AWS deployment or HTTP API in this codebase.

Before final submission, record the actual dataset sizes, blocking recall, untouched
validation score, selected model/configuration, full-data runtime, and any official
submission constraints in this document.
