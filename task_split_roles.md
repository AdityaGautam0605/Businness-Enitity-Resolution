# Business Entity Resolution — Team Task Split (36 Hours, 4 People)

3 ML engineers + 1 Backend engineer. Roles are split by pipeline stage so
each person owns a clear, independently-testable piece, with explicit
handoff points so nobody blocks anybody for long.

---

## ML-1 — Data & Blocking Lead

**Mission:** Turn raw, messy records into a clean candidate set. This role
determines the pipeline's **recall ceiling** — nothing downstream can
recover a true match that never gets shortlisted here, so it's the most
consequential early role.

**Responsibilities:**
- Explore all 3 source files and `train_ground_truth.tsv`: record counts,
  match rate, how many entities are singletons, and concrete examples of
  how the same business's name/address differ across sources (typos,
  abbreviations, missing fields, reordered tokens).
- Build the **normalization layer**: lowercasing, punctuation stripping,
  abbreviation expansion (St→Street, Inc→Incorporated, etc.), legal-suffix
  removal, whitespace cleanup, street-number extraction.
- Design the **blocking key(s)**: the cheap key used to bucket records so
  only same-bucket pairs get compared. Start simple (name prefix + street
  prefix), then iterate — try a phonetic key (Soundex/Metaphone), an
  address-first key, or a hybrid, and union the results if one key alone
  misses too many true matches.
- Generate `candidate_pairs.tsv` and continuously measure **blocking
  recall**: what fraction of ground-truth matches survive blocking. This
  number is the ceiling on the final score — push it as high as possible
  without exploding candidate volume.
- Flag data-quality issues early (encoding problems, empty fields, obvious
  duplicates) so the rest of the team isn't surprised later.

**Deliverable:** a candidate-pair generator with a measured, reported
recall ceiling, plus a normalization module the rest of the team codes
against.

**Skills leaned on:** string cleaning/regex, basic data profiling, a feel
for name/address variation patterns.

---

## ML-2 — Feature Engineering & Matching (Baseline) Lead

**Mission:** Turn each candidate pair into a set of signals that separate
true matches from look-alikes, and ship a working baseline matcher fast.

**Responsibilities:**
- Build **pairwise similarity features** on the normalized name/address
  fields: string similarity (Jaro-Winkler, Levenshtein/edit-distance ratio,
  token sort/set ratio), token-overlap measures (Jaccard on name tokens,
  address tokens), and address-specific checks (street-number match,
  city/state match).
- Sanity-check every feature against real true/false pairs pulled from
  `train_ground_truth.tsv` before trusting it — a feature that doesn't
  separate matches from look-alikes on known examples won't help the model.
- Ship a **threshold-based baseline matcher** early (combined similarity
  score ≥ threshold → match). This becomes the team's safety-net submission
  if the classifier isn't ready in time.
- Do error analysis together with ML-3: pull false positives/negatives from
  the model, look for patterns (e.g., regional abbreviations not expanded,
  address reordering not handled), and add or fix features accordingly.
- Handle region-specific patterns called out in the brief (naming and
  address conventions that vary by area).

**Deliverable:** a feature table generator + a working threshold baseline
that produces a valid submission on its own.

**Skills leaned on:** feature engineering, string-similarity metrics, error
analysis.

---

## ML-3 — Modeling, Tuning & Evaluation Lead

**Mission:** Build the scoring harness first so the whole team can measure
progress, then own the trained model and its calibration.

**Responsibilities:**
- Implement the **local macro F0.5 evaluator** *first*, before any model
  exists — this lets everyone score any candidate submission against
  `train_ground_truth.tsv` immediately, including a trivial
  "predict-empty-for-everyone" baseline to get a real number on the board
  from hour 2 onward.
- Train the **classifier** on labeled candidate pairs (positive = in
  ground truth, negative = candidate but not a true match) — gradient
  boosting (XGBoost/LightGBM/sklearn's GradientBoosting) is a strong
  default. Cross-validate to avoid overfitting to the small train set.
- **Calibrate the decision threshold with precision bias**: a false merge
  costs ~2x as much as a miss under F0.5, so when the model is unsure, the
  correct call is to *not* merge. Tune specifically for this asymmetry,
  not for raw accuracy.
- Verify **singleton handling**: entities with no true match must get an
  empty prediction (scores 1.0); any predicted match on a true singleton
  scores 0. This is an easy, high-leverage thing to get exactly right.
- Own the **aggregation step** — collapsing pairwise match/no-match
  decisions into one row per Source 1 entity with a comma-separated match
  list.
- Run the final local evaluation and report the score the team is actually
  optimizing against throughout.

**Deliverable:** the F0.5 scoring script (used by everyone), the trained
matcher, and the aggregation logic that produces the final prediction file.

**Skills leaned on:** classical ML (gradient boosting), threshold/precision-
recall tradeoff reasoning, careful metric implementation.

---

## BE-1 — Pipeline, Infra & Submission Lead

**Mission:** Make sure the three ML pieces actually connect into one
runnable pipeline, run fast enough, and produce a submission that passes
every formatting rule — every hour.

**Responsibilities:**
- Set up the repo/project skeleton, shared data folder, dependency file,
  and branch/merge workflow so the three ML engineers can work in parallel
  without stepping on each other.
- Build the **end-to-end orchestrator** (a CLI/script that runs
  normalization → blocking → features → matching → aggregation →
  evaluation in one command) and keep it working as each ML piece lands —
  wire in stubs early so integration isn't a big-bang event at the end.
- Own **performance**: profile and vectorize/batch any slow steps once
  the pipeline runs on the full-size test set, not just small samples.
- Build the **submission validator**: schema checks, duplicate-id checks,
  correct id-prefix checks, coverage checks (every Source 1 id present
  exactly once).
- Build an **error-dump tool** (mismatches between predictions and ground
  truth) so ML-1/ML-2/ML-3 can quickly see what's going wrong without
  writing their own debugging scripts.
- Own **final packaging**: assemble `matching_results.tsv`,
  `candidate_pairs.tsv`, the runnable code, and the filled-in
  `Documentation_template.md` into the submission `.zip`, and do a clean
  "fresh checkout" test run before the deadline to catch missing
  dependencies or hardcoded paths.
- Track the clock — flag when a milestone is at risk so the team can cut
  scope early instead of scrambling at hour 34.

**Deliverable:** a working end-to-end pipeline runnable in one command, a
validator, and the final, verified submission package.

**Skills leaned on:** software engineering/tooling, performance profiling,
attention to spec compliance and edge cases.

---

## How the roles depend on each other

```
ML-1 (normalize + block)
        │  candidate_pairs
        ▼
ML-2 (features + baseline matcher) ──┐
        │  feature table             │ threshold baseline
        ▼                            │ (safety net, ships early)
ML-3 (classifier + aggregate + eval) │
        │  matching_results.tsv      │
        ▼                            ▼
BE-1 (orchestrates everything end-to-end, validates, packages)
```

ML-3's evaluator should exist before anyone's model does — it's the
measuring stick for every other decision. BE-1's orchestrator should accept
stubbed-out stages from hour 2 onward so real modules can be swapped in
one at a time instead of integrated all at once near the deadline.

---

## Milestone checkpoints (all 4 people)

| Hour | Checkpoint |
|---|---|
| 2 | Data explored, schema agreed, evaluator scoring a trivial baseline |
| 8 | Every module runs individually on sample data |
| 16 | **First full end-to-end run** — baseline F0.5 recorded (your safety net) |
| 24 | Improved model beats baseline; remaining priorities frozen |
| 30 | Edge cases handled, validator passing, docs drafted |
| 34 | Final package validated and ready — no new features after this |
| 36 | Submitted, buffer used or not |
