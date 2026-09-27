"""ML-3: labeled pair training with entity-grouped out-of-fold calibration."""
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from blocking import PAIR_COLUMNS, blocking_recall
from evaluate import evaluate_sets, prediction_sets, tune_threshold
from features import FEATURE_COLUMNS, FEATURE_VERSION
from model import ARTIFACT_VERSION, baseline_scores, positive_probability


def truth_groups(truth):
    """Source entities sharing a known true target stay in the same fold."""
    parent = {source: source for source in truth}
    def root(source):
        while parent[source] != source:
            parent[source] = parent[parent[source]]
            source = parent[source]
        return source
    owner = {}
    for source, targets in truth.items():
        for target in sorted(targets):
            if target in owner:
                parent[root(source)] = root(owner[target])
            else:
                owner[target] = source
    return {source: root(source) for source in truth}


def fit_classifier(features, labels, config):
    if len(set(labels)) == 1:
        estimator = DummyClassifier(strategy="constant", constant=int(labels[0]))
    else:
        estimator = GradientBoostingClassifier(
            n_estimators=config.n_estimators, learning_rate=config.learning_rate,
            max_depth=config.max_depth, random_state=config.random_state
        )
    estimator.fit(features[FEATURE_COLUMNS], labels)
    return estimator


def train_matcher(features, truth, config):
    labeled = features[features.source1_entity_id.isin(truth)].reset_index(drop=True).copy()
    if labeled.empty:
        raise ValueError("No labeled candidate pairs. Inspect blocking recall and input IDs.")
    labels = np.array([int(target in truth[source]) for source, target in labeled[PAIR_COLUMNS].itertuples(index=False, name=None)])
    if len(set(labels)) < 2:
        raise ValueError("Training needs both positive and negative candidate pairs; add labeled matches and nonmatches.")
    group_map = truth_groups(truth)
    entity_ids = np.array(sorted(truth))
    groups = np.array([group_map[source] for source in entity_ids])
    folds = min(config.cv_folds, len(set(groups)))
    if folds < 2:
        raise ValueError("Training needs at least two independent labeled entity groups")
    oof = np.full(len(labeled), np.nan)
    fold_ids = np.full(len(labeled), -1, dtype=int)
    fold_reports, entity_folds = [], {}
    for fold, (train_indices, valid_indices) in enumerate(GroupKFold(n_splits=folds).split(entity_ids, groups=groups)):
        train_ids, valid_ids = set(entity_ids[train_indices]), set(entity_ids[valid_indices])
        train_mask = labeled.source1_entity_id.isin(train_ids).to_numpy()
        valid_mask = labeled.source1_entity_id.isin(valid_ids).to_numpy()
        entity_folds.update({source: fold for source in valid_ids})
        if not train_mask.any():
            raise ValueError("A CV fold has no training candidates; use more labeled entities or fewer folds")
        estimator = fit_classifier(labeled.loc[train_mask], labels[train_mask], config)
        oof[valid_mask] = positive_probability(estimator, labeled.loc[valid_mask])
        fold_ids[valid_mask] = fold
        fold_reports.append({"fold": fold, "train_entities": len(train_ids), "validation_entities": len(valid_ids), "train_pairs": int(train_mask.sum()), "validation_pairs": int(valid_mask.sum()), "single_class_training": len(set(labels[train_mask])) == 1})
    if not np.isfinite(oof).all():
        raise RuntimeError("Cross-validation did not score every labeled candidate")
    scored = labeled[PAIR_COLUMNS].copy()
    scored["score"], scored["label"], scored["fold"] = oof, labels, fold_ids
    threshold, curve = tune_threshold(scored, truth)
    matches = scored.loc[scored.score >= threshold, PAIR_COLUMNS]
    cv_report, errors = evaluate_sets(prediction_sets(matches), truth)
    baseline = labeled[PAIR_COLUMNS].copy()
    baseline["score"] = baseline_scores(labeled)
    baseline_report, _ = evaluate_sets(prediction_sets(baseline.loc[baseline.score >= config.baseline_threshold]), truth)
    empty_report, _ = evaluate_sets({}, truth)
    estimator = fit_classifier(labeled, labels, config)
    feature_summary = labeled[FEATURE_COLUMNS].assign(label=labels).groupby("label").mean().reset_index()
    artifact = {
        "artifact_version": ARTIFACT_VERSION, "feature_version": FEATURE_VERSION,
        "feature_columns": FEATURE_COLUMNS, "sklearn_version": sklearn.__version__,
        "config": config.to_dict(), "threshold": threshold, "estimator": estimator,
        "training_source_ids": sorted(truth),
        "training_true_target_ids": sorted(set().union(*truth.values())),
    }
    report = {
        "threshold": threshold, "cross_validation": cv_report, "baseline": baseline_report,
        "predict_empty_baseline": empty_report, "blocking": blocking_recall(labeled, truth),
        "labeled_pairs": len(labeled), "positive_pairs": int(labels.sum()),
        "negative_pairs": int((labels == 0).sum()), "folds": fold_reports,
        "feature_importance": dict(zip(FEATURE_COLUMNS, map(float, estimator.feature_importances_))),
        "score_note": "Threshold selected on grouped out-of-fold predictions; this is a tuning score, not an independent test score.",
    }
    assignments = pd.DataFrame([{"source1_entity_id": s, "fold": entity_folds[s], "group": group_map[s]} for s in sorted(truth)])
    return artifact, report, {"oof_scores": scored, "oof_errors": errors, "threshold_curve": curve, "feature_summary": feature_summary, "fold_assignments": assignments}
