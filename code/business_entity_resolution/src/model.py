"""ML-2 baseline and ML-3 persisted model inference."""
from pathlib import Path
import joblib
import numpy as np
import sklearn
from blocking import PAIR_COLUMNS
from config import PipelineConfig
from features import FEATURE_COLUMNS, FEATURE_VERSION

ARTIFACT_VERSION = 1


def baseline_scores(features):
    if features.empty:
        return np.array([], dtype=float)
    # Do not renormalize missing fields into perfect agreement.
    score = 0.65 * features["token_sort"] + 0.35 * features["address_token_sort"]
    score = score - 0.20 * features["street_number_conflict"]
    return score.clip(0, 1).to_numpy(dtype=float)


def positive_probability(estimator, features):
    if features.empty:
        return np.array([], dtype=float)
    classes = list(estimator.classes_)
    if 1 not in classes:
        return np.zeros(len(features))
    return estimator.predict_proba(features[FEATURE_COLUMNS])[:, classes.index(1)]


def score_pairs(features, artifact=None):
    result = features[PAIR_COLUMNS].copy()
    result["score"] = baseline_scores(features) if artifact is None else positive_probability(artifact["estimator"], features)
    return result


def predict_matches(feature_df, threshold=0.86, artifact=None):
    if not np.isfinite(threshold) or not 0 <= threshold <= np.nextafter(1.0, 2.0):
        raise ValueError("Threshold must be in [0, 1] or the saved reject-all sentinel")
    scored = score_pairs(feature_df, artifact)
    return scored.loc[scored.score >= threshold, PAIR_COLUMNS].reset_index(drop=True)


def save_model(path, artifact):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_model(path):
    # joblib/pickle files are executable: load only artifacts you trust.
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or artifact.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError("Unsupported model artifact; retrain with this codebase")
    if artifact.get("feature_version") != FEATURE_VERSION or artifact.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError("Model feature schema differs from this codebase; retrain")
    if artifact.get("sklearn_version") != sklearn.__version__:
        raise ValueError("Model scikit-learn version differs from installed version; use its environment or retrain")
    PipelineConfig(**artifact["config"])
    threshold = artifact.get("threshold")
    if not isinstance(threshold, (int, float)) or not np.isfinite(threshold) or not 0 <= threshold <= np.nextafter(1.0, 2.0):
        raise ValueError("Invalid saved model threshold")
    if not hasattr(artifact.get("estimator"), "predict_proba"):
        raise ValueError("Artifact is missing a fitted classifier")
    return artifact
