"""Backend orchestration shared by the CLI and Python callers."""
from contextlib import contextmanager
import logging
from pathlib import Path
import platform
import time
import pandas as pd
import sklearn

from aggregate import aggregate_to_tsv_format
from blocking import PAIR_COLUMNS, blocking_recall, generate_candidate_pairs
from config import PipelineConfig
from data_io import load_sources, load_truth, write_json
from evaluate import evaluate_sets, prediction_sets
from features import compute_pairwise_features
from model import load_model, save_model, score_pairs
from normalize import normalize_dataset
from training import train_matcher
from validate import validate_outputs


@contextmanager
def timed_stage(logger, timings, name):
    logger.info("Starting %s", name)
    start = time.perf_counter()
    yield
    timings[name] = round(time.perf_counter() - start, 4)
    logger.info("Finished %s (%.3fs)", name, timings[name])


def execute_pipeline(s1_path, s2_path, s3_path, out_dir, *, mode="baseline", truth_path=None,
                     model_path=None, config=None, threshold=None, save_intermediates=False):
    if mode not in {"baseline", "train", "predict"}:
        raise ValueError("mode must be baseline, train, or predict")
    if mode == "train" and truth_path is None:
        raise ValueError("Training requires --truth")
    if mode == "predict" and model_path is None:
        raise ValueError("Prediction requires --model")
    if mode == "baseline" and model_path is not None:
        raise ValueError("--model requires train or predict mode")
    if threshold is not None and not 0 <= threshold <= 1:
        raise ValueError("--threshold must be in [0, 1]")
    if mode == "train" and threshold is not None:
        raise ValueError("Training tunes its threshold; omit --threshold")
    artifact = load_model(model_path) if mode == "predict" else None
    if config is None:
        config = PipelineConfig(**artifact["config"]) if artifact else PipelineConfig()
    if mode == "train" and model_path and Path(model_path).exists():
        raise ValueError("Model output already exists; choose a new model path")
    if mode == "train" and model_path and Path(model_path).suffix != ".joblib":
        raise ValueError("Model output must use the .joblib extension")
    output = Path(out_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output directory must be new or empty; use a separate directory for each experiment")
    output.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"entity_resolution.{id(output)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    file_handler = logging.FileHandler(output / "pipeline.log", encoding="utf-8")
    stream_handler = logging.StreamHandler()
    for handler in (file_handler, stream_handler):
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    timings, blocking_stats = {}, {}
    start = time.perf_counter()
    report = {"mode": mode, "config": config.to_dict(), "python_version": platform.python_version(),
              "sklearn_version": sklearn.__version__, "timings_seconds": timings}
    try:
        with timed_stage(logger, timings, "load"):
            source, targets, profiles = load_sources(s1_path, s2_path, s3_path)
            truth = load_truth(truth_path, source.entity_id, targets.entity_id) if truth_path else None
            if truth is not None and not truth:
                raise ValueError("Ground-truth file has no labeled entities")
            report["inputs"] = {name: {"path": str(Path(path).resolve()), "bytes": Path(path).stat().st_size}
                                for name, path in (("s1", s1_path), ("s2", s2_path), ("s3", s3_path))}
            if truth_path:
                report["ground_truth_path"] = str(Path(truth_path).resolve())
            report["data_profile"] = profiles
            if truth is not None:
                report["labels"] = {"entities": len(truth), "singletons": sum(not ids for ids in truth.values()),
                                    "entities_with_matches": sum(bool(ids) for ids in truth.values())}
        with timed_stage(logger, timings, "normalize"):
            source_norm, target_norm = normalize_dataset(source), normalize_dataset(targets)
            if save_intermediates:
                source_norm.to_csv(output / "normalized_source1.tsv", sep="\t", index=False)
                target_norm.to_csv(output / "normalized_targets.tsv", sep="\t", index=False)
        with timed_stage(logger, timings, "block"):
            pairs = generate_candidate_pairs(source_norm, target_norm, config.max_candidates, config.max_block_size, blocking_stats)
            if truth is not None:
                blocking_stats.update(blocking_recall(pairs, truth))
                retained = prediction_sets(pairs)
                missed = [(s, t) for s, ids in truth.items() for t in sorted(ids - retained.get(s, set()))]
                pd.DataFrame(missed, columns=PAIR_COLUMNS).to_csv(output / "blocking_misses.tsv", sep="\t", index=False)
            report["blocking"] = blocking_stats
            aggregate_to_tsv_format(source.entity_id, pairs, "candidate_entity_ids").to_csv(output / "candidate_pairs.tsv", sep="\t", index=False)
        with timed_stage(logger, timings, "features"):
            features = compute_pairwise_features(pairs, source_norm, target_norm)
            if save_intermediates:
                features.to_csv(output / "pair_features.tsv", sep="\t", index=False)
        if mode == "train":
            with timed_stage(logger, timings, "train"):
                artifact, training_report, diagnostics = train_matcher(features, truth, config)
                report["training"] = training_report
                for name, table in diagnostics.items():
                    table.to_csv(output / f"{name}.tsv", sep="\t", index=False)
                model_path = Path(model_path) if model_path else output / "model.joblib"
                save_model(model_path, artifact)
                logger.info("Saved model to %s", model_path)
        with timed_stage(logger, timings, "score"):
            decision_threshold = threshold if threshold is not None else artifact["threshold"] if artifact else config.baseline_threshold
            scored = score_pairs(features, artifact)
            scored["is_match"] = scored.score >= decision_threshold
            scored.to_csv(output / "pair_scores.tsv", sep="\t", index=False)
            matches = scored.loc[scored.is_match, PAIR_COLUMNS]
            aggregate_to_tsv_format(source.entity_id, matches, "matched_entity_ids").to_csv(output / "matching_results.tsv", sep="\t", index=False)
            report["threshold"] = decision_threshold
            report["matched_pairs"] = len(matches)
            if artifact:
                report["model_path"] = str(Path(model_path).resolve())
                report["model_config_overridden"] = config.to_dict() != artifact["config"]
        with timed_stage(logger, timings, "validate"):
            if not validate_outputs(output / "matching_results.tsv", output / "candidate_pairs.tsv", s1_path, s2_path, s3_path):
                raise ValueError("Output validation failed")
        if truth is not None:
            with timed_stage(logger, timings, "evaluate"):
                metrics, errors = evaluate_sets(prediction_sets(matches), truth)
                errors.to_csv(output / "errors.tsv", sep="\t", index=False)
                report["evaluation"] = metrics
                if artifact:
                    source_overlap = set(truth) & set(artifact["training_source_ids"])
                    target_overlap = set().union(*truth.values()) & set(artifact["training_true_target_ids"])
                    report["evaluation_training_overlap"] = {"source_entities": len(source_overlap), "true_targets": len(target_overlap)}
                    if source_overlap or target_overlap:
                        report["evaluation_note"] = "Evaluation overlaps training entities. This is not an independent test score; use training.cross_validation for OOF tuning diagnostics."
                        logger.warning(report["evaluation_note"])
                logger.info("Macro F0.5: %.6f; blocking recall: %s", metrics["macro_f05"], blocking_stats["blocking_recall"])
        report["total_seconds"] = round(time.perf_counter() - start, 4)
        report["status"] = "success"
        write_json(output / "run_report.json", report)
        logger.info("Completed successfully; artifacts: %s", output.resolve())
        return report
    except Exception as exc:
        logger.exception("Pipeline failed")
        write_json(output / "failure.json", {"status": "failed", "error": str(exc), "completed_stages": timings})
        raise
    finally:
        for handler in (file_handler, stream_handler):
            logger.removeHandler(handler)
            handler.close()
