"""ML-3: entity-level macro F0.5, error analysis, and threshold selection."""
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
from data_io import read_id_lists, write_json
from blocking import PAIR_COLUMNS

ERROR_COLUMNS = ["source1_entity_id", "ground_truth", "predicted", "false_positives", "false_negatives"]


def entity_f05(tp, predicted_count, truth_count):
    if truth_count == 0:
        return float(predicted_count == 0)
    # Equivalent to beta=0.5 F-score, including the tp=0 case.
    return 1.25 * tp / (0.25 * truth_count + predicted_count)


def prediction_sets(pairs):
    result = defaultdict(set)
    for source, target in pairs[PAIR_COLUMNS].itertuples(index=False, name=None):
        result[source].add(target)
    return dict(result)


def evaluate_sets(predictions, truth):
    if not truth:
        raise ValueError("Evaluation requires at least one labeled Source 1 entity")
    scores, errors = [], []
    total_tp = total_fp = total_fn = 0
    for source, expected in truth.items():
        predicted = predictions.get(source, set())
        tp, fp, fn = len(predicted & expected), len(predicted - expected), len(expected - predicted)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        scores.append(entity_f05(tp, len(predicted), len(expected)))
        if predicted != expected:
            errors.append(dict(zip(ERROR_COLUMNS, [
                source, ",".join(sorted(expected)), ",".join(sorted(predicted)),
                ",".join(sorted(predicted - expected)), ",".join(sorted(expected - predicted))
            ])))
    report = {
        "macro_f05": float(np.mean(scores)), "evaluated_entities": len(truth),
        "singleton_entities": sum(not ids for ids in truth.values()),
        "true_positives": total_tp, "false_positives": total_fp, "false_negatives": total_fn,
        "micro_precision": total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0,
        "micro_recall": total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0,
    }
    return report, pd.DataFrame(errors, columns=ERROR_COLUMNS)


def tune_threshold(scored_pairs, truth):
    """Exact O(P log P) sweep; ties prefer the higher, more conservative threshold."""
    if not truth:
        raise ValueError("Threshold tuning requires ground truth")
    scores = scored_pairs["score"].to_numpy(dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Scores must be finite and in [0, 1]")
    if scored_pairs.duplicated(PAIR_COLUMNS).any():
        raise ValueError("Threshold tuning requires unique pairs")
    active = scored_pairs[scored_pairs.source1_entity_id.isin(truth)].sort_values("score", ascending=False)
    counts, positives = defaultdict(int), defaultdict(int)
    total = float(sum(not ids for ids in truth.values()))
    best_score, best_threshold = total / len(truth), float(np.nextafter(1.0, 2.0))
    curve = [{"threshold": best_threshold, "macro_f05": best_score}]
    for threshold, batch in active.groupby("score", sort=False):
        for source, target in batch[PAIR_COLUMNS].itertuples(index=False, name=None):
            n_true = len(truth[source])
            total -= entity_f05(positives[source], counts[source], n_true)
            counts[source] += 1
            positives[source] += int(target in truth[source])
            total += entity_f05(positives[source], counts[source], n_true)
        current = total / len(truth)
        curve.append({"threshold": float(threshold), "macro_f05": current})
        if current > best_score + 1e-12:
            best_score, best_threshold = current, float(threshold)
    return best_threshold, pd.DataFrame(curve)


def compute_macro_f05(pred_file, ground_truth_file, dump_error_tsv=None):
    predictions = read_id_lists(pred_file, "matched_entity_ids")
    truth = read_id_lists(ground_truth_file, "matched_entity_ids")
    if set(truth) - set(predictions):
        raise ValueError("Predictions must include every labeled Source 1 ID")
    report, errors = evaluate_sets(predictions, truth)
    if dump_error_tsv:
        errors.to_csv(dump_error_tsv, sep="\t", index=False)
    print(f"Macro F0.5: {report['macro_f05']:.6f} over {report['evaluated_entities']} entities")
    return report["macro_f05"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions")
    parser.add_argument("ground_truth")
    parser.add_argument("errors", nargs="?")
    parser.add_argument("--report")
    args = parser.parse_args()
    try:
        predictions = read_id_lists(args.predictions, "matched_entity_ids")
        truth = read_id_lists(args.ground_truth, "matched_entity_ids")
        if set(truth) - set(predictions):
            raise ValueError("Predictions must include every labeled Source 1 ID")
        report, errors = evaluate_sets(predictions, truth)
        if args.errors:
            errors.to_csv(args.errors, sep="\t", index=False)
        if args.report:
            write_json(args.report, report)
        print(f"Macro F0.5: {report['macro_f05']:.6f}")
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Evaluation failed: {exc}\n")


if __name__ == "__main__":
    main()
