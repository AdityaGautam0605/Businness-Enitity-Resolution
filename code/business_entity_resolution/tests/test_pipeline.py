"""Regression and integration tests use generated data; no private dataset required."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from aggregate import aggregate_to_tsv_format
from blocking import PAIR_COLUMNS, blocking_recall, generate_candidate_pairs, soundex
from config import PipelineConfig
from data_io import load_sources, load_truth, read_id_lists, read_source
from evaluate import compute_macro_f05, entity_f05, evaluate_sets, prediction_sets, tune_threshold
from features import FEATURE_COLUMNS, compute_pairwise_features
from make_demo import make_demo
from model import baseline_scores, load_model, predict_matches, save_model, score_pairs
from normalize import clean_address, clean_name, clean_text, normalize_dataset
from pipeline import execute_pipeline
from package_submission import package_submission
from training import train_matcher, truth_groups
from validate import validate_outputs


def frame(rows):
    return normalize_dataset(pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address", "country"]))


class FilesTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, content):
        path = self.root / name
        path.write_text(content, encoding="utf-8")
        return path


class InputTests(FilesTestCase):
    def test_literal_na_and_leading_zeros(self):
        path = self.write("s.tsv", "entity_id\tbusiness_name\n001\tNA\nNA\tNull\n")
        df = read_source(path)
        self.assertEqual(df.entity_id.tolist(), ["001", "NA"])
        self.assertEqual(df.business_name.tolist(), ["NA", "Null"])
        self.assertTrue(df.business_address.eq("").all())

    def test_bad_ids_rejected(self):
        for rows in ("x\nx\n", '""\n', " x\n", "x,y\n"):
            with self.subTest(rows=rows):
                path = self.write("s.tsv", "entity_id\n" + rows)
                with self.assertRaises(ValueError):
                    read_source(path)

    def test_missing_id_column(self):
        with self.assertRaisesRegex(ValueError, "entity_id"):
            read_source(self.write("s.tsv", "name\nAcme\n"))

    def test_ambiguous_tsv_schema_rejected(self):
        for content in ("entity_id\tentity_id\na\tb\n", "entity_id\tbusiness_name\na\tb\textra\n", "entity_id\nb\n\n"):
            with self.subTest(content=content), self.assertRaises(ValueError):
                read_source(self.write("bad.tsv", content))

    def test_overlapping_target_ids(self):
        a = self.write("a.tsv", "entity_id\nx\n")
        with self.assertRaisesRegex(ValueError, "globally unique"):
            load_sources(a, a, a)

    def test_truth_unknown_ids(self):
        path = self.write("truth.tsv", "source1_entity_id\tmatched_entity_ids\ns\tt\n")
        with self.assertRaises(ValueError):
            load_truth(path, ["s"], ["other"])
        with self.assertRaises(ValueError):
            load_truth(path, ["other"], ["t"])

    def test_truth_partial_and_singleton(self):
        path = self.write("truth.tsv", "source1_entity_id\tmatched_entity_ids\ns\t\n")
        self.assertEqual(load_truth(path, ["s", "unlabeled"], []), {"s": set()})

    def test_malformed_lists_rejected(self):
        for values in ("a,a", "a,", "a, b", ",a"):
            with self.subTest(values=values):
                path = self.write("truth.tsv", "source1_entity_id\tmatched_entity_ids\ns\t" + values + "\n")
                with self.assertRaises(ValueError):
                    read_id_lists(path, "matched_entity_ids")

    def test_duplicate_truth_rows_rejected(self):
        path = self.write("truth.tsv", "source1_entity_id\tmatched_entity_ids\ns\ta\ns\tb\n")
        with self.assertRaises(ValueError):
            read_id_lists(path, "matched_entity_ids")

    def test_config_rejects_bad_values(self):
        for kwargs in ({"max_candidates": 0}, {"max_candidates": True}, {"cv_folds": 1}, {"learning_rate": float("nan")}, {"baseline_threshold": 2}, {"random_state": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PipelineConfig(**kwargs)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            PipelineConfig.load(self.write("config.json", '{"typo": 5}'))


class NormalizationTests(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(clean_name("The Café & Sons Pvt. Ltd."), "cafe and sons")
        self.assertEqual(clean_address("12 MAIN Rd., Apt 4"), "12 main road apartment 4")
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text("東京"), "東京")

    def test_country_and_number(self):
        df = frame([("s", "Acme", "12A Main St", "USA"), ("x", "Other", "Main St 90210", "US")])
        self.assertEqual(df.norm_country.tolist(), ["united states"] * 2)
        self.assertEqual(df.street_number.tolist(), ["12a", ""])

    def test_does_not_mutate_input(self):
        source = pd.DataFrame({"entity_id": ["s"], "business_name": ["ACME"]})
        before = source.copy(deep=True)
        normalize_dataset(source)
        pd.testing.assert_frame_equal(source, before)

    def test_soundex(self):
        self.assertEqual(soundex("Robert"), soundex("Rupert"))
        self.assertEqual(soundex("Ashcraft"), "a261")
        self.assertEqual(soundex("東京"), "")


class BlockingFeatureTests(unittest.TestCase):
    def test_address_pass_recovers_name_typo(self):
        s = frame([("s", "Zebra", "12 Oak Rd", "IN")])
        t = frame([("t", "Xebra", "12 Oak Road", "India")])
        self.assertEqual(len(generate_candidate_pairs(s, t)), 1)

    def test_country_missing_and_conflicting(self):
        s = frame([("s", "Acme", "", "IN")])
        t = frame([("missing", "Acme", "", ""), ("wrong", "Acme", "", "US")])
        self.assertEqual(generate_candidate_pairs(s, t).target_entity_id.tolist(), ["missing"])
        s["norm_country"] = ""
        self.assertEqual(len(generate_candidate_pairs(s, t)), 2)

    def test_best_candidate_survives_cap_and_shuffle(self):
        s = frame([("s", "Acme Steel", "10 Main Road", "IN")])
        t = frame([("bad", "Acme Tires", "500 Main Road", "IN"), ("good", "Acme Steel", "10 Main Road", "IN")])
        stats = {}
        a = generate_candidate_pairs(s, t, 1, diagnostics=stats)
        b = generate_candidate_pairs(s, t.iloc[::-1], 1)
        self.assertEqual(a.target_entity_id.tolist(), ["good"])
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(stats["sources_capped"], 1)

    def test_dense_name_block_still_uses_specific_address(self):
        s = frame([("s", "Acme", "1 Oak Road", "IN")])
        t = frame([("a", "Acme", "1 Oak Road", "IN"), ("b", "Acme", "2 Pine Road", "IN")])
        stats = {}
        pairs = generate_candidate_pairs(s, t, max_block_size=1, diagnostics=stats)
        self.assertEqual(pairs.target_entity_id.tolist(), ["a"])
        self.assertGreater(stats["oversized_bucket_queries_skipped"], 0)

    def test_missing_values_do_not_match(self):
        s, t = frame([("s", "", "", "")]), frame([("t", "", "", "")])
        self.assertTrue(generate_candidate_pairs(s, t).empty)
        pairs = pd.DataFrame([("s", "t")], columns=PAIR_COLUMNS)
        features = compute_pairwise_features(pairs, s, t)
        self.assertEqual(float(features[FEATURE_COLUMNS].sum(axis=1).iloc[0]), 0.0)
        self.assertEqual(baseline_scores(features).tolist(), [0.0])

    def test_street_conflict_penalizes_baseline(self):
        s = frame([("s", "Acme", "10 Main Rd", "IN")])
        t = frame([("a", "Acme", "10 Main Road", "IN"), ("b", "Acme", "11 Main Road", "IN")])
        features = compute_pairwise_features(pd.DataFrame([("s", "a"), ("s", "b")], columns=PAIR_COLUMNS), s, t)
        self.assertEqual(predict_matches(features).target_entity_id.tolist(), ["a"])
        self.assertTrue(np.isfinite(features[FEATURE_COLUMNS].to_numpy()).all())

    def test_feature_unknown_ids_fail(self):
        with self.assertRaisesRegex(ValueError, "unknown"):
            compute_pairwise_features(pd.DataFrame([("s", "bad")], columns=PAIR_COLUMNS), frame([("s", "a", "", "")]), frame([("t", "a", "", "")]))

    def test_empty_feature_contract(self):
        features = compute_pairwise_features(pd.DataFrame(columns=PAIR_COLUMNS), frame([]), frame([]))
        self.assertEqual(list(features), PAIR_COLUMNS + FEATURE_COLUMNS)
        self.assertTrue(predict_matches(features).empty)

    def test_recall_counts_missed_entities(self):
        pairs = pd.DataFrame([("s", "t")], columns=PAIR_COLUMNS)
        self.assertEqual(blocking_recall(pairs, {"s": {"t", "u"}, "x": {"v"}})["blocking_recall"], 1 / 3)
        self.assertIsNone(blocking_recall(pairs, {"s": set()})["blocking_recall"])


class EvaluationTests(FilesTestCase):
    def test_f05_hand_calculation_and_singletons(self):
        self.assertEqual(entity_f05(0, 0, 0), 1)
        self.assertEqual(entity_f05(0, 1, 0), 0)
        self.assertAlmostEqual(entity_f05(1, 2, 1), 5 / 9)
        self.assertAlmostEqual(entity_f05(1, 1, 2), 5 / 6)
        report, errors = evaluate_sets({"a": {"t", "wrong"}}, {"a": {"t"}, "b": set()})
        self.assertAlmostEqual(report["macro_f05"], (5 / 9 + 1) / 2)
        self.assertEqual(errors.false_positives.tolist(), ["wrong"])

    def test_threshold_sweep_matches_brute_force(self):
        rng = np.random.default_rng(21)
        for _ in range(15):
            truth = {f"s{i}": ({f"t{i}"} if i % 2 else set()) for i in range(6)}
            rows = [(s, f"t{j}", float(rng.choice([0.1, 0.4, 0.9, 1.0]))) for s in truth for j in range(6)]
            scored = pd.DataFrame(rows, columns=PAIR_COLUMNS + ["score"])
            threshold, curve = tune_threshold(scored, truth)
            best_score, best_threshold = -1, None
            for candidate in sorted(set(scored.score) | {float(np.nextafter(1.0, 2.0))}, reverse=True):
                metrics, _ = evaluate_sets(prediction_sets(scored.loc[scored.score >= candidate]), truth)
                if metrics["macro_f05"] > best_score + 1e-12:
                    best_score, best_threshold = metrics["macro_f05"], candidate
            self.assertEqual(threshold, best_threshold)
            self.assertAlmostEqual(curve.macro_f05.max(), best_score)

    def test_reject_all_including_score_one(self):
        scored = pd.DataFrame([("s", "t", 1.0)], columns=PAIR_COLUMNS + ["score"])
        threshold, _ = tune_threshold(scored, {"s": set()})
        self.assertGreater(threshold, 1.0)

    def test_threshold_empty_and_bad_scores(self):
        empty = pd.DataFrame(columns=PAIR_COLUMNS + ["score"])
        threshold, _ = tune_threshold(empty, {"s": {"missed"}, "singleton": set()})
        self.assertGreater(threshold, 1)
        for score in (float("nan"), -0.1, 1.1):
            with self.assertRaises(ValueError):
                tune_threshold(pd.DataFrame([("s", "t", score)], columns=PAIR_COLUMNS + ["score"]), {"s": {"t"}})

    def test_zero_errors_replaces_stale_dump(self):
        pred = self.write("pred.tsv", "source1_entity_id\tmatched_entity_ids\ns\tt\n")
        errors = self.write("errors.tsv", "stale data")
        self.assertEqual(compute_macro_f05(pred, pred, errors), 1)
        self.assertTrue(pd.read_csv(errors, sep="\t").empty)

    def test_evaluation_requires_coverage(self):
        pred = self.write("pred.tsv", "source1_entity_id\tmatched_entity_ids\ns\t\n")
        truth = self.write("truth.tsv", "source1_entity_id\tmatched_entity_ids\nother\t\n")
        with self.assertRaises(ValueError):
            compute_macro_f05(pred, truth)

    def test_aggregate_sorted_unique_and_empty(self):
        pairs = pd.DataFrame([("s", "b"), ("s", "a"), ("s", "b")], columns=PAIR_COLUMNS)
        result = aggregate_to_tsv_format(["s", "empty"], pairs, "matched_entity_ids")
        self.assertEqual(result.matched_entity_ids.tolist(), ["a,b", ""])


class TrainingIntegrationTests(FilesTestCase):
    def setUp(self):
        super().setUp()
        self.data = make_demo(self.root / "data", count=12)
        self.paths = [self.data / f"source{i}.tsv" for i in (1, 2, 3)]
        self.truth_path = self.data / "train_ground_truth.tsv"
        self.config = PipelineConfig(n_estimators=15, cv_folds=3)

    def prepare(self):
        s, t, _ = load_sources(*self.paths)
        truth = load_truth(self.truth_path, s.entity_id, t.entity_id)
        sn, tn = normalize_dataset(s), normalize_dataset(t)
        features = compute_pairwise_features(generate_candidate_pairs(sn, tn), sn, tn)
        return features, truth

    def test_shared_truth_targets_grouped_transitively(self):
        groups = truth_groups({"a": {"x"}, "b": {"x", "y"}, "c": {"y"}, "d": set()})
        self.assertEqual(groups["a"], groups["c"])
        self.assertNotEqual(groups["a"], groups["d"])

    def test_training_partial_labels_and_roundtrip(self):
        features, truth = self.prepare()
        del truth["S1_011"]
        artifact, report, diagnostics = train_matcher(features, truth, self.config)
        self.assertNotIn("S1_011", set(diagnostics["oof_scores"].source1_entity_id))
        self.assertEqual(set(diagnostics["fold_assignments"].source1_entity_id), set(truth))
        self.assertTrue(diagnostics["oof_scores"].groupby("source1_entity_id").fold.nunique().eq(1).all())
        model = self.root / "model.joblib"
        save_model(model, artifact)
        pd.testing.assert_frame_equal(score_pairs(features, artifact), score_pairs(features, load_model(model)))
        self.assertGreaterEqual(report["cross_validation"]["macro_f05"], 0)
        self.assertLessEqual(report["cross_validation"]["macro_f05"], 1)

    def test_single_class_training_rejected(self):
        features, truth = self.prepare()
        with self.assertRaisesRegex(ValueError, "positive and negative"):
            train_matcher(features, {s: set() for s in truth}, self.config)

    def test_model_schema_mismatch(self):
        path = self.root / "bad.joblib"
        joblib.dump({"artifact_version": 999}, path)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            load_model(path)

    def test_baseline_train_predict_and_debug_artifacts(self):
        baseline = self.root / "baseline"
        report = execute_pipeline(*self.paths, baseline, truth_path=self.truth_path, config=self.config, save_intermediates=True)
        self.assertEqual(report["status"], "success")
        self.assertEqual(report["blocking"]["blocking_recall"], 1)
        self.assertTrue((baseline / "pair_features.tsv").exists())
        trained = self.root / "trained"
        execute_pipeline(*self.paths, trained, mode="train", truth_path=self.truth_path, config=self.config)
        predicted = self.root / "predicted"
        result = execute_pipeline(*self.paths, predicted, mode="predict", model_path=trained / "model.joblib", truth_path=self.truth_path)
        self.assertEqual(result["config"], self.config.to_dict())
        self.assertGreater(result["evaluation_training_overlap"]["source_entities"], 0)
        self.assertEqual((trained / "matching_results.tsv").read_text(), (predicted / "matching_results.tsv").read_text())
        self.assertTrue(validate_outputs(predicted / "matching_results.tsv", predicted / "candidate_pairs.tsv", *self.paths))
        for name in ("oof_scores", "oof_errors", "threshold_curve", "feature_summary", "fold_assignments"):
            self.assertTrue((trained / f"{name}.tsv").exists())

    def test_packaged_code_and_model_run_from_fresh_directory(self):
        trained = self.root / "trained"
        execute_pipeline(*self.paths, trained, mode="train", truth_path=self.truth_path, config=self.config)
        archive = package_submission(trained, self.root / "submission.zip", *self.paths)
        checkout = self.root / "fresh"
        with ZipFile(archive) as bundle:
            self.assertIn("model.joblib", bundle.namelist())
            self.assertIn("code/business_entity_resolution/USER_MANUAL.md", bundle.namelist())
            bundle.extractall(checkout)
        command = [sys.executable, str(checkout / "code/business_entity_resolution/src/run_pipeline.py"), "--mode", "predict", "--model", str(checkout / "model.joblib"), "--out", str(checkout / "predictions")]
        for i, path in enumerate(self.paths, 1):
            command.extend([f"--s{i}", str(path)])
        result = subprocess.run(command, cwd=checkout, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual((trained / "matching_results.tsv").read_text(), (checkout / "predictions/matching_results.tsv").read_text())

    def test_reject_model_output_collision(self):
        with self.assertRaisesRegex(ValueError, "extension"):
            execute_pipeline(*self.paths, self.root / "run", mode="train", truth_path=self.truth_path, model_path=self.root / "run/matching_results.tsv")

    def test_validator_rejects_non_candidate_match(self):
        out = self.root / "run"
        execute_pipeline(*self.paths, out, config=self.config)
        candidates = pd.read_csv(out / "candidate_pairs.tsv", sep="\t", dtype=str, keep_default_na=False)
        candidates["candidate_entity_ids"] = ""
        candidates.to_csv(out / "candidate_pairs.tsv", sep="\t", index=False)
        self.assertFalse(validate_outputs(out / "matching_results.tsv", out / "candidate_pairs.tsv", *self.paths))

    def test_output_reuse_rejected(self):
        with self.assertRaisesRegex(ValueError, "new or empty"):
            execute_pipeline(*self.paths, self.data)

    def test_failure_report(self):
        out = self.root / "failed"
        with self.assertRaises(ValueError):
            execute_pipeline(self.paths[0], self.paths[1], self.paths[1], out)
        self.assertEqual(json.loads((out / "failure.json").read_text())["status"], "failed")

    def test_empty_sources_end_to_end(self):
        empty = self.write("empty.tsv", "entity_id\tbusiness_name\n")
        report = execute_pipeline(empty, empty, empty, self.root / "empty-output")
        self.assertEqual(report["matched_pairs"], 0)

    def test_no_candidates_still_covers_source(self):
        empty = self.write("empty.tsv", "entity_id\tbusiness_name\n")
        out = self.root / "no-candidates"
        execute_pipeline(self.paths[0], empty, empty, out)
        matches = read_id_lists(out / "matching_results.tsv", "matched_entity_ids")
        self.assertEqual(len(matches), 12)
        self.assertTrue(all(not v for v in matches.values()))

    def test_cli_and_error_exit(self):
        command = [sys.executable, str(SRC / "run_pipeline.py"), "--s1", str(self.paths[0]), "--s2", str(self.paths[1]), "--s3", str(self.paths[2]), "--out", str(self.root / "cli"), "--truth", str(self.truth_path)]
        success = subprocess.run(command, capture_output=True, text=True, timeout=60)
        self.assertEqual(success.returncode, 0, success.stdout + success.stderr)
        failure = subprocess.run(command + ["--mode", "predict"], capture_output=True, text=True, timeout=60)
        self.assertNotEqual(failure.returncode, 0)
        self.assertIn("--model", failure.stderr)


if __name__ == "__main__":
    unittest.main()
