"""Run a baseline, train a classifier, or predict with a saved model."""
import argparse
from config import PipelineConfig
from pipeline import execute_pipeline


def run(s1_path, s2_path, s3_path, out_dir, **kwargs):
    """Backward-compatible Python entry point; defaults to the baseline."""
    return execute_pipeline(s1_path, s2_path, s3_path, out_dir, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s1", required=True, help="Source 1 TSV")
    parser.add_argument("--s2", required=True, help="Source 2 TSV")
    parser.add_argument("--s3", required=True, help="Source 3 TSV")
    parser.add_argument("--out", default="output", help="New or empty run directory")
    parser.add_argument("--mode", choices=("baseline", "train", "predict"), default="baseline")
    parser.add_argument("--truth", help="Ground truth TSV; required for training, optional for evaluation")
    parser.add_argument("--model", help="Saved model input in predict mode; optional output path in train mode")
    parser.add_argument("--config", help="JSON settings; predict defaults to the saved model settings")
    parser.add_argument("--threshold", type=float, help="Override baseline/predict threshold, in [0,1]")
    parser.add_argument("--save-intermediates", action="store_true", help="Save normalized data and pair features")
    args = parser.parse_args()
    try:
        run(args.s1, args.s2, args.s3, args.out, mode=args.mode, truth_path=args.truth,
            model_path=args.model, config=PipelineConfig.load(args.config) if args.config else None,
            threshold=args.threshold, save_intermediates=args.save_intermediates)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Pipeline failed: {exc}\n")


if __name__ == "__main__":
    main()
