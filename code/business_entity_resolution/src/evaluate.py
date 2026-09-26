import sys
import pandas as pd

def compute_macro_f05(pred_file: str, ground_truth_file: str, dump_error_tsv: str = None):
    pred_df = pd.read_csv(pred_file, sep="\t", dtype=str, keep_default_na=False)
    truth_df = pd.read_csv(ground_truth_file, sep="\t", dtype=str, keep_default_na=False)

    preds = {r["source1_entity_id"]: set([x.strip() for x in r["matched_entity_ids"].split(",") if x.strip()])
             for _, r in pred_df.iterrows()}
    truths = {r["source1_entity_id"]: set([x.strip() for x in r["matched_entity_ids"].split(",") if x.strip()])
              for _, r in truth_df.iterrows()}

    f05_scores = []
    error_records = []
    beta = 0.5
    beta_sq = beta ** 2

    for s1_id, y_true in truths.items():
        y_pred = preds.get(s1_id, set())

        if len(y_true) == 0:
            score = 1.0 if len(y_pred) == 0 else 0.0
        else:
            tp = len(y_true & y_pred)
            fp = len(y_pred - y_true)
            fn = len(y_true - y_pred)

            if tp == 0:
                score = 0.0
            else:
                p = tp / (tp + fp)
                r = tp / (tp + fn)
                score = (1 + beta_sq) * (p * r) / ((beta_sq * p) + r)

        f05_scores.append(score)

        if y_true != y_pred and dump_error_tsv:
            error_records.append({
                "source1_entity_id": s1_id,
                "ground_truth": ",".join(sorted(y_true)),
                "predicted": ",".join(sorted(y_pred)),
                "false_positives": ",".join(sorted(y_pred - y_true)),
                "false_negatives": ",".join(sorted(y_true - y_pred))
            })

    macro_f05 = sum(f05_scores) / len(f05_scores) if f05_scores else 0.0
    print("=" * 45)
    print(f" Macro F_0.5 Evaluation Score: {macro_f05:.5f}")
    print(f" Evaluated over {len(f05_scores)} Source 1 records")
    print("=" * 45)

    if dump_error_tsv and error_records:
        pd.DataFrame(error_records).to_csv(dump_error_tsv, sep="\t", index=False)
        print(f"[*] Discrepancies exported to: {dump_error_tsv}")

    return macro_f05

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python src/evaluate.py <predictions.tsv> <train_ground_truth.tsv> [out_errors.tsv]")
        sys.exit(1)
    err_file = sys.argv[3] if len(sys.argv) > 3 else None
    compute_macro_f05(sys.argv[1], sys.argv[2], err_file)