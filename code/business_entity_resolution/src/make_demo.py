"""Create a deterministic synthetic dataset for smoke tests, not a real benchmark."""
import argparse
from pathlib import Path
import pandas as pd


def make_demo(out_dir, count=24):
    output = Path(out_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Demo directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    source, target2, target3, truth = [], [], [], []
    for i in range(count):
        name = f"Orchid {i:03d} Supplies"
        number = 10 + i
        s = f"S1_{i:03d}"
        source.append({"entity_id": s, "business_name": name + " Pvt Ltd", "business_address": f"{number} Cedar Rd", "country": "IN", "city": "Pune"})
        matched = []
        if i % 4 != 0:
            t = f"S2_{i:03d}"
            target2.append({"entity_id": t, "business_name": name + " Private Limited", "business_address": f"{number} Cedar Road", "country": "India", "city": "Pune"})
            matched.append(t)
            if i % 3 == 0:
                t3 = f"S3_MATCH_{i:03d}"
                target3.append({"entity_id": t3, "business_name": name, "business_address": f"{number} Cedar Road", "country": "India", "city": "Pune"})
                matched.append(t3)
        target3.append({"entity_id": f"S3_DECOY_{i:03d}", "business_name": name, "business_address": f"{number + 500} Cedar Road", "country": "India", "city": "Mumbai"})
        truth.append({"source1_entity_id": s, "matched_entity_ids": ",".join(matched)})
    for name, rows in (("source1", source), ("source2", target2), ("source3", target3), ("train_ground_truth", truth)):
        pd.DataFrame(rows).to_csv(output / f"{name}.tsv", sep="\t", index=False)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    make_demo(args.out)
    print(f"Synthetic demo data saved to {args.out}")
