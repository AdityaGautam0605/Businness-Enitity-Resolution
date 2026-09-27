"""Validate a completed run and bundle outputs, code, configuration and documentation."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from validate import validate_outputs


def package_submission(run_dir, destination, s1, s2, s3):
    run_dir, destination = Path(run_dir), Path(destination)
    report = json.loads((run_dir / "run_report.json").read_text(encoding="utf-8"))
    if report.get("status") != "success":
        raise ValueError("Only a completed successful run can be packaged")
    if not validate_outputs(run_dir / "matching_results.tsv", run_dir / "candidate_pairs.tsv", s1, s2, s3):
        raise ValueError("Submission validation failed")
    if destination.exists():
        raise ValueError("Archive already exists; choose a new destination")
    root = Path(__file__).resolve().parents[3]
    project = Path(__file__).resolve().parents[1]
    files = [(run_dir / name, name) for name in ("matching_results.tsv", "candidate_pairs.tsv", "run_report.json")]
    for folder in ("src", "tests"):
        files.extend((path, path.relative_to(root).as_posix()) for path in (project / folder).rglob("*.py"))
    for path in (project / "requirements.txt", project / "requirements-tested.txt", project / "config.example.json", project / "USER_MANUAL.md", project / "src" / "README.md", root / "Documentation_template.md"):
        if path.exists():
            files.append((path, path.relative_to(root).as_posix()))
    if report.get("model_path"):
        model = Path(report["model_path"])
        if not model.is_file():
            raise ValueError("The run's model file is missing")
        files.append((model, "model.joblib"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "x", compression=ZIP_DEFLATED) as archive:
        for path, name in files:
            archive.write(path, name)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "out", "s1", "s2", "s3"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    try:
        print(package_submission(args.run, args.out, args.s1, args.s2, args.s3))
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Packaging failed: {exc}\n")
