# Business Entity Resolution

Modular normalization, multi-pass blocking, 16 pairwise features, a conservative
baseline, grouped cross-validation training, persisted model inference, macro
F0.5 evaluation, output validation, and submission packaging.

Start with [the user manual](../USER_MANUAL.md). It contains environment setup,
dataset contracts, copyable commands, a synthetic demo, debugging guidance,
module ownership, and known limitations.

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r code/business_entity_resolution/requirements.txt
.\.venv\Scripts\python.exe code/business_entity_resolution/src/make_demo.py --out dataset/demo
.\.venv\Scripts\python.exe code/business_entity_resolution/src/run_pipeline.py --s1 dataset/demo/source1.tsv --s2 dataset/demo/source2.tsv --s3 dataset/demo/source3.tsv --truth dataset/demo/train_ground_truth.tsv --out output/demo-baseline
.\.venv\Scripts\python.exe -m unittest discover -s code/business_entity_resolution/tests -v
```

Each run needs a new or empty output directory. The real datasets and download
links are not included. Synthetic tests do not establish real-data accuracy.
