# HiF4 Quantization Contest Workspace

The active candidate is always `solution.py`. Git tags preserve standalone
variants; the separate submission repository is updated only after a winner is
selected.

## Quick checks

```bash
python3 -m unittest discover -s tests -v
python3 example/self_check.py --solution_dir . --datasets_dir example/mini_sample
python3 tools/evaluate.py --baseline solution/v000-baseline \
  --candidate solution.py --variant v-local --no-append
```

Add `--mini-sample example/mini_sample` for the slower public-data quality run.
The evaluator writes detailed records under `benchmarks/records/`. Successful
runs append to `progress/results.jsonl` unless `--no-append` is supplied.

Reproduce the public NVFP4 source-scale fingerprint report:

```bash
python3 tools/fingerprint_nvfp4.py \
  --markdown-output docs/public-nvfp4-fingerprint.md
```

## Dashboard

```bash
python3 -m http.server 8000
```

Open `http://localhost:8000/progress/`. The plotted metric is the arithmetic mean
over cases of `100 * (baseline_mse - candidate_mse) / baseline_mse`; it is not the
unavailable official score.

## Variant workflow

1. Start from a clean tagged candidate.
2. Change only the active `solution.py` algorithm and supporting documentation.
3. Run unit tests, the official checker, and synthetic/public comparisons.
4. Commit the solution and its benchmark record together.
5. Create an annotated `solution/vNNN-name` tag.

Export a tagged candidate without using the working tree:

```bash
python3 tools/export_solution.py \
  --ref solution/v008-gated-hadamard --output solution.zip
```

The exporter writes a deterministic archive containing only root-level
`solution.py` and prints the source and archive SHA-256 hashes.

Contract details and evaluator assumptions are in `docs/problem-contract.md`.
Proved optimization facts and experiment priorities are in
`docs/research-notes.md`.

## Model downloads

The real-model data pipeline keeps model snapshots and partial downloads under
ignored `data/`. Set up its isolated dependency once:

```bash
python3 -m venv --system-site-packages .venv-data
.venv-data/bin/pip install -r requirements-data.txt
```

Using system site packages reuses an existing CPU PyTorch installation instead
of downloading a second large PyTorch/CUDA stack into the data-tooling venv.

Inspect the model profiles, then start the laptop-sized profile in the
background:

```bash
.venv-data/bin/python tools/download_models.py --list
mkdir -p data
nohup .venv-data/bin/python -u tools/download_models.py --profile laptop \
  > data/download-models.log 2>&1 &
```

The Hugging Face cache retains completed blobs and interrupted partial files.
Internet failures are retried with exponential backoff. After process
termination or reboot, rerun the same command to resume. Current status is
available without network access:

```bash
.venv-data/bin/python tools/download_models.py --status
```

On the high-memory work machine, `--profile work` selects the small and medium
models plus a dense Qwen 72B model, approximately 223.66 GB in total. Add
`--profile work-large` for the optional 30B-72B breadth suite; combining both
profiles selects approximately 414.72 GB after deduplication:

```bash
.venv-data/bin/python tools/download_models.py \
  --profile work --profile work-large --dry-run
```

To put model data on a different disk, pass both `--cache-dir` and
`--state-file`; reuse the same paths when resuming.

## Real-model capture

Capture a short integration dataset from the pinned Qwen3-0.6B snapshot:

```bash
.venv-data/bin/python tools/capture_real.py --smoke --threads 1
```

After the smoke run succeeds, capture the full five-calibration/five-test
sequence-length pattern:

```bash
.venv-data/bin/python tools/capture_real.py --threads 1
```

By default the capture selects the first, middle, and last transformer layers
and records `q_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj` Linear
groups plus post-normalization, post-RoPE Q/K and attention-input V. Raw BF16
groups are written atomically under ignored
`data/real-captures/<dataset_id>/`. Rerunning an identical command verifies
completed shard hashes and resumes missing groups; use `--force` only to
discard and rebuild that dataset.

## Real-model evaluation

Evaluate a capture one shard at a time against tagged solutions. The source
NVFP4 data is derived from raw BF16 independently under ceiling, nearest, and
seeded stochastic E4M3 scale selection:

```bash
python3 tools/evaluate_real.py \
  --dataset 45ac5df1ecf7cca4 \
  --baseline solution/v000-baseline \
  --candidate solution/v008-gated-hadamard \
  --threads 1
```

Use repeatable `--candidate`, `--modes`, and `--group-filter` arguments to
restrict comparisons. For example, `--modes ceil --group-filter linear
--group-filter role:q_proj,role:o_proj --limit 2` selects at most two matching
Linear groups.

The evaluator validates and releases each shard before loading the next one.
It writes atomic per-case records, baseline caches, and run manifests under
ignored `benchmarks/realdata/`. Resume keys include source-file hashes or
resolved Git commits plus the evaluator semantics, so editing a path candidate
cannot silently reuse stale results. `--force` recomputes all selected units.

Tracked, disaggregated results from completed captures are recorded in
[`docs/real-model-benchmarks.md`](docs/real-model-benchmarks.md).
User-reported contest-platform scores and runtimes are recorded separately in
[`docs/official-submission-results.md`](docs/official-submission-results.md).
