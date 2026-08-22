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
