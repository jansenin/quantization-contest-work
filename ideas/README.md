# Contest Ideas And Handoffs

This directory preserves ideas developed in the discussion-only fork so another
working session can evaluate and implement them. An idea is not a verified
improvement unless its file explicitly says so.

## Current ideas

| File | Topic | Status |
|---|---|---|
| `testing-dataset.md` | Build realistic open-model validation data | Proposed; high priority |
| `artifact-preservation.md` | Promote useful `/tmp/opencode` experiments into Git | Proposed; inventory delegated |
| `linear-experiments.md` | Permutations, error-feedback rounding, and broader search | Proposed/partially related to prior experiments |

## Working convention

- Preserve the motivation and the failure mode an idea is intended to address.
- Distinguish a hypothesis from measured evidence.
- Record exact model/data provenance, commands, seeds, configurations, runtime,
  and output records for every promoted experiment.
- Keep synthetic stress tests, public tests, and real-model tests separate in
  reports. Success on one category must not be presented as success on another.
- Do not copy all temporary files into Git blindly. Promote a minimal reproducible
  experiment and its useful result record; omit caches, duplicate exports, and
  incidental logs.
