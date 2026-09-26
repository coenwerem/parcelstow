# Three-Task Evaluation Records

This bundle contains 136 canonical task × actor × speedup-factor conditions
from 20 source series, totaling 23,200 episodes. It includes expert, ACT,
Diffusion Policy, and DAgger evaluations. No episode series is pooled with an
overlapping evaluation.

From the code repository root:

```bash
python3 scripts/reproduce_manuscript.py --output-dir outputs/reproduce/canonical
```

The output directory must be new. The command verifies record hashes, counts,
task-stage outcomes, terminal failures, monitor implications, and episode-level
pairing. It produces `success_counts.csv`, `paired_results.json`, and
`validation.json`. Add `--bootstrap` with NumPy installed to compute the
20,000-resample paired confidence intervals. It runs no policy or simulator.

| File | Use |
|---|---|
| [MANUSCRIPT_EVIDENCE.csv](MANUSCRIPT_EVIDENCE.csv) | condition counts, record spans, checkpoint/dataset hashes, audit status, and claim limits |
| [PAIRING_STATUS.csv](PAIRING_STATUS.csv) | expert–learner pairing and permitted statistical treatment |
| [EXCLUDED_SERIES.csv](EXCLUDED_SERIES.csv) | overlapping, superseded, and pilot series excluded from the canonical selection |
| [ARTIFACT_INDEX.csv](ARTIFACT_INDEX.csv) | code/Hugging Face paths, sizes, and hashes for 17 checkpoints and five tensor datasets |
| [FILES.json](FILES.json) | original source paths mapped to packaged files, with original and packaged hashes |
| `records/` | complete compressed source series |
| `provenance/` | frozen training configurations, checkpoint selections, and audit reports |

Paths inside the frozen CSVs refer to the original research checkout. Resolve
record and provenance paths through `FILES.json`; they need not exist at their
original locations in a fresh clone. Decompressing a packaged record reproduces
the original source bytes. The catalogs and original metadata are preserved.

There are 104 verified paired learner conditions and nine standalone conditions.
Pairing is within a speedup factor. The parcel checks establish equality of
recorded initial object poses and reset metadata; those records do
not serialize initial joint states or a bank digest. Unverified conditions
receive standalone success rates, and floor-saturated nominal policies support
failure characterization. Training runs and factors are not pooled.

The [runbook](https://github.com/coenwerem/parcelstow/blob/main/docs/RUNBOOK.md) supplies commands for training and new
simulator evaluations, the artifact map, and the recorded software environments.
