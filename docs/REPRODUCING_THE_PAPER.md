# Reproducing the Results

The [reproduction runbook](RUNBOOK.md) contains training, checkpoint selection,
evaluation, and data-collection commands. Start with the saved records to
reproduce the reported counts without Isaac Lab, a GPU, or model downloads.

## Counts and Pairing

From the repository root:

```bash
python3 scripts/reproduce_manuscript.py --output-dir outputs/reproduce/counts
```

Choose a new output directory. The command validates the packaged and original
record hashes, episode identities, success counts, task-stage completion,
terminal failures, monitor implications, and pairing metadata.

Expected coverage: **136 conditions, 20 source series, and 23,200 episodes**,
with **104 verified paired learner conditions and nine standalone conditions**.
The outputs are `success_counts.csv`, `paired_results.json`, and `validation.json`.

## Confidence Intervals

With NumPy installed:

```bash
python3 scripts/reproduce_manuscript.py --bootstrap \
  --output-dir outputs/reproduce/bootstrap
```

This adds pointwise 95% paired bootstrap intervals from 20,000 resamples.
Differences are expert minus learner. Unpaired conditions receive standalone
success rates and Wilson binomial intervals.

| Task and speedup factor | Expert success | ACT success | Paired difference, percentage points | 95% interval |
|---|---:|---:|---:|---:|
| Parcel, `r=2` | 84/100 | 53/100 | 31 | [18, 44] |
| Upright, `r=2` | 105/200 | 24/200 | 40.5 | [32, 49] |
| Peg, `r=1.5` | 150/200 | 3/200 | 73.5 | [67.5, 79.5] |

The [record guide](../data/manuscript_20260921/README.md) explains the catalogs
and their statistical scope. Training runs, task conditions, and overlapping
episode series are not pooled.

## Training and Simulator Evaluation

Follow the [runbook environment setup](RUNBOOK.md#5-environments-and-fresh-output-directories),
[checkpoint evaluation](RUNBOOK.md#6-evaluate-the-frozen-checkpoints), and
[training procedures](RUNBOOK.md#7-reproduce-training-procedures).
Each run uses a new output directory. The recipes identify datasets and
checkpoints by hash and specify the task's complete factor grid and episode bank.
New simulator runs produce new records; saved-record reproduction above is the
source of the exact reported counts.
