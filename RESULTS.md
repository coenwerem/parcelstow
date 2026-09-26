# Policy Results

ParcelStow accepts results from policies evaluated through the public task
interface and the fixed task-specific protocols. The tables below distinguish
results included with ParcelStow from external submissions.

## Included Policy Results

| Task | Expert at `r=1` | ACT at `r=1` | Higher `r` | Expert | ACT |
|---|---:|---:|---:|---:|---:|
| Parcel insertion | **100/100** | **100/100** | 2 | **84/100** | **53/100** |
| Upright placement | **185/200** | **194/200** | 2 | **105/200** | **24/200** |
| Keyed peg insertion | **182/200** | **191/200** | 1.5 | **150/200** | **3/200** |

Each ACT column reports one trained policy per task. Expert-minus-ACT
differences at the highlighted factors are 31, 40.5, and 73.5 percentage points,
with pointwise 95% paired bootstrap intervals `[18,44]`, `[32,49]`, and
`[67.5,79.5]`. Parcel `r=2` is within its demonstrated range; upright `r=2`
and peg `r=1.5` are outside theirs. Training replications are reported
individually in the evaluation records.

Diffusion Policy (DP) nominal success is 68/100, 181/200, and 95/200 on parcel,
upright, and peg. Its parcel results support standalone success rates because
episode-level pairing and distribution matching with the expert are not
established. DAgger nominal success is 3/100, 1/200, and 0/200; these results
support failure characterization but not execution-speed degradation claims.
The record catalogs provide task-stage counts, terminal failures, hashes,
pairing restrictions, and excluded series.

[Canonical inventory](data/manuscript_20260921/MANUSCRIPT_EVIDENCE.csv) ·
[Runbook and reproduction commands](docs/RUNBOOK.md)

```bash
python3 scripts/reproduce_manuscript.py --output-dir outputs/reproduce/results
```

## Community Policy Results

No external policy results are listed yet.

## Submit a Policy Result

A submission may cover one task or all three. For every submitted task:

1. Evaluate the policy through `scripts/evaluate.py` on the complete registered
   speed grid with 100 parcel episodes or 200 upright/peg episodes per speed,
   using the task-specific bank and explicit command in [RUNBOOK.md](docs/RUNBOOK.md).
2. Use the declared task-specific initial-condition bank without changing the
   task geometry, phase schedule, initial-condition distribution, observation,
   action, success predicates, or failure reasons.
3. Provide the episode records and summary produced by the evaluator. Do not
   edit the JSON Lines records after evaluation.
4. Identify the ParcelStow commit, policy code commit, checkpoint path and
   SHA-256, evaluation seed, software environment, and compute hardware used
   for the simulator evaluation.
5. Provide a CPU-only command that verifies the submitted counts from the
   episode records and one short representative video for each submitted task.
6. Report every evaluated condition, including failures. State whether the
   policy used ParcelStow demonstrations, additional data, task-specific
   training, privileged observations, or an observation adapter.

Start with the [Policy Interface](docs/POLICY_INTERFACE.md) and follow the
[contribution requirements](CONTRIBUTING.md#pull-request-evidence). Add the
proposed result rows under **Community Policy Results** and link each row to
its code, checkpoint provenance, records, and reproduction command.

Maintainer review checks record integrity, matched initial conditions, task
compatibility, and reproducibility. Listing a submitted result does not make
the policy part of a stable ParcelStow release.
