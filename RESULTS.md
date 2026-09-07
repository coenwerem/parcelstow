# Policy Results

ParcelStow accepts results from policies evaluated through the public task
interface and the fixed task-specific protocols. The tables below distinguish
results included with ParcelStow from external submissions.

## Included Policy Results

Each entry reports 100 episodes at the indicated speed. The expert and policy
use the same indexed initial conditions within each task and speed. The
[Benchmark Specification](docs/BENCHMARK.md) defines the complete evaluation
grids, and `python3 scripts/reproduce.py all-tasks` recomputes every count from
the episode records.

| Task | Policy | `r=1` | `r=2` | Evidence Scope |
|---|---|---:|---:|---|
| Parcel insertion | Expert | 100/100 | 84/100 | `v1.0.0` and arXiv v1 |
| Parcel insertion | ACT (ACT-A checkpoint) | 100/100 | 53/100 | `v1.0.0` and arXiv v1 |
| Upright placement | Expert | 92/100 | 43/100 | current `main` development records |
| Upright placement | ACT | 39/100 | 8/100 | current `main` development records |
| Keyed peg insertion | Expert | 93/100 | 87/100 | current `main` development records |
| Keyed peg insertion | ACT | 75/100 | 0/100 | current `main` development records |

The parcel rows are the primary nominally matched comparison because the
expert and ACT both succeed in 100/100 episodes at `r=1`. The included upright
and keyed peg ACT checkpoints do not meet that condition. Their rows are
task-specific development results, not primary matched comparisons or results
from a released v2 package.

## Community Policy Results

No external policy results are listed yet.

## Submit a Policy Result

A submission may cover one task or all three. For every submitted task:

1. Evaluate the policy through `scripts/evaluate.py` on the complete registered
   speed grid with 100 episodes per speed.
2. Use the evaluator's indexed initial-condition bank without changing the
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
