# ParcelStow

Evaluate how learned dexterous-manipulation policies respond to higher
task rates.

ParcelStow is an Isaac Lab benchmark for measuring policy success as the
requested task rate `r` increases. A Unitree G1 arm with a RealHand L6
dexterous hand acquires a small parcel, reorients it by 90 degrees, and
inserts it into a cubby with 10 mm clearance. The task rate scales the
duration of the manipulation phases; the scene geometry, object, grasp,
acquisition timing, success criteria, and policy interfaces remain fixed.
Every policy receives `r` in its observation. The success fraction as a
function of `r` is the policy's task-rate operating envelope.
[docs/BENCHMARK.md](docs/BENCHMARK.md) specifies the rate perturbation,
evaluation protocol, and reference actors.

| | | | |
|---|---|---|---|
| [Quick start](#quick-start) | [Evaluate your policy](#evaluate-your-policy) | [Benchmark specification](docs/BENCHMARK.md) | [Policy interface](docs/POLICY_INTERFACE.md) |
| [Reproduce the paper](docs/REPRODUCING_THE_PAPER.md) | [Data and checkpoints](docs/DATA_AND_CHECKPOINTS.md) | [Diagnostics](docs/DIAGNOSTICS.md) | [Citation](#citation) |

At `r=2`, the expert completes the insertion, while DAgger leaves the
parcel on its side against the receptacle wall.

<p align="center">
  <img src="media/task_rate_robustness.gif" alt="Expert and DAgger at r=2" width="640">
</p>

DAgger succeeds in only 3 of 100 episodes at the nominal rate. Its
behavior at `r=2` therefore cannot isolate sensitivity to task rate from
its nominal-rate failures. ACT-A succeeds in all 100 nominal-rate
episodes. In the `r=2` episode shown below, ACT-A finishes with a 17.1
degree orientation error, which exceeds the 10 degree settling
tolerance.

<p align="center">
  <img src="media/terminal_states_r2.png" alt="Terminal states at r=2 for the expert, ACT-A, and DAgger" width="700">
</p>

Terminal states at `r=2`, with the receptacle interior magnified. Each
subcaption reports the final parcel orientation error for the displayed
episode; successful settling requires an error of at most 10 degrees.
The panels show separate episodes, not outcomes from a shared initial
condition.

The expert and ACT-A, which was trained on the expert's demonstrations,
each succeed in 100 of 100 episodes at `r=1`. At `r=2`, within ACT-A's
training range of `[0.5, 2]`, expert success decreases by 16 percentage
points and ACT-A success decreases by 47 percentage points.

| actor | `r=1` | `r=2` | change |
|---|---|---|---|
| Expert | 100/100 | 84/100 | -16 |
| ACT-A | 100/100 | 53/100 | -47 |

Each condition contains 100 episodes, with identical evaluation draws
for every actor ([docs/BENCHMARK.md](docs/BENCHMARK.md)). At `r=2`, the
expert exceeds ACT-A by 31 percentage points; a 20,000-resample paired
bootstrap gives a 95% confidence interval of `[0.18, 0.44]` for this
gap. Two ACT policies trained with different parameter-initialization
seeds decrease by 34 and 48 percentage points between `r=1` and `r=2`,
although their nominal success rates are lower at 70/100 and 62/100.
Across these three training seeds, ACT success decreases more than expert
success as the task rate rises from `r=1` to `r=2`. Comparing the expert
and policy operating envelopes isolates sensitivity to task rate only
when their success rates match at the nominal rate.

<p align="center">
  <img src="media/operating_envelope.png" alt="Task-rate operating envelope" width="600">
</p>

Success fraction against task rate for every reference actor. Each point
contains 100 episodes and shows a Wilson 95% confidence interval. Rates
above `r=2` lie outside the demonstrated training range.

The released records identify the first failed task predicate in each
episode. They also report in-hand translation and rotation, actuator
utilization, target-tracking error, and realized contact sets
([docs/DIAGNOSTICS.md](docs/DIAGNOSTICS.md)).

## Quick start

**Tier 0, no Isaac, no GPU.** The complete episode-level evaluation
records ship in this repository. Reproduce the success-versus-rate
figure, success table, and paired-bootstrap interval directly from a
clone,

```bash
pip install numpy matplotlib
python scripts/reproduce.py envelope
```

**Tier 1, first simulator run.** With Isaac Lab installed
([installation](#installation)), run the scripted expert for five
episodes,

```bash
python scripts/run_task.py
```

**Tier 2, run a released reference actor.** Evaluate the ACT-A
checkpoint over a rate pair,

```bash
python scripts/download_artifacts.py --demo     # ACT-A checkpoint
python scripts/evaluate.py --actor act --rates 1.0 2.0 --episodes 100
```

**Tier 3, full reproduction.** Fetch demonstrations and all checkpoints,
or retrain them, then rerun the full evaluation
([docs/REPRODUCING_THE_PAPER.md](docs/REPRODUCING_THE_PAPER.md)).

## Installation

The analysis tier needs Python 3.10+ with numpy and matplotlib. The
simulation tiers need [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)
(tested with Isaac Sim 5.1). Install the task extension into the Isaac
Lab Python environment,

```bash
uv pip install -p <isaaclab-venv>/bin/python -e source/parcelstow
```

then run the geometry and simulator-backed physics tests,

```bash
python -m pytest tests/ -q            # pure geometry tests, no simulator
python -m pytest tests/ --isaac -q    # simulator-backed physics tests
```

## Evaluate your policy

Implement three members, `name`, `reset(ids, obs)`, and
`act(obs) -> (action, q_target)`, over the frozen 147-D state observation
(requested rate included at index 146) and 16-D joint-position action at
50 Hz. `examples/custom_policy.py` is a complete runnable example,

```bash
python scripts/evaluate.py --actor examples.custom_policy:HoldPosturePolicy \
    --rates 1.0 --episodes 5 --num_envs 8
python scripts/plot_envelope.py --summary outputs/eval/summary.jsonl
```

[docs/POLICY_INTERFACE.md](docs/POLICY_INTERFACE.md) documents the
observation slices, action semantics, reset protocol, and record schema.

## Repository map

| path | content |
|---|---|
| `scripts/run_task.py`, `evaluate.py`, `plot_envelope.py`, `reproduce.py`, `download_artifacts.py` | supported public commands |
| `source/parcelstow/` | the Isaac Lab extension, task, geometry, monitor |
| `scripts/manipulation/` | the validated experiment drivers behind the public commands |
| `examples/custom_policy.py` | minimal policy showing the integration point |
| `data/records/` | released episode-level evaluation records |
| `experiments/paper/results/` | frozen derived analyses of the paper |
| `artifacts/manifest.json` | external artifact inventory with checksums |
| `docs/` | benchmark spec, task spec, interface, reproduction, diagnostics |
| `tests/` | pure geometry tests and simulator-backed physics tests |

## Citation

The accompanying preprint is in preparation. The arXiv identifier will
appear here and in `CITATION.cff` once assigned.

```bibtex
@misc{enwerem2026parcelstow,
  title  = {ParcelStow: Task-Rate Robustness Evaluation for Learned
            Dexterous Manipulation},
  author = {Enwerem, Clinton},
  year   = {2026},
  note   = {arXiv identifier pending}
}
```
