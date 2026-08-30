# ParcelStow

Matched expert–learner evaluation of temporal robustness in dexterous
manipulation.

ParcelStow is an Isaac Lab benchmark for a contact-rich parcel insertion
task. The robot acquires a free rigid parcel, lifts it from the table,
reorients it by 90 degrees, transports it to an open-front receptacle,
inserts it with 10 mm of clearance per side along the tight axis, and
releases it. A successful episode requires the parcel to settle within
the final position and orientation tolerances. The simulation uses a
fixed-base Unitree G1 humanoid with a RealHand L6 anthropomorphic right
hand.

ParcelStow tests whether an imitation (learner) policy preserves its expert's
success response under variation in execution speed. At each requested speed,
the expert and learner receive the same task conditions and initial-condition
draws. The speedup factor `r` divides the nominal durations of lift,
reorientation, transfer, insertion, release, and retreat after the parcel has
been acquired. Acquisition timing, task geometry, success criteria,
observation, and action remain fixed, and every policy receives `r` in its
observation. The demonstrations contain speeds in `r ∈ [0.5, 2]`; evaluations
above `r=2` test extrapolation beyond the demonstrated range.
[docs/BENCHMARK.md](docs/BENCHMARK.md) specifies the speed variation,
evaluation protocol, and reference policies.

| | | | |
|---|---|---|---|
| [Quick Start](#quick-start) | [Evaluate Your Own Policy](#evaluate-your-policy) | [Benchmark Specification](docs/BENCHMARK.md) | [Policy Interface](docs/POLICY_INTERFACE.md) |
| [Reproduce Our Paper's Results](docs/REPRODUCING_THE_PAPER.md) | [Data and checkpoints](docs/DATA_AND_CHECKPOINTS.md) | [Diagnostics](docs/DIAGNOSTICS.md) | [Citation](#citation) |

At `r=2`, the maximum demonstrated speed, the expert completes the insertion,
while DAgger leaves the parcel on its side against the receptacle wall.

<p align="center">
  <img src="media/task_rate_robustness.gif" alt="Expert and DAgger at r=2" width="640">
</p>

DAgger succeeds in only 3 of 100 episodes at nominal speed. Its behavior at
`r=2` therefore cannot distinguish sensitivity to execution speed from its
nominal failures. ACT-A succeeds in all 100 nominal-speed episodes. In the
`r=2` episode shown below, ACT-A finishes with a 17.1 degree orientation error,
which exceeds the 10 degree settling tolerance.

<p align="center">
  <img src="media/terminal_states_r2.png" alt="Terminal states at r=2 for the expert, ACT-A, and DAgger" width="700">
</p>

Terminal states at `r=2`, with the receptacle interior magnified. Each
subcaption reports the final parcel orientation error for the displayed
episode; successful settling requires an error of at most 10 degrees.
The panels show separate episodes, not outcomes from a shared initial
condition.

The expert and ACT-A, which was trained on the expert's demonstrations,
each succeed in 100 of 100 episodes at `r=1`. At `r=2`, within the
demonstrated speed range `[0.5, 2]`, expert success decreases by 16 percentage
points and ACT-A success decreases by 47 percentage points.

| policy | `r=1` | `r=2` | decrease |
|---|---|---|---|
| Expert | 100/100 | 84/100 | -16 |
| ACT-A | 100/100 | 53/100 | -47 |

Each condition contains 100 episodes, with identical evaluation draws for the
expert and each ACT learner ([docs/BENCHMARK.md](docs/BENCHMARK.md)). At `r=2`,
the expert exceeds ACT-A by 31 percentage points; a 20,000-resample paired
bootstrap gives a 95% confidence interval of `[0.18, 0.44]` for this
gap. Two ACT policies trained with different parameter-initialization
seeds decrease by 34 and 48 percentage points between `r=1` and `r=2`,
although their nominal success rates are lower at 70/100 and 62/100.
Across these three parameter initializations, ACT success decreases more than
expert success as execution speed increases from `r=1` to `r=2`. Only ACT-A
supports an expert–learner comparison after equal nominal success because
ACT-B and ACT-C start below expert success at `r=1`.

<p align="center">
  <img src="media/operating_envelope.png" alt="Task success across execution speeds" width="600">
</p>

Task success against execution speed for every reference policy. Each point
contains 100 episodes and shows a Wilson 95% confidence interval. Speeds above
`r=2` lie outside the demonstrated range.

The stage outcomes locate 35 of ACT-A's 47 failures at `r=2` in insertion
misalignment. Under the relative-motion handoff, all 100 ACT-A acquisitions
retain the parcel through free-space reorientation and transfer under the
expert's relative hand motion. Continuing the same motion through insertion,
release, and settling produces 64 successes after ACT-A acquisition and 95
after expert acquisition. The handoff rules out parcel loss during free-space
transport as the source of the difference, but it does not separate the state
at handoff from subsequent receptacle contact.

Across the six evaluated policies and all tested speeds, none of the 414
acquisitions without force closure completes the task. This is a one-sided
observation: force closure is necessary in these evaluations but does not
predict success, and the continuous Ferrari-Canny margin does not rank or
calibrate success consistently across policies and speeds. The released
records also contain stage completion, terminal failure reason, in-hand
translation and rotation, arm joint-velocity utilization, target-tracking
error, and realized contact sets
([docs/DIAGNOSTICS.md](docs/DIAGNOSTICS.md)).

## Quick Start
**Tier 0, no Isaac, no GPU.** The complete episode-level evaluation
records ship in this repository. Reproduce the success-versus-speed
figure, success table, and paired-bootstrap interval directly from a
clone,

```bash
python3 -m pip install numpy matplotlib
python3 scripts/reproduce.py envelope
```

**Tier 1, First Simulator Run.** With Isaac Lab installed
(see [installation](#installation)), run the scripted expert for five
episodes,

```bash
python scripts/run_task.py
```

**Tier 2, Run a Released Reference Policy.** Evaluate the ACT-A
checkpoint at two speeds,

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
(speedup factor included at index 146) and 16-D joint-position action at
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
  title  = {Does Imitation Preserve Temporal Robustness in Dexterous
            Manipulation? An Expert–Learner Comparison Across Task Execution Speeds},
  author = {Enwerem, Clinton and Baras, John S. and Belta, Calin},
  year   = {2026},
  note   = {arXiv identifier pending}
}
```
