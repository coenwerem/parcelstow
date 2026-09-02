# ParcelStow

An Isaac Lab benchmark for expert-learner evaluation across task
execution speeds.

[![Dataset on HF](https://img.shields.io/badge/%F0%9F%A4%97-Dataset-lightgrey)](https://huggingface.co/datasets/cenwerem/parcelstow)
[![Paper (arXiv)](https://img.shields.io/badge/arXiv-2609.01453-b31b1b)](https://arxiv.org/abs/2609.01453)

<p align="justify">
ParcelStow's task is contact-rich parcel insertion.
In the task provided with $\texttt{v1.0.0}$, a poly-articulated robot equipped with an anthropomorphic end-effector is charged with inserting a rigid parcel placed stably on a planar surface at a known initial pose in the robot's workspace into an open-front receptacle whose pose is also known. The parcel insertion task is further divided into several stages that comprise acquiring the rigid parcel (acquisition/grasping),
reorienting the parcel by 90 degrees (reorientation), transporting the parcel to the receptacle (transport), inserting the parcel with 10 mm of clearance per side along the tight axis (insertion), and releasing the parcel (release). A successful episode requires the parcel to settle within the final position and orientation tolerances. The simulation tooling, policy training, and policy evaluation assets provided in $\texttt{v1.0.0}$ utilize a fixed-base Unitree G1 humanoid with a RealHand L6 anthropomorphic right hand.
</p>

<p align="justify">
At its core, ParcelStow primarily tests if an imitation (learner) policy preserves its expert's success response under temporal variation, with execution speed serving as the immediate temporal variation channel. At each specified speed,
we supply both the expert and learner policies with the same task conditions and initial-condition draws. The speedup factor $r$ partitions the nominal durations of lift,
reorientation, transfer, insertion, release, and retreat after the parcel has
been acquired. We also fix the acquisition timing, task geometry, success criteria,
observation, and action, and include $r$ in each policy's
observation vector. The demonstrations contain speeds in $r \in [0.5, 2]$; evaluations
above $r=2$ test extrapolation beyond the demonstrated range. The dedicated <a href="docs/BENCHMARK.md">BENCHMARK.md file</a> specifies the speed variation,
evaluation protocol, and reference policies.
</p>

<table align="center">
  <tr>
    <td align="center">
      <img src="media/task_rate_robustness.gif" alt="Expert and DAgger at r=2" width="80%">
      <br>
      <p align="justify"><sub><b>Illustrative Temporal-Sensitivity Demo. </b>At $r=2$, the maximum demonstrated speed, the expert completes the insertion task, while DAgger jerkily inserts the parcel in the wrong orientation, consequently failing the task.</sub></p>
    </td>
  </tr>
</table>

## Outline

- [Results Preview](#results-preview)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Evaluate Your Own Policy](#evaluate-your-own-policy)
- [Repository Map](#repository-map)
- [Benchmark Specification](docs/BENCHMARK.md)
- [Policy Interface](docs/POLICY_INTERFACE.md)
- [Reproducing Our Paper's Results](docs/REPRODUCING_THE_PAPER.md)
- [Data and Checkpoints](docs/DATA_AND_CHECKPOINTS.md)
- [Diagnostics](docs/DIAGNOSTICS.md)
- [Citation](#citation)

## Results Preview

### A. Orientation Error at the Maximum Demonstrated Speed
ACT-A succeeds in all 100 nominal-speed episodes. In the
`r=2` episode shown below, ACT-A finishes with a 17.1 degree orientation error,
which exceeds the 10 degree settling tolerance.

<table align="center">
  <tr>
    <td align="center">
      <img src="media/terminal_states_r2.png" alt="Terminal states at r=2 for the expert, ACT-A, and DAgger" width="85%">
      <br>
      <p align="justify"><sub><b>Comparing Expert-Learner Insertion Performance at the Maximum Demonstrated Speed. </b>Terminal states at $r=2$, with the receptacle interior magnified. Each subcaption reports the final parcel orientation error for the displayed episode; successful settling requires an error of at most 10 degrees. The panels show separate episodes, not outcomes from a shared initial condition.</sub></p>
    </td>
  </tr>
</table>

### B. Success Rate vs. Execution Speed
The expert and ACT-A each succeed in 100 of 100 episodes at nominal speed.
At $r=2$, the maximum demonstrated speed, the expert succeeds in 84 episodes
and ACT-A succeeds in 53. Their success rates differ by 31 percentage points;
a paired bootstrap gives a 95% confidence interval of $[0.18, 0.44]$ for this
difference. ACT-B and ACT-C also lose more success than the expert between
$r=1$ and $r=2$, but neither matches the expert at nominal speed.

<table align="center">
  <tr>
    <td align="center">
      <img src="media/operating_envelope.png" alt="Task success across execution speeds" width="65%">
      <br>
      <p align="justify"><sub><b>Per-Policy Success Rate versus Execution Speed. </b>Task success across execution speeds for the expert and learner policies. Each point contains 100 episodes and shows a Wilson 95% confidence interval; speeds above $r=2$ lie outside the demonstrated range. The paper reports the stage, relative-motion-handoff, and force-closure analyses; the corresponding measurements are documented in <a href="docs/DIAGNOSTICS.md">docs/DIAGNOSTICS.md</a>.</sub></p>
    </td>
  </tr>
</table>

## arXiv-v2 Extension Tasks

<p align="justify">
The arXiv-v2 extension carries the matched expert-learner evaluation to two
further tasks on the same embodiment, observation grammar, and control
interface: upright placement (terminal quasi-static stability on a marked
target region) and keyed-peg insertion (tight-clearance containment, 3 mm
per side, through a lead-in funnel). Frozen specifications live in
<a href="docs/TASK_SPEC_UPRIGHT.md">TASK_SPEC_UPRIGHT.md</a> and
<a href="docs/TASK_SPEC_PEG.md">TASK_SPEC_PEG.md</a>, the released records
under <code>data/records/upright/</code> and <code>data/records/peg/</code>,
and the increment-by-increment evidence in
<a href="docs/EXTENSION_PLAN.md">EXTENSION_PLAN.md</a>.
</p>

<table align="center">
  <tr>
    <td align="center">
      <img src="media/upright_expert_r1.gif" alt="Expert upright placement at r=1" width="97%">
      <br>
      <p align="justify"><sub><b>Upright Placement. </b>The scripted expert stands the 180 mm cuboid on the marked target region at $r=1$; the expert succeeds in 92 of 100 evaluation episodes at this speed.</sub></p>
    </td>
    <td align="center">
      <img src="media/peg_expert_r1.gif" alt="Expert keyed-peg insertion at r=1" width="97%">
      <br>
      <p align="justify"><sub><b>Keyed-Peg Insertion. </b>The scripted expert inserts the same cuboid into a square pocket with 3 mm of clearance per side at $r=1$; the expert succeeds in 94 of 100 evaluation episodes at this speed.</sub></p>
    </td>
  </tr>
</table>

<table align="center">
  <tr>
    <td align="center">
      <img src="media/peg_expert_vs_act_r1.gif" alt="Expert and ACT peg insertion at r=1" width="80%">
      <br>
      <p align="justify"><sub><b>Expert-Learner Contrast at Nominal Speed. </b>At $r=1$ the expert (left) threads the funnel while the ACT pilot (right) drops the peg during transport; the paired evaluation reads 94 against 76 of 100, and the ACT pilot loses acquisition entirely at $r \geq 1.5$, above its demonstrated range.</sub></p>
    </td>
  </tr>
</table>

## Quick Start
**Tier 0, no Isaac Lab, no GPU**

The complete episode-level evaluation records are bundled with this repository. Reproduce the success-versus-speed figure, success table, and paired-bootstrap interval from a
clone using the following command:

```bash
python3 -m pip install numpy matplotlib
python3 scripts/reproduce.py envelope
```

**Tier 1, First Simulator Run**

With Isaac Lab installed (see [Installation](#installation)), run the scripted expert for five
episodes,

```bash
python scripts/run_task.py
```

**Tier 2, Run a Released Reference Policy**

Evaluate the ACT-A checkpoint, fetched from the [Hugging Face
dataset](https://huggingface.co/datasets/cenwerem/parcelstow), at two speeds:

```bash
python scripts/download_artifacts.py --demo     # ACT-A checkpoint
python scripts/evaluate.py --actor act --rates 1.0 2.0 --episodes 100
```

**Tier 3, Full Reproduction**

Retrieve demonstrations and all checkpoints from the [Hugging Face
dataset](https://huggingface.co/datasets/cenwerem/parcelstow) (or obtain them from a fresh training pass), then rerun the full evaluation following
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

## Evaluate Your Own Policy
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

## Repository Map
| Path | Content |
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
The accompanying preprint is
[arXiv:2609.01453](https://arxiv.org/abs/2609.01453) \[cs.RO\].

```bibtex
@misc{enwerem2026parcelstow,
  title  = {Does Imitation Learning Preserve Temporal Robustness in Dexterous
            Manipulation? An Expert-Learner Comparison Across Task Execution Speeds},
  author = {Enwerem, Clinton and Baras, John S. and Belta, Calin},
  year   = {2026},
  eprint = {2609.01453},
  archivePrefix = {arXiv},
  primaryClass = {cs.RO},
  url    = {https://arxiv.org/abs/2609.01453}
}
```
