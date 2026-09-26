# Benchmark Specification

ParcelStow evaluates how the success-rate difference between an imitation
learner and its expert varies under changes in task conditions. The three tasks
use execution timing as the experimental variable, with fixed geometry,
physical parameters, action interface, and success criteria.

The expert–ACT comparisons use verified episode-level pairing at each speedup
factor. Parcel ACT matches the expert's observed nominal success; upright and
peg ACT exceed it. All three comparisons favor the expert at the highlighted
higher factors. Other learners have different nominal performance and pairing
restrictions, recorded in [PAIRING_STATUS.csv](../data/manuscript_20260921/PAIRING_STATUS.csv).
The [runbook](RUNBOOK.md) specifies artifacts, evaluation banks, and commands.

| Task | Task Specification | Demonstrated `r` | Evaluation Grid |
|---|---|---:|---|
| Parcel insertion | [TASK_SPEC.md](TASK_SPEC.md) | `[0.5, 2.0]` | `{0.5, 1, 1.5, 2, 2.25, 2.5, 3}` |
| Upright placement | [TASK_SPEC_UPRIGHT.md](TASK_SPEC_UPRIGHT.md) | `[0.75, 1.75]` | `{0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5}` |
| Keyed peg insertion | [TASK_SPEC_PEG.md](TASK_SPEC_PEG.md) | `[0.5, 1.0]` | `{0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5}` |

## Execution-Speed Intervention

The speedup factor `r` divides the nominal duration of phases marked as scaled
in the selected task specification. Acquisition and settling retain fixed
durations in all three tasks. Consequently, `r=1` gives a task's
nominal phase schedule, whereas `r=2` halves only that task's scaled phases.
For parcel insertion, the complete cycle lasts 14.1 s at `r=1` and 10.2 s at
`r=2`.

Each task samples demonstrations uniformly from the range in the table above.
Evaluation outside that range tests speed extrapolation. Every policy observes
`r`; the task identity is selected by the public command and is not appended to
the observation.

Changing `r` preserves geometry, mass, friction, the expert's reference path,
initial-condition distribution, observation layout, action semantics, and
success predicates. The observation contains the speedup factor and task phase,
so the intervention changes both the schedule and the timing inputs to the policy.
The expert–ACT comparisons use verified recorded initial conditions at
each speed; baseline exceptions are listed in the pairing catalog.

## Task Success

Each task specification defines task-specific stage outcomes, terminal failure
reasons, and physical success predicates. Analytical grasp scores do not enter
task success. Parcel requires insertion and settling in its receptacle;
upright requires a released, stable upright pose inside the target region; peg
requires insertion and settling inside the square pocket.

## Evaluation Protocol

The canonical parcel conditions have 100 episodes per actor and speedup factor,
with base seed 12345 and pairing verified from recorded initial poses. Upright and peg final
conditions have 200 episodes per actor and factor, an indexed initial-condition
bank with base seed 73001, and 32 environments. All use 10 mm planar jitter,
no observation corruption and no action noise. Per-factor seeds add 1000 times
the zero-based index in the full ordered grid.

Pairing is established per condition from the recorded initial conditions and
configuration. Some parcel DP/DAgger conditions fail pairing and support standalone
rates. See [PAIRING_STATUS.csv](../data/manuscript_20260921/PAIRING_STATUS.csv).
Report Wilson 95% intervals and paired bootstrap differences only where pairing
is verified. Different factors and training seeds are not pooled.

## Evaluated Policies

| policy | construction | role in the evaluation |
|---|---|---|
| Expert | scripted policy following an inverse-kinematics trajectory after grasp-bank acquisition | generates the demonstrations and provides the matched reference at each speed |
| ACT | trained on each task’s demonstrations, with task-specific checkpoint selection | one policy per task in the headline comparison; two additional training runs reported individually |
| Diffusion Policy | state-based `ConditionalUnet1D`, trained separately on each task's admitted demonstrations | secondary speed-response evidence with task-dependent nominal performance |
| DAgger | multilayer-perceptron policy trained by dataset aggregation | illustrates why high-speed differences are not interpretable as temporal sensitivity when nominal success is already low |

The episode records also contain stage outcomes, hand–object relative motion,
arm joint-velocity utilization, target-tracking error, and realized contact
sets. [DIAGNOSTICS.md](DIAGNOSTICS.md) states what each measurement supports.

## Evaluating Another Policy

[POLICY_INTERFACE.md](POLICY_INTERFACE.md) defines the Python actor interface
used by the evaluator. Run a compatible policy on the released speedup grid
with

```bash
python scripts/evaluate.py --task parcel --actor your.module:YourPolicy
python scripts/evaluate.py --task upright --actor your.module:YourPolicy
python scripts/evaluate.py --task peg --actor your.module:YourPolicy
```

Retain the new policy records separately and validate their identities before
comparison. `reproduce_manuscript.py` reads only the frozen canonical bundle; it
does not analyze a newly submitted policy.
