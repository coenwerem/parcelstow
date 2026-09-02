# Benchmark Specification

ParcelStow compares an imitation learner with the expert that generated its
demonstrations as task execution speed changes. The expert and learner receive
the same initial-condition draws, task geometry, success criteria, observation,
and action interface at each tested speed. This matched evaluation measures
whether the learner preserves the expert's task-success response to execution
speed rather than whether the learner succeeds under nominal conditions alone.
Parcel insertion supplies the primary nominally matched comparison reported in
arXiv:2609.01453. Upright placement and keyed peg insertion retain the same
evaluation procedure but their released ACT checkpoints do not match expert
success at nominal speed.

| Task | Task Specification | Demonstrated `r` | Evaluation Grid |
|---|---|---:|---|
| Parcel insertion | [TASK_SPEC.md](TASK_SPEC.md) | `[0.5, 2.0]` | `{0.5, 1, 1.5, 2, 2.25, 2.5, 3}` |
| Upright placement | [TASK_SPEC_UPRIGHT.md](TASK_SPEC_UPRIGHT.md) | `[0.75, 1.75]` | `{0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5}` |
| Keyed peg insertion | [TASK_SPEC_PEG.md](TASK_SPEC_PEG.md) | `[0.5, 1.0]` | `{0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5}` |

## Execution-Speed Intervention

The speedup factor `r` divides the nominal duration of phases marked as scaled
in the selected task specification. Acquisition and settling retain fixed
durations in all three released tasks. Consequently, `r=1` gives a task's
nominal phase schedule, whereas `r=2` halves only that task's scaled phases.
For parcel insertion, the complete cycle lasts 14.1 s at `r=1` and 10.2 s at
`r=2`.

Each task samples demonstrations uniformly from the range in the table above.
Evaluation outside that range tests speed extrapolation. Every policy observes
`r`; the task identity is selected by the public command and is not appended to
the observation.

Changing `r` does not change a task's geometry, mass, friction, expert path,
initial-condition distribution, observation, action, or success predicates.
The expert and learner use the same sampled initial conditions at each speed.

## Task Success

Each task specification defines task-specific stage outcomes, terminal failure
reasons, and physical success predicates. Analytical grasp scores do not enter
task success. Parcel requires insertion and settling in its receptacle;
upright requires a released, stable upright pose inside the target region; peg
requires insertion and settling inside the square pocket.

## Evaluation Protocol

The released evaluation contains 100 episodes for each policy and speedup
factor in the task-specific grids above. It runs 32 environments per process
with observation corruption disabled. The seed for speed index `i` is
`12345 + 1000*i`; reusing that seed for every policy pairs their initial
conditions. Reported task-success intervals are Wilson 95% intervals. The
expert–learner difference at `r=2` uses a paired bootstrap over the shared
initial-condition draws.

## Evaluated Policies

| policy | construction | role in the evaluation |
|---|---|---|
| Expert | scripted policy following an inverse-kinematics trajectory after grasp-bank acquisition | generates the demonstrations and provides the matched reference at each speed |
| ACT-A/B/C | state-based Action Chunking with Transformers policies trained on the same 297 expert demonstrations | test sensitivity to parameter initialization; ACT-A matches the expert's observed success at `r=1` |
| Diffusion Policy | state-based `ConditionalUnet1D` policy trained on the same demonstrations | evaluates a second imitation-learning architecture that does not reach nominal expert success |
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

Then plot its task-success curve with

```bash
python scripts/reproduce.py all-tasks
```
