# Benchmark Specification

ParcelStow compares an imitation learner with the expert that generated its
demonstrations as task execution speed changes. The expert and learner receive
the same initial-condition draws, task geometry, success criteria, observation,
and action interface at each tested speed. This matched evaluation measures
whether the learner preserves the expert's task-success response to execution
speed rather than whether the learner succeeds under nominal conditions alone.

## Execution-Speed Intervention

The speedup factor `r` divides the nominal durations of the manipulation phases
from lift through retreat. Acquisition and settling retain their nominal
durations. Consequently, `r=1` gives the nominal phase schedule, whereas `r=2`
halves the durations of the scaled phases. The complete cycle lasts 14.1 s at
`r=1` and 10.2 s at `r=2` because the acquisition and settling durations remain
fixed.

The demonstrations sample `r` uniformly from `[0.5, 2]`. Thus, `r=2` is the
maximum demonstrated speed and the boundary of the training support; speeds
above `r=2` test extrapolation beyond the demonstrated range. Every policy
observes `r`. [TASK_SPEC.md](TASK_SPEC.md) records the frozen phase schedule and
training distribution.

Changing `r` does not change the 80 x 55 x 40 mm parcel, its mass or friction,
the five-contact acquisition grasp, the expert's geometric path, or the
open-front receptacle. The receptacle provides 10 mm of clearance per side
along its tight axis. Evaluation also retains the same 10 mm planar start-pose
jitter and uses the same sampled start poses for the expert and each learner.

## Task Success

An episode succeeds only when the parcel is acquired with force-bearing
contacts, lifted clear of the table, reoriented, inserted to the required
depth, released, and settled inside the final position and orientation
tolerances. These predicates depend on simulated physical state. The
Ferrari–Canny margin and the other grasp measurements are diagnostics and do
not enter the success criterion.

## Evaluation Protocol

The released evaluation contains 100 episodes for each policy and speedup
factor in `{0.5, 1.0, 1.5, 2.0, 2.25, 2.5, 3.0}`. It runs 32 environments per
process with observation corruption disabled. The seed for speed index `i` is
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
python scripts/evaluate.py --actor your.module:YourPolicy \
    --rates 0.5 1.0 1.5 2.0 2.25 2.5 3.0 --episodes 100
```

Then plot its task-success curve with

```bash
python scripts/plot_envelope.py --summary outputs/eval/summary.jsonl
```
