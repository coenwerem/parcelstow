# Upright and Peg ACT Pipeline Audit

## Scope

This audit covers the ACT policies for upright placement and keyed peg
insertion on commit `bf32175a7e9aea38b4eb210e064d1f2d815805dd`, the synchronized
`main` commit from which `debug/v2-learner-comparability` was created. The
parcel-insertion v1.0.0 release remains fixed at
`39e131de5e52298607435e6f64d829600f8d748e` and is outside the intervention
scope.

The observations below describe the checked-in code, hosted artifacts, local
training logs, and current development evaluation records. Proposed
explanations remain hypotheses until a discriminating test supports them.

## Implementation Source Card

| Object | Executable Source |
| --- | --- |
| State-only ACT | `scripts/manipulation/state_act.py` and `third_party/act/transformer.py` |
| Policy execution | `ACTActor` in `scripts/manipulation/stow_runtime.py` |
| Upright demonstrations | `scripts/manipulation/run_upright_expert.py` |
| Peg demonstrations | `scripts/manipulation/run_peg_expert.py` |
| Upright training | `scripts/manipulation/run_upright_act.py` |
| Peg training | `scripts/manipulation/run_peg_act.py` |
| Upright evaluation | `scripts/manipulation/eval_upright_policies.py` |
| Peg evaluation | `scripts/manipulation/eval_peg_policies.py` |
| Upright task definition | `source/parcelstow/parcelstow/tasks/manager_based/upright_place/` |
| Peg task definition | `source/parcelstow/parcelstow/tasks/manager_based/peg_insert/` |
| Public routing | `scripts/task_registry.py` and `scripts/evaluate.py` |

Both policies use the same implementation: a 72.63-million-parameter
state-only DETRVAE with 100 action queries, hidden dimension 512, feedforward
dimension 3200, four encoder layers, seven decoder layers, eight attention
heads, dropout 0.1, and latent dimension 32. Training uses AdamW at
`1e-5`, weight decay `1e-4`, batch size 8, KL weight 10, and 2,000 epochs.
Inference uses a zero latent vector and queries a new chunk at every 50 Hz
control step. The temporal ensemble uses exponent 0.01 and combines the
predictions that cover the current step.

## Interface Source Card

The two tasks each expose a 147-dimensional state observation and a
16-dimensional joint-position action. The layouts were checked against both
task configuration files and the stored demonstration tensors.

| Observation Slice | Meaning |
| --- | --- |
| `0:51` | Robot joint position relative to the default position |
| `51:102` | Robot joint velocity relative to the default velocity |
| `102:118` | Previous 16-dimensional action |
| `118:125` | Manipulated-object position and `wxyz` orientation in the pelvis frame |
| `125:140` | Five fingertip positions in the pelvis frame |
| `140:145` | Five fingertip force magnitudes |
| `145` | Task-specific phase index plus phase fraction, divided by 13 |
| `146` | Raw speedup factor `r` |

The action uses the `CHAIN_ACTUATED` joint order. Each action is decoded as
`q_target = q_default + 0.5 * action`. Demonstration collection stores the
pre-step observation and the expert action computed from that observation.
Training and evaluation use the same ordering, normalization convention, and
action decoder. This inspection found no observation offset, action offset, or
joint-order mismatch.

The phase feature has the same numerical encoding across tasks, but the phase
names retain their task-specific meanings. The raw speedup factor remains in
the observation during acquisition and settle phases even though those phase
durations do not scale with `r`.

## Artifact and Data Source Card

| Task | Demonstrations | Checkpoint |
| --- | --- | --- |
| Upright placement | `outputs/upright/demos/expert_episodes.pt`, 315 admitted of 330 attempted, SHA-256 `2959a476b294bbd63fe76a4a7ab4e64e1a87dfadd7e146b132559b953acfbb07` | `outputs/upright/act/act_upright.pt`, SHA-256 `23792507a4a9e03b11682b28de929485211fb662c464bea10a3ed770caafbbc3` |
| Keyed peg insertion | `outputs/peg/demos/expert_episodes.pt`, 325 admitted of 330 attempted, SHA-256 `4512ae3a7b0b651faa380351ff727cdabe85ec1b132fdec0ac5f3ec0c0fff17d` | `outputs/peg/act/act_peg.pt`, SHA-256 `207132aaed13d16bcdabe674c0df7450214997cc7c0dbcc6d50e7a3a0a1a1747` |

The hashes, paths, and byte counts agree with `artifacts/manifest.json`.
Upright demonstrations sample `r` uniformly from `[0.75, 1.75]`; peg
demonstrations sample it uniformly from `[0.5, 1.0]`. Both use 10 mm planar
object-position jitter, observation corruption, seed 1, and successful full
expert episodes only.

The upright admitted demonstrations contain 63, 87, 74, and 91 episodes in
the four equal-width rate intervals from 0.75 to 1.75. Seven rejected episodes
ended in `placement_miss`, and eight ended in `tipped_after_release`. The peg
admitted demonstrations contain 67, 90, 77, and 91 episodes in the four
equal-width intervals from 0.5 to 1.0. All five rejected peg episodes ended in
`insertion_jam`.

The admitted object-position ranges cover the evaluation jitter square for
both tasks. The demonstration files do not contain indexed initial-condition
identifiers or the initial randomized robot joint state, so exact coverage of
the evaluation joint-state draws cannot be reconstructed from those files.

## Training and Selection Observations

The current scripts concatenate every admitted episode to compute observation
and action statistics. They do not reserve validation episodes. Each epoch
draws one uniformly sampled start step from every episode and trains on the
next 100 actions. Only the checkpoint after epoch 2,000 is retained. No
validation metric or simulator rollout governs checkpoint selection.

The masked L1 implementation sums zeroed padded entries and then averages over
the complete `batch x 100 x 16` tensor. A chunk with fewer valid actions
therefore contributes less reconstruction loss than a full chunk. The KL term
does not receive the same length-dependent scaling, so the effective balance
between reconstruction and KL loss changes near episode ends.

The upright training log reports a final loss of 0.03416 after 2,673 seconds.
Its post-training nominal diagnostic produced 14 successes in 50 episodes.
The peg training log reports a final loss of 0.03077 after 2,945 seconds. Its
post-training nominal diagnostic produced 43 successes in 50 episodes, but the
script then requested the nonexistent upright stage key `placed` and raised
`KeyError`. The peg checkpoint was already written; the diagnostic summary was
not appended to `results.jsonl`.

## Evaluation Observations

The current records contain 100 episodes for each actor and speed. Expert and
ACT object poses match exactly for all 800 upright pairs and all 800 peg pairs.
The peg records also contain 800 unique `(rate, initial_condition_id)` pairs
per actor. Upright records predate the explicit identifier field, but their
poses match by `(rate, episode)`.

At nominal speed, upright placement records 92 expert successes and 39 ACT
successes. ACT acquires 99 objects, with 24 transport drops, 28 placement
misses, eight post-release tips, and one acquisition failure. ACT success rises
from 39 at `r=1` to 74 at `r=1.75`, then falls to eight at `r=2` and zero at
`r=2.5`. This curve alone does not identify a cause.

At nominal speed, keyed peg insertion records 93 expert successes and 75 ACT
successes. ACT acquires every object, followed by 16 transport drops and nine
insertion jams. At every evaluated `r >= 1.5`, ACT acquisition is 0/100 even
though the first five phase durations are fixed. The checkpoint's speedup
normalization has mean 0.7391 and standard deviation 0.1379; `r=1.5` is 5.52
standard deviations above that mean. In an offline intervention on recorded
fixed-phase observations, replacing only the rate feature changed the ACT
first-action output by root-mean-square values of 0.047 at `r=1.5`, 0.123 at
`r=1.75`, 0.204 at `r=2`, and 0.352 at `r=2.5`, relative to `r=1`.

A paired simulator intervention then evaluated 40 indexed episodes at actual
`r=1.5`. The documented observation produced 0/40 acquisitions and 0/40 task
successes. Replacing only the rate feature with 1.0 during phases 0 through 4
produced 40/40 acquisitions and 27/40 task successes on the identical
initial-condition bank. The remaining failures were six transport drops and
seven alignment failures. This intervention establishes that out-of-range rate
conditioning caused the observed acquisition failures for this checkpoint and
bank. It does not explain the nominal `r=1` gap and does not support changing
the documented observation.

## Chunk and Phase Observations

One ACT chunk spans 2.0 seconds at the 50 Hz control rate. Every stored start
step in phases shorter than two seconds has a target chunk that crosses a phase
boundary. In the longer scaled phases, the fraction depends on the task-rate
distribution. For upright demonstrations, the crossing fractions for lift,
reorient, transfer, lower, and retreat are 90.2%, 74.6%, 74.6%, 96.4%, and
96.5%. For peg demonstrations, the corresponding fractions for lift,
reorient, transfer, insert, and retreat are 60.2%, 45.3%, 45.3%, 48.3%, and
72.5%. The settle phase does not cross a later boundary.

Crossing a boundary is not itself an error: the target chunk contains the
expert's later-phase actions. A defect would require evidence that the model or
temporal ensemble applies incompatible predictions after a transition. The
first diagnostic therefore measures action error around transitions before
testing a phase-aware execution variant.

## Failure Hypotheses and Discriminating Tests

| Observation | Plausible Explanation | Smallest Discriminating Test | Status |
| --- | --- | --- | --- |
| Final epoch is the only saved checkpoint | The retained model may not be the best generalizing checkpoint | Save periodic candidates, rank by held-out action error, then select on a disjoint nominal development bank | Seed-0 pilot completed; candidate task success varied by 40 points for upright and 11 points for peg |
| Padded chunks receive less L1 weight | Late phases may be underweighted and the effective KL ratio may vary with valid length | Unit-test a valid-element reduction, then compare one fixed-seed training pilot with the current objective | Verified implementation defect; effect untested |
| Peg acquisition is 0/100 for `r >= 1.5` | The raw rate feature extrapolates outside the training range during fixed acquisition phases | Hold the rate feature at 1.0 only in phases 0 through 4 while the simulator remains at `r=1.5` | Confirmed on 40 paired diagnostic episodes: acquisition changed from 0/40 to 40/40 |
| Upright success rises from `r=1` to `r=1.75` | Rate conditioning, phase duration, or contact dynamics may favor faster execution | Evaluate action error by rate and run paired nominal interventions that vary the policy rate input without changing simulator timing | Pending |
| Many 100-step targets cross phase boundaries | Temporal ensembling may retain actions inferred before a phase change | Compare error before and after transitions; reset only the diagnostic actor's ensemble at transitions if the error localizes there | Pending |
| Demonstrations retain successful expert episodes only | Failure filtering or uneven coverage may omit difficult starts | Compare admitted and rejected rate and pose strata; collect indexed demonstrations only if the stored metadata is insufficient | Rate and object-pose coverage checked; robot-state coverage unavailable |
| Training and evaluation decode the same 16 actions | A basic action-order or scale defect is unlikely | Regression tests for the observation slices and action decoder | Existing contract verified; focused tests pending |
| Peg development diagnostic was 43/50 but the current nominal bank is 75/100 | Performance varies across indexed banks or the earlier asynchronous bank | Use one fixed indexed development bank for every checkpoint and reserve a disjoint final bank | Fixed development bank implemented and verified; final bank remains untouched |

## Seed-0 Bounded Pilot

The seed-0 training runs completed under the committed protocol. Upright
training took 1,984 seconds, and peg training took 2,030 seconds. Both runs
used the documented episode split, training-only normalization, valid-element
reconstruction loss, and fixed validation starts. The selector verified 100
unique condition identifiers, exact summary counts, and one initial-condition
bank across every candidate and expert run. The common bank hash was
`592eaf90642778f173a28db8a088869ee068ba9826eb3fdd9cbc6e3da5ee47c6`.

| Task | Epoch | Validation Chunk L1 | Development Success | Expert Success | Paired Outcomes: Both / Expert Only / ACT Only / Neither |
| --- | ---: | ---: | ---: | ---: | ---: |
| Upright placement | 2000 | 0.022889 | 98/100 | 97/100 | 95 / 2 / 3 / 0 |
| Keyed peg insertion | 1400 | 0.027234 | 93/100 | 95/100 | 88 / 7 / 5 / 0 |

The selected upright checkpoint had SHA-256
`94888be60215900c6aff75d73ae8d4fd15535691ff16a7fcb545d34556e22fa7`.
Its two failures were one acquisition failure and one placement miss. The
selected peg checkpoint had SHA-256
`c9c1e8cd08dae4d5501dbbeceae933e69e2037633f31007e4257914d346a911d`.
Its seven failures were two acquisition failures, four transport drops, and
one insertion jam.

| Task | Retained Epoch and Success Counts |
| --- | --- |
| Upright placement | 1200: 85; 1300: 92; 1700: 83; 1900: 58; 2000: 98 |
| Keyed peg insertion | 1400: 93; 1600: 87; 1800: 87; 1900: 82; 2000: 88 |

The current public development records report 39/100 for the upright ACT
checkpoint and 75/100 for the peg ACT checkpoint on their existing evaluation
bank. The pilot point estimates are higher, but the pilot changed the loss,
data partition, normalization source, and checkpoint-selection procedure
together. These runs therefore do not isolate the contribution of one change.

The pilot bank governed checkpoint selection, so its paired outcomes are model
development evidence rather than final estimates. Only policy seed 0 has been
run, the nominal comparison criterion still requires author approval, and the
new final bank has not been generated or inspected. The complete compact
evidence, including file and checkpoint hashes, is stored in
`results/upright_seed0_pilot.json` and `results/peg_seed0_pilot.json`.

## Audit Conclusion

The audit identified and corrected two implementation defects in the training
path: reconstruction loss used the complete padded tensor as its denominator,
and the peg diagnostic summary requested the wrong stage. The bounded pilot
also established that checkpoint task success can differ substantially despite
similar validation loss. The fixed-phase intervention isolated out-of-range
rate conditioning as the cause of the tested peg acquisition failures at
`r=1.5`. No evidence supports changing task geometry, expert trajectories,
success predicates, action semantics, or observation ordering.

The evidence supports retaining ACT and the corrected training and selection
procedure. The next scientific step is to review the nominal comparison
criterion, then train policy seeds 1 and 2 as replication instances. Another
learned-policy method is not warranted before that replication and the
predeclared final evaluation are complete.
