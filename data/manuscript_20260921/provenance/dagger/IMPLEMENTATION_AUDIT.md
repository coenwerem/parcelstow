# DAgger implementation audit, 2026-09-20

Outcome: **PASS for the specified audit.** No implementation error was found
that invalidates the completed upright or peg DAgger run. The protected
measurements are valid negative results for these two frozen training runs.
They do not establish a general limitation of DAgger, and neither policy is a
strong nominal baseline.

Record validation passed first, after the user resolved the validator's
incorrect adjacent-stage assumption. `RESOLUTION_20260920.md` supersedes that
part of `STOP_REPORT.md`; `validation.json` contains the complete passing
result and four valid peg predicate regression cases. Neither the protocol
nor any monitor, checkpoint, summary, or protected record was changed.

The audit used the current training driver and shared runtime, the frozen
demonstrations and aggregates, the final checkpoints, and round-zero through
round-three checkpoints solely to check which policy generated each
aggregation round. Round zero was also evaluated on stored demonstration
states to distinguish fit error from rollout error. No checkpoint was
substituted for the predeclared final policy. `AUDIT_SHA256SUMS` binds the
inspected sources and diagnostic evidence.

The training entry is `scripts/manipulation/run_stow_distill.py`. Evaluation
dispatches through `scripts/evaluate.py` to `eval_upright_policies.py` or
`eval_peg_policies.py`. Both use the `Student` and `DaggerActor` classes and
`run_episodes` in `scripts/manipulation/stow_runtime.py`. Task-specific
experts and monitors are selected through `task_runtime`. ACT and Diffusion
Policy use the same observation/action interface and episode runner; the
released parcel DAgger uses the same MLP, fit loop, and expert-label timing.

| Audit item | Evidence and outcome |
|---|---|
| Observation layout | Exactly 147 float32 elements in the order below, from demonstrations through aggregation and evaluation. The training log, task configs, runtime term dimensions, stored data, and diagnostic agree. |
| Actions | Exactly 16 normalized absolute joint-position targets. `q_target = q_default + 0.5 * action`; expert inversion is `2 * (q_target - q_default)`. No delta, velocity, torque, or joint-limit normalization is used. Runtime conversion error is zero in both diagnostics. |
| Phase and rate | Observation 145 is `(k+f)/13`; observation 146 is the sampled execution rate. The task-specific 13-phase schedule scales only its designated phases. Every demonstration rate equals its record exactly; reconstructed phase error is at most 2.98e-7. Live phase/rate errors are zero. |
| Expert-label timing | The task expert is called once at the pre-step state; its action labels the observation stored before `env.step`. Its parallel integral correction follows the visited joint states. Replay of every saved state agrees within 9.66e-6 action units. |
| Reset behavior | The episode runner applies indexed reset states, recomputes observations, clears the monitor and expert integrator, and then queries the stateless MLP. Last-action observations and expert corrections are zero after reset. Initial and automatic resets were checked. |
| Checkpoint loading | Final files match the handoff hashes and contain exactly the round-four tensors. Strict loading into the actual 147→512→256→128→16 ELU MLP succeeds. Live GPU actions agree with offline CPU loading within 7.63e-6 action units. |
| Normalization and fit | Each checkpoint's mean and standard deviation exactly match its training inputs, with the implemented 1e-3 standard-deviation floor. The fit uses Adam, action MSE, 40 epochs, batch 4096, and a reinitialized model each round. |
| Aggregation | Each aggregate begins with the complete, byte-equivalent demonstration tensors, then four sets of 100 episodes. Every executed clean action matches the preceding checkpoint within 7.68e-6 after accounting for observation clipping. Measured exploration-noise standard deviations range from 0.09973 to 0.10008. |
| Prediction error | Per-joint errors for all 13 phases are saved in demonstration_prediction_error.csv and stored_data_audit.json. Final mean joint-target RMSE is 0.0166006 rad upright and 0.0208668 rad peg on demonstration states. |
| Closed-loop divergence | With the frozen final policies at development seed 42001, rate 1, both tasks exceed the predeclared diagnostic threshold of 0.05 rad in at least one joint on the first action after reset. Details follow below. |

The complete observation order is:

| Zero-based slice | Content | Units / transformation |
|---|---|---|
| 0:51 | All 51 robot joint positions in articulation order | radians relative to defaults, clipped to [-10,10] |
| 51:102 | Same 51 joint velocities | rad/s relative to defaults, clipped to [-50,50] |
| 102:118 | Previous 16-dimensional action | action units, clipped to [-10,10] |
| 118:125 | Object position and quaternion | pelvis-frame metres, then wxyz quaternion; term clipped to [-5,5] |
| 125:140 | Five distal-phalanx positions | pelvis-frame metres, xyz for thumb/index/middle/ring/pinky |
| 140:145 | Five net contact-force magnitudes | thumb/index/middle/ring/pinky, divided by 10 N and capped at 5 |
| 145 | Task phase | `(k+f)/13` |
| 146 | Task rate | dimensionless execution rate |

`diagnostics_20260920/observation_schema.csv` enumerates every index and joint
name. The action order is waist yaw, roll, pitch; right shoulder pitch, roll,
yaw; right elbow; right wrist roll, pitch, yaw; right thumb CMC roll and
pitch; then right index, middle, ring, and pinky MCP pitch. The schema uses
the exact simulator identifiers. The live actuator uses this order with
`preserve_order=True`.

Actions are not restricted to [-1,1]. The MLP has a linear output layer and
the joint-position action term has no action clip. Only the previous-action
observation is clipped to [-10,10]. The diagnostic checks distinguish this
observation clipping from command clipping; no extra output clipping was
introduced into the audit.

The PLAY configurations disable observation corruption when the observation
manager is created. Consequently, `EnvSwitches.saved_noise` contains `None`
for every term, and the later aggregation `corrupt=True` argument restores
no noise. This is also the released parcel PLAY behavior. Historical
demonstration metadata records the requested corruption flag, not effective
noise. Exact expert-label replay from the saved joint observations supports
the effective-noise interpretation. This metadata naming deserves clarification
in future code, but does not violate this frozen protocol's specified action
exploration or protected no-corruption evaluation.

The training demonstration and added-state counts are:

| Task | Demonstrations | Demonstration samples | Added aggregation episodes | Added samples | Final samples |
|---|---:|---:|---:|---:|---:|
| Upright | 315 | 297,985 | 400 | 366,897 | 664,882 |
| Peg | 325 | 475,758 | 400 | 470,989 | 946,747 |

The actual demonstration rate intervals are [0.7550781,1.7436700] and
[0.5019096,0.9968350]; aggregation intervals are [0.7509331,1.7481087] and
[0.5014057,0.9994782], respectively. All lie within their declared supports.
There are 100 complete episodes per aggregation round. Half of the parallel
environments receive noise; their share of completed episodes can differ
from 50 because episode durations differ. The first 32 episode assignments
and the saved action residuals agree with noisy environments 0–15 and clean
environments 16–31.

Expert replay used the actual task expert code, the verified joint order and
default positions, recorded initial offsets, saved rates, measured joint
states, and a reset integral correction. All 1,611,629 saved state/action
pairs were checked, counting each aggregate's demonstration prefix once.
The largest errors were:

| Task | Demonstrations | Added aggregation states |
|---|---:|---:|
| Upright | 7.03e-6 | 9.66e-6 |
| Peg | 6.68e-6 | 5.96e-6 |

These differences are in action units and are consistent with CPU/GPU
floating-point differences. No pair exceeded 1e-4. The stored labels match
the current experts despite older Git identifiers in demonstration metadata;
those identifiers alone do not identify a complete historical working tree.

Two bounded development diagnostics used 32 environments, seed 42001, rate 1,
10 mm jitter, no corruption, and no action noise. They ran one nominal cycle
plus three steps: 1,078 steps upright and 1,178 peg. The 100-entry development
initial-condition bank supports indexed resets; these diagnostics did not
run 100 completed evaluation episodes or write new success records. Both
processes exited zero. Their traces remain separate from all protected files.

For the first action after reset, the largest same-state student–expert
joint-target discrepancy across the initial 32 environments was 0.08118 rad
for upright (environment 10, right shoulder pitch) and 0.11351 rad for peg
(environment 21, right wrist yaw). The expert action is zero in PARK.
The threshold was defined before these diagnostics as 0.05 rad maximum
absolute joint-target discrepancy, with 0.02 and 0.1 rad also recorded.
Upright exceeds 0.1 rad on step 1: environment 30 differs by 2.30454 rad in
right wrist pitch. Peg exceeds all three thresholds on step 0.

The large upright step-1 error coincides with passive left-leg velocity
inputs far from the aggregate mean: left hip pitch is approximately -143
standard deviations and left knee approximately +123. Those velocities are
the simulator observations; recomputing the full observation yields exactly
the actor input. The first-action errors and subsequent distribution shift
are consistent with policy sensitivity outside its well-fitted states. No
causal feature ablation was performed, so the passive-joint inputs are not
claimed to be the sole cause of failure.

All initial and automatic reset events observed in these diagnostics had
zero phase, rate 1, zero last-action input, and zero expert correction. The
MLP has no hidden state or action queue to clear. The live traces and
offline loading check also exclude a stale model, wrong checkpoint, wrong
normalizer, or action conversion as the explanation for this divergence.

The audit is complete for the requested checks. It is not a proof that every
possible implementation defect is absent. The final checkpoint is retained;
no replacement run is proposed as a condition of reporting its low success.
Scientific comparisons were computed only after `implementation_audit.json`
and `validation.json` both recorded PASS.
