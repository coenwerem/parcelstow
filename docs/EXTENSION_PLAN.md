# ParcelStow arXiv-v2 Extension Plan

The arXiv-v1 study is frozen at tag `v1.0.0` (commit `d2f7622`). All
extension work happens on the branch `extension/arxiv-v2`, which starts
at that tag. This note records the boundary between the frozen v1
release and the extension: which artifacts must not change, which
evaluation machinery the new tasks reuse, which machinery stays
ParcelStow-specific, the candidate tasks for the arXiv-v2 evaluation,
the gates each task must pass before full learner training, and the
first infrastructure increment.

The extension question is whether the expert-learner separation across
task execution speeds observed in ParcelStow also occurs in
manipulation tasks with different physical demands. The manuscript
scopes its v1 claim to one task and states that other tasks,
architectures, observation modalities, and physical systems remain to
be evaluated; the extension supplies the task evidence, in simulation.

## Frozen v1 Artifacts

The following artifacts reproduce the v1 numbers and must not change
on the extension branch. Nothing below may be rewritten, renamed,
regenerated in place, or edited for compatibility with new tasks.

| artifact | location |
|---|---|
| ParcelStow task definition | `source/parcelstow/parcelstow/tasks/manager_based/parcel_stow/` (geometry constants, phase schedule, stage predicates, failure reasons, observation and action interface), frozen by `docs/TASK_SPEC.md` |
| released episode records | `data/records/**` (episode records, `eval_summary.jsonl`, replication records) |
| frozen derived analyses | `experiments/paper/results/**` |
| checkpoints and demonstrations | external artifacts inventoried in `artifacts/manifest.json` (names, paths, sizes, sha256) |
| frozen construction inputs | `assets/parcel_stow_geometry.json`, `assets/parcel_stow_trajectory.json`, `assets/gdf_bank_parcel.json`, `assets/provenance/**` |
| v1 plots and reproduction commands | `media/operating_envelope.*`, `media/stages_vs_rate.*`, `media/terminal_states_r2.png`, the `scripts/reproduce.py` targets `envelope`, `stages`, `certificate`, `certificate-oos`, `expert-ceiling`, and the commands in `docs/REPRODUCING_THE_PAPER.md` |
| legacy schema and API identifiers | the record fields written by `stow_runtime.run_episodes` and `stow_runtime.summarize` (`task_rate`, `policy`, `episode`, `env`, `task_duration_s`, the stage booleans, `failure_reason`, the `config` stamp), the summary fields added by `eval_stow_policies.py` (`policy`, `rate`, `cycle_time_s`, `seed`), the gym id `ParcelStow-L6-Distill-Play-v0`, the actor interface (`name`, `reset(ids, obs)`, `act(obs) -> (action, q_target)`), and the `--actor` argument of `scripts/evaluate.py` |

Two schema facts constrain the extension. First, the per-episode
records name the evaluated policy in the field `policy`; no record
carries a field named `actor`, and `actor` appears only in the CLI
argument, the loader `stow_runtime.load_actor`, and transient keys of
one analyzer. Second, no record carries a top-level task identifier;
the only task name sits inside the configuration stamp at
`config.task`, and the released records store the pre-release id
`G1Locomanip-L6-ParcelStow-Distill-Play-v0` there. Multi-task records
therefore need a new top-level identifier, added without touching the
released files (see First Infrastructure Increment).

## Shared Evaluation Machinery

The matched expert-learner evaluation is carried by six pieces of
machinery. Each can serve additional tasks; the table records where
each piece lives today and what currently ties it to ParcelStow.

| machinery | location | ParcelStow coupling |
|---|---|---|
| speedup-factor scheduling | phase table and duration law in `geometry.py:52-86`; runtime clock, per-env rate buffer, rate sampling event, and completion termination in `mdp/task_clock.py` | `task_clock` imports the schedule constants (`NOMINAL_DURATIONS`, `RATE_SCALED`, `N_PHASES`) from the ParcelStow geometry module; the clock logic itself reads nothing else from the task |
| matched initial-condition draws | per-speed seed law `eval_seed + 1000 * rate_index` in `eval_stow_policies.py:83`, applied to the global torch RNG immediately before `env.reset()` in `stow_runtime.py:311-320`; draw identity across policies is asserted by `tests/test_relative_handoff.py` and re-verified by the `draws_identical_to_expert` columns of the summarizers | none; the law is task-independent |
| episode execution and record writing | `stow_runtime.run_episodes`, `summarize`, `write_jsonl`, `load_actor`, `EnvSwitches` | module-scope imports of the ParcelStow geometry, task clock, monitor, and scripted expert; hardcoded stage-key list in `summarize` |
| Wilson intervals and paired comparisons | `geometry.wilson` (`geometry.py:513-521`); `paired_gap_ci` in `scripts/plot_envelope.py:51-68`, pairing episodes by the `episode` field | none in the statistics; `wilson` lives in the ParcelStow geometry module by placement only |
| policy loading and the actor interface | `stow_runtime.load_actor` (built-in names plus `module.path:ClassName`), interface contract in `docs/POLICY_INTERFACE.md` | the observation width (147) and action width (16) are ParcelStow-frozen; the loading mechanism is not |
| task-success plotting and reproduction | `scripts/plot_envelope.py`, `scripts/reproduce.py` | actor-name constants, file-name conventions, and the ParcelStow stage list; the plotting and bootstrap code read only `policy`, `task_rate`, `task_success`, `episode` |

The four analyzer scripts (`analyze_stow_certificate.py`,
`analyze_certificate_oos.py`, `analyze_expert_ceiling.py`,
`analyze_rate_conditioned_margin.py`) contain task-independent
statistics coupled to ParcelStow only through field-name and
actor-name constants. `mdp/ferrari_canny.py` (the Ferrari-Canny margin
scorer) and `mdp/guards.py` (state-validity termination) import
nothing from the task and are reusable as they stand.

## ParcelStow-Specific Machinery

The following machinery encodes the ParcelStow task and stays
task-specific. New tasks implement their own counterparts under their
own names; none of these objects may be renamed or generalized in
place.

- Parcel and receptacle geometry: `geometry.StowGeometry`, the frozen
  parameters in `assets/parcel_stow_geometry.json`, the object path
  `geometry.object_pose`, and the parcel constants
  (`PARCEL_EXTENTS`, `PARCEL_MASS`, `PARCEL_FRICTION`,
  `PARCEL_START`).
- The acquisition, reorientation, insertion, release, and settling
  phases: the 14-entry phase table in `geometry.py:52-67` and the
  scripted expert's phase-target dispatch in
  `scripts/manipulation/parcel_stow_expert.py`.
- ParcelStow success predicates: the latched stage markers and the
  `task_success` conjunction in `mdp/metrics.py` (thresholds mirrored
  from `docs/TASK_SPEC.md` section 8).
- ParcelStow failure reasons: the ten-entry cascade in
  `mdp/metrics.py:424-463`, including the slip split at receptacle
  contact.
- The relative-motion handoff: `stow_relative.py`,
  `stow_relative_controller.py`, and `stow_relative_handoff.py`,
  which replay the scripted expert's command path relative to the
  policy's hand pose at handoff; the reference path, trigger, and
  endpoints are keyed to ParcelStow phases and the receptacle.
- Force-closure measurements tied to the realized grasp and contact
  model: the contact-set construction in
  `StowMonitor.contact_geometry` samples the five L6 distal segments
  against the parcel box, so the recorded contact sets, and every
  margin computed from them, presuppose that hand and that object.
  The scorer itself (`mdp/ferrari_canny.py`) is task-independent.

`mdp/contacts.py` and the monitor's body-name constants are specific
to the L6 hand rather than to the ParcelStow task; they transfer to
any G1-plus-L6 task unchanged and to other end-effectors only with
new body names.

## Multi-Task Boundary

Modules that can serve several tasks without changing v1 behavior:

- the runtime clock in `mdp/task_clock.py`, once it receives its
  schedule from the task rather than from the ParcelStow geometry
  import;
- the seed law, `run_episodes` loop, record writing, and actor
  loading in `stow_runtime.py`, once the module-scope ParcelStow
  imports become per-task inputs;
- `wilson`, `paired_gap_ci`, and the analyzer statistics;
- `mdp/ferrari_canny.py` and `mdp/guards.py` as they stand.

Modules that remain task-specific by design: each task's geometry,
phase table, stage predicates, failure reasons, scripted expert, and
monitor. New tasks define their own stage names and failure reasons;
the shared machinery reads stage keys from the records it is given
rather than from a universal stage vocabulary. Generality obtained by
renaming ParcelStow objects, or by forcing another task's contact
sequence into the ParcelStow phase names, is prohibited.

## Candidate Tasks

Each candidate below is specified against the same template: object
and geometry, contact sequence, expert construction, speed-scaled and
fixed phases, demonstrated speed range, success predicate, stage
outcomes, initial-condition distribution, the evidence it adds beyond
ParcelStow, xArm7 feasibility, and cost. All simulated candidates use
the fixed-base G1 with the right RealHand L6 in Isaac Lab, the
embodiment the repository already supports, so that task demands vary
while the embodiment, observation grammar, and control interface stay
fixed. Demonstrated speed ranges follow the v1 procedure: an
expert-only calibration sweep fixes the range where the expert
succeeds in at least 0.9 of episodes, before any learner runs. Grasp
banks for the new objects are synthesized with the local FRoGGeR
checkout and ship as frozen construction inputs with provenance
records, following the v1 protocol in `docs/ASSET_PROVENANCE.md`.

### Parcel Reorientation and Receptacle Insertion (Retained)

The v1 task continues unchanged as the anchor of the cross-task
comparison. Its released records, demonstrations, checkpoints, and
curves are reused; nothing is re-collected. Its role in v2 is the
reference column against which the new tasks' expert-learner
separations are compared.

### Keyed-Peg Insertion

- Object: the upright task's 55 x 55 x 180 mm cuboid at the v1
  parcel mass of 0.120 kg, one object across both new tasks (the
  original 25 mm sketch fell to the synthesis aperture floor: no
  seated force-closed grasp below 55 mm width), resting on its side;
  a table-mounted pocket block with a square cavity providing 3 mm of
  clearance per side, a lead-in funnel at the mouth, and the pocket
  top 120 mm above the table. The square cross-section admits four
  yaw solutions (C4 symmetry about the peg axis).
- Contact sequence: grasp the shaft, lift, reorient 90 degrees to
  vertical, transport to the pocket axis, lower into the guided
  cavity, release inside it, retreat up and back, settle.
- Expert construction: as v1 in method; the shared grasp-bank entry,
  task-space object waypoints along the frozen path, damped least
  squares IK at fixed knots, joint-target interpolation between
  knots, plus a raised approach via and the measured realized-grasp
  compensation (the tenth increment).
- Speed-scaled phases: lift, reorient, transfer, insert, insert
  dwell, release, retreat. Fixed phases: park, approach, pregrasp
  dwell, close, grasp dwell, settle.
- Demonstrated speed range: from the expert calibration sweep; the
  tighter clearance narrowed the range to [0.5, 1.0] against the
  upright's [0.75, 1.75] (the eleventh increment), measured rather
  than assumed.
- Success predicate: peg base past a 40 mm depth threshold, all four
  base corners inside the cavity cross-section, released, and settled
  within a 5 deg final tilt tolerance; the cross-section encodes the
  yaw tolerance the clearance derives (2 c / a = 6.2 deg).
- Stage outcomes: acquired, lifted_clear, reoriented_upright,
  aligned, inserted, released, settled.
- Failure reasons: acquisition_failure, dropped_during_transport,
  alignment_failure, insertion_jam, timeout, other.
- Initial conditions: planar start jitter in x and y, as v1; no yaw
  jitter in the first version.
- Evidence added: the tolerance-to-clearance ratio is roughly an
  order of magnitude tighter than ParcelStow's 10 mm per side, so
  success is bound by arrival alignment precision at the mouth and by
  wedge-mediated jamming inside the cavity rather than by a wide
  entrance. Whether the v1 separation persists when the accuracy
  demand rather than the transport dynamics binds is exactly the
  cross-task question.
- xArm7 feasibility: high; an insertion scene already exists for the
  xArm7-L6 MuJoCo arena, and the hardware fixture is one machined
  block.
- Cost: one new environment configuration, geometry module, grasp
  bank entry, and IK trajectory; the phase clock, monitor pattern,
  drivers, and statistics are reused. Demonstrations under 1 h, ACT
  training about 45 min, evaluation about 2 h per policy at v1
  episode counts.

### Upright Placement

- Object: a tall rigid cuboid, 55 x 55 x 180 mm at the v1 parcel mass
  of 0.120 kg, starting on its side; a marked circular target region
  on the table. No receptacle. The 55 mm width is the aperture floor
  the grasp synthesis established (no seated force-closed grasp at 40
  or 50 mm), and the 180 mm length admits the end-shifted grasp the
  kinematic probe requires.
- Contact sequence: grasp the shaft, lift, reorient 90 degrees to
  vertical, transport, lower to the surface, release, settle with
  tipping possible.
- Expert construction: as v1 in method; grasp bank on the cuboid,
  object waypoints ending in a vertical placement, IK at fixed
  knots.
- Speed-scaled phases: lift, reorient, transport, lower, release,
  retreat. Fixed phases: the acquisition phases and an extended
  settle window (about 1.0 s) so that tipping resolves inside the
  episode.
- Demonstrated speed range: from the expert calibration sweep.
- Success predicate: object base center inside the target region,
  tilt from vertical at most 5 degrees at the end of the settle
  window, and linear and angular speeds below the v1 settling
  thresholds.
- Stage outcomes: acquired, lifted_clear, reoriented_upright,
  placed, released, settled.
- Failure reasons: acquisition_failure, dropped_during_transport,
  placement_miss, tipped_after_release, timeout.
- Initial conditions: planar start jitter in x and y, as v1.
- Evidence added: the terminal predicate is quasi-static stability
  of a slender object rather than geometric containment, and the
  only environment contact is the table. Faster execution raises the
  residual object velocity at release directly, so this task
  isolates release-transient sensitivity from the insertion-contact
  sensitivity that dominates ParcelStow failures. A separation here
  cannot be jam-mediated.
- xArm7 feasibility: highest of the three; simulation and hardware
  need only the object and a marked pad.
- Cost: lowest of the three; no receptacle geometry, and the
  reorientation path reuses the v1 waypoint pattern.

### Drawer Opening

- Object: a table-mounted drawer with one prismatic joint, travel
  150 mm, fixed joint friction and damping, and a bar handle sized
  for the L6 hand.
- Contact sequence: approach, close the hand around the handle,
  pull along the prismatic axis under sustained hand-handle contact,
  release, retreat; the drawer must remain open through settling.
- Expert construction: hand closure on the handle from a bank entry
  or a scripted wrap, then arm waypoints along the drawer axis; no
  free transport of a grasped rigid body occurs.
- Speed-scaled phases: pull, release, retreat. Fixed phases: park,
  approach, pregrasp dwell, close, grasp dwell, settle.
- Demonstrated speed range: from the expert calibration sweep;
  faster pulls raise the required interaction force against the
  joint damping and inertia.
- Success predicate: drawer displacement at least 120 mm at the end
  of the settle window with the hand clear of the handle.
- Stage outcomes: handle_grasped, pull_started, opened,
  handle_released, settled_open.
- Failure reasons: grasp_failure, handle_slip, partial_open,
  timeout.
- Initial conditions: planar jitter of the drawer-unit pose, and
  optionally of the initial drawer displacement.
- Evidence added: the manipulated object is never a free rigid body
  in the hand; the environment imposes a kinematic constraint, and
  the interaction is force-mediated and sustained. ParcelStow
  contains no sustained hand-environment contact during the scaled
  phases, so this task tests the separation in a contact regime the
  v1 study does not reach. The slip diagnostics become hand-handle
  relative motion plus joint displacement.
- xArm7 feasibility: good; drawer fixtures are standard hardware.
- Cost: highest of the three; an articulated asset and new monitor
  logic for joint displacement and handle contact, with no
  reorientation trajectory.

### Candidates Considered and Not Recommended for the Initial Suite

A hinged lid or door duplicates the drawer's sustained-contact
constrained-motion regime at higher asset cost; stacking a second
cuboid duplicates the upright-placement stability regime with an
added support-surface confound. Both remain available if a fourth
regime is needed. The bimanual half-humanoid (G1 with both L6 hands;
the description already ships in `assets/g1_l6/`) has no task
software anywhere in the laboratory's checkouts and stays out of the
committed plan.

## Recommended Initial Suite

Three additional simulation tasks with distinct contact requirements,
plus the retained anchor:

1. ParcelStow (retained, frozen): reorientation plus insertion at
   10 mm clearance; failures concentrate at insertion.
2. Keyed-peg insertion: tight-clearance containment; terminal
   alignment precision binds.
3. Upright placement: terminal stability; release transients bind;
   no receptacle contact.
4. Drawer opening: sustained constrained contact; interaction force
   binds.

Implementation order: upright placement first (lowest cost, fastest
gate turnaround), then keyed-peg insertion, then drawer opening.
Task count beyond these three requires a regime the suite does not
already cover, not more instances of a covered regime.

Hardware is out of scope for arXiv-v2; the evaluation is
simulation-only on the G1-L6 embodiment. The per-candidate xArm7
notes record that each recommended task transfers to the 7-DoF xArm7
with the right RealHand L6 without redesign, so a later revision can
add hardware; no hardware work, asset conversion, or instrumentation
enters the v2 plan.

## Experimental Gates

A task enters full learner training only after passing, in order:

- Gate A, task implementation. Geometry and success predicates are
  deterministic and covered by pure tests in the pattern of
  `tests/test_geometry_pure.py`; changing the speedup factor r
  changes phase timing only, with the geometric path a function of
  phase index and in-phase fraction alone, asserted by a
  rate-invariance test; task-specific stages latch and record
  correctly in short simulator-backed runs.
- Gate B, expert feasibility. The scripted expert reaches at least
  0.9 success at nominal speed over at least 64 episodes; success
  stays meaningful across several demonstrated speeds so that a
  range with expert success at or above 0.9 exists; failures at
  higher speeds trace to the task physics, not to an implementation
  defect; arm joint-velocity utilization stays well below saturation
  and no timeout artifact produces the appearance of temporal
  sensitivity.
- Gate C, learner pilot. One ACT policy trained with the v1
  hyperparameters reaches nominal success close enough to the expert
  (overlapping Wilson intervals at r = 1) to permit a matched
  interpretation. A condition that does not reach nominal parity is
  reported as a secondary architecture or training result, not used
  for the central matched claim.
- Gate D, full evaluation. Demonstrations, the speedup grid, and the
  evaluation seeds are frozen before the final learner comparison;
  expert and learner use identical initial-condition draws at each
  speed under the v1 seed law; task-success curves, stage outcomes,
  and Wilson intervals are produced from released records by public
  commands.

Demonstrated speed ranges, speedup grids, and episode counts are
fixed at Gates B and D per task, never shared across tasks by
default.

## Infrastructure Increments

The smallest shared change the audit justifies is a task identifier
in newly generated records, plus tests that pin the released schemas.

- `stow_runtime.run_episodes` adds a top-level `task` field holding
  the gym id (the existing `TASK` constant) to every newly written
  episode record, and `stow_runtime.summarize` propagates the field
  into summary rows when the input records carry it. Released
  records, which lack the field, summarize exactly as before.
- A new pure test file asserts that the released
  `data/records/eval_summary.jsonl` and episode records still parse
  with the v1 fields (`policy`, `task_rate`, `task_success`,
  `episode`, the stage keys, the Wilson blocks) and that the new
  field appears in newly written records without displacing any v1
  field.

Every v1 command, output, record file, frozen constant, and media
name is preserved. No demonstration collection, no task skeleton, and
no bimanual work happens in these increments. The second increment,
implemented after this note was reviewed, separates the task-schedule
input of `mdp/task_clock.py` from the ParcelStow geometry import:
`parcelstow/phase_schedule.py` defines a task-level `PhaseSchedule`,
and the ParcelStow mdp package binds its frozen `geometry.PHASES` to
`task_clock.SCHEDULE` at import time, leaving the environment's
behavior unchanged. A shared framework beyond that adapter waits
until two concrete tasks exist to support it.

The third increment adds the upright placement skeleton,
`tasks/manager_based/upright_place/geometry.py`: the phase table, the
deterministic object path, the stage and failure vocabulary, and the
success predicates, covered by `tests/test_upright_geometry_pure.py`
through the shared `PhaseSchedule`. No environment, scripted expert,
or gym registration exists yet.

The fourth and later increments add the kinematic probe and the full
upright placement implementation; the paragraphs below record each
with its measured evidence.

The fourth increment adds the kinematic probe,
`scripts/manipulation/probe_upright_geometry.py`, the upright analog
of the v1 geometry probe: DLS IK over the manipulation knot list with
the `ChainIK` solver, ranked by joint-limit margin, with the grasp
hypothesis derived from the frozen v1 acquisition hand pose. Four
probe passes on 2026-09-01 (reports under `outputs/probe/`) found
that a shaft-centered grasp saturates the waist roll while lowering
(one feasible candidate in 132, minimum margin 0.001), that goal-yaw
offsets drive the wrist yaw to its limit during reorientation, and
that moving the grasp point 50 mm toward the future top end of the
shaft resolves both.

The fifth increment carries the geometry through grasp synthesis.
FRoGGeR (the local checkout at the v1 provenance commit, the v1
scene arguments) returned no seated force-closed grasp for a cuboid
of 40 or 50 mm width and three grasps at the v1-proven 55 mm, so the
object extents revised to 55 x 55 x 180 mm on kinematic criteria
alone, the 180 mm length admitting the end-shifted grasp region; the
synthesis on the exact 180 mm mesh then placed the five-contact
grasp at +46 to +91 mm along the shaft (centroid +72 mm) on its own,
with l_star 0.0034 against the v1 parcel's 0.0036. The provenance
ships under `assets/provenance/frogger_upright/`, and the bank
(`assets/upright_place_bank.json`, all 49 jitter-grid entries
feasible) and IK trajectory (`assets/upright_place_trajectory.json`,
63 knots, worst 2.0 mm, margin 0.110) follow the v1 builders.

The sixth increment implements the task: the environment
(`upright_place_env_cfg.py`, gym id `UprightPlace-L6-Play-v0`, the
v1 scene minus the receptacle plus a visual target disk), the
physical monitor (`upright_place/mdp/monitor.py`, the StowMonitor
pattern with the upright stage markers and failure cascade), the
scripted expert (`scripts/manipulation/upright_place_expert.py`) and
its driver (`run_upright_expert.py`), and backward-compatible
generalizations of the shared runtime (a task id and cycle-time
input to `run_episodes`, a reset-term name for `EnvSwitches`, stage
keys for `summarize`), pinned by the record-schema tests. The
upright simulator tests run in their own process
(`pytest tests/test_upright_physics.py --isaac-upright`), one Isaac
environment per process.

The seventh increment validates the expert end to end and freezes
the remaining geometry from what the simulation measured, kinematic
and expert-only criteria throughout. Three mechanisms invisible to
the kinematic probe appeared in validation traces: the idle left
hand at the arm-zero default occupies the left-side placement zone
(and any static re-park of the left arm shifts the torso sag enough
to break the millimeter-margin open-loop acquisition), so the target
moved to the robot's right of the transport axis at (0.527, 0.035),
0.207 m clear of the idle fingers; an object that pivots in the
grasp hangs below the grasp point and its end struck the table, so
the lift rose to 0.18 m; and the synthesized pinky contact at
+91 mm sat on the shaft's end edge and ejected the object axially
under squeeze, so the bank slides the grasp 20 mm toward the center
of mass (contact centroid +52 mm) and the expert's squeeze overdrive
excludes the pinky. At the frozen configuration the probe margin is
0.085, and the scripted expert validates 20 of 20 at r = 0.5 with
final tilt 0.0 deg and base offsets of 7 to 19 mm inside the 30 mm
target.

The eighth increment runs the Gate B expert-only calibration and
freezes the speed protocol. A first sweep found the expert's
placement bias (the in-hand pitch accumulated under the gravity
moment of the end-shifted grasp) crossing the 30 mm target radius
from r = 1, with the object standing at tilt 0.0 but 32 to 42 mm off
target, so the provisional nominal durations were re-anchored by
doubling the scaled phases (the phase table records the calibration),
the v1 procedure for setting the rate protocol from expert-only
evidence. The frozen sweep (64 episodes per speed, 10 mm jitter)
reads 55, 59, 62, 61, 63, 58, 26, 0, 0 of 64 at r in {0.5, 0.75,
1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0}: a plateau of 0.91 to 0.98 over
[0.75, 1.75], a collapse above r = 1.75, and a dip at r = 0.5 where
the slow cycle gives the in-hand pivot more time to creep. The
frozen training range is r uniform in [0.75, 1.75]; the frozen
evaluation grid is {0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5} with
r = 0.5 testing extrapolation below the demonstrated range and
r >= 2.0 above it, 100 episodes per policy and speed, the v1 seed
law, corruption off. The frozen values are stated in
[TASK_SPEC_UPRIGHT.md](TASK_SPEC_UPRIGHT.md).

The ninth increment runs the frozen protocol end to end and releases
the records. Demonstrations: 315 of 330 expert episodes admitted by
physical task success (r uniform in [0.75, 1.75], 10 mm jitter,
corruption on). One ACT policy trained with the v1 configuration
(2000 epochs, final loss 0.034). The matched evaluation (100 episodes
per policy and speed, paired draws under the v1 seed law) reads, at
r in {0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5}, expert 86, 93, 92,
97, 96, 90, 43, 3 and ACT 25, 11, 39, 53, 66, 74, 8, 0 of 100. The
ACT pilot does not reach nominal expert parity (0.39 against 0.92 at
r = 1), so under the Gate C rule this condition is a secondary
architecture result and supports no central matched claim; within
that scope, the paired differences are 0.82 [0.73, 0.90] at r = 0.75,
0.53 [0.42, 0.63] at r = 1, and 0.16 [0.05, 0.27] at r = 1.75, and
the ACT curve rises with speed inside the demonstrated range (drops
and placement misses dominate its slow-speed failures), the opposite
direction of the v1 ACT-A. The released records live under
`data/records/upright/` (evaluation summary and episode records,
expert calibration sweep, demonstration summary), the figure at
`media/upright_operating_envelope.*` regenerates with
`python scripts/plot_envelope.py --summary
data/records/upright/eval_summary.jsonl --actors expert act`, and the
checkpoint and demonstrations are inventoried in
`artifacts/manifest.json` (bundle `upright`, hosting pending like the
v1 flow).

The tenth increment implements the keyed-peg insertion task and
carries it to a green expert validation, with every geometric change
below fixed by a measured mechanism before any learner ran. The scene
adds a pocket block of nine kinematic slabs (a floor, four walls, and
four lead-in slabs, the v1 receptacle pattern) holding a 61 mm square
cavity, 60 mm deep, whose top sits 120 mm above the table; the peg is
the upright task's 55 x 55 x 180 mm cuboid, one object across both
tasks, and the pocket clearance is 3 mm per side. The path to the
green validation, in order: the block rose from 70 to 120 mm (waist
roll saturation at the descent bottom); the mouth gained the lead-in
funnel (sustained 45 N rim wedging at a 1 mm, 1 deg arrival); the
descent ends 10 mm above the pocket floor and releases inside the
guided cavity (commanding a full seat lowers the hand onto the mouth
hardware); the approach routes through a raised via (the open fingers
sweep the pocket's airspace on the direct blend); the pocket moved
from the probed upright target to (0.6187, 0.1273), transport 0.297 m
(at 0.180 m the block's near face sits at the lying peg's far end and
the grasp fingers extend into the fixture's airspace, 18 of 20
acquisition failures, all acquired after the move); the transport
corridor rose to 0.26 m (the loaded arm carries the hand 20 to 22 mm
below the commanded transfer height, and the peg base struck the near
lead mid-sweep, traced as a pocket-force spike with a slip step at
TRANSFER fraction 0.5 to 0.75); the expert's integral sag correction
extends through TRANSFER and the hand targets carry a
(-19.3, +1.1) mm world-frame compensation from the descent on (with
the droop closed, the object still arrived 19 mm past the pocket
center along x: it settles shifted in the grasp relative to the
synthesized hand-object transform, the same realized-grasp offset the
upright task absorbed in its 30 mm target); the retreat climbs
0.12 m as it withdraws (a horizontal retreat sweeps the open fingers
through the seated peg's proud shaft and levers it out of the
cavity); the squeeze overdrive rose to 0.30 rad (the object pivots
quasistatically in the grasp under its gravity moment, and arrivals
pitched past about 15 deg wedge in the funnel); and the bank builder
retries grid entries whose grasp solve returns its seed unmoved (one
entry admitted at 3.49 mm against 0.3 to 0.9 mm for its neighbors,
and every validation episode drawing it seated badly and pivoted).
The expert validates 19 of 20 at r = 1 under 10 mm start jitter, with
the one failure a pivot-creep outlier of the kind the operating
envelope characterizes.

The eleventh increment runs the Gate B expert-only calibration and
freezes the speed protocol. The first 64-episode sweep measured the
20 mm lead-in's capture edge exactly (at r = 0.5 every success
arrived within 12.8 mm of the pocket center and every jam at 17.2 mm
or more, against the 14 mm per-side widening), so the lead height
rose to 35 mm (24.5 mm per side, the lead's outer edge inside the
wall footprint); every r <= 1 timeout was a seated upright peg whose
settle window ran out, so the unscaled SETTLE rose from 1.0 to 2.0 s;
and the descent wedged at speed with centered arrivals (52 of 64
insertion jams at r = 1.5), so INSERT rose from 2.0 to 3.0 s, 43 mm/s
at r = 1. The frozen sweep (64 episodes per speed, 10 mm jitter)
reads 63, 62, 58, 53, 47, 48, 57, 58, 55 of 64 at r in {0.5, 0.75,
1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0}: at least 0.9 over [0.5, 1.0],
a valley of 0.73 to 0.75 at r in [1.5, 1.75], and a recovery to 0.86
to 0.91 at r >= 2. The valley-and-recovery shape separates the two
failure clocks of the tight-clearance regime: the in-hand pivot creep
is time-driven and shrinks as the cycle shortens, while the descent
excitation grows with speed, and above r = 2 the shorter cycle
starves the creep faster than the descent degrades. The training
range froze at [0.5, 1.0] by the 0.9 rule, lower and narrower than
the upright's [0.75, 1.75]; the evaluation grid keeps the eight
upright speeds so the matched evaluation measures the full
non-monotonic envelope.
