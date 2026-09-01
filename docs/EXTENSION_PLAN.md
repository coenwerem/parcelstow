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

- Object: a rigid square prism, approximately 25 x 25 x 90 mm and
  0.10 kg, resting on its side; a table-mounted fixture block with a
  square through-slot providing 2 to 3 mm of clearance per side. The
  square cross-section admits four yaw solutions (C4 symmetry about
  the peg axis).
- Contact sequence: grasp the shaft, lift, transport to the slot
  axis, align, insert axially with possible two-point wedging
  contact, release, settle.
- Expert construction: as v1 in method; a grasp-bank entry
  synthesized for the peg, task-space object waypoints along the
  frozen path, damped least squares IK at fixed knots, joint-target
  interpolation between knots.
- Speed-scaled phases: lift, transport, alignment dwell, insert,
  insert dwell, release, retreat. Fixed phases: park, approach,
  pregrasp dwell, close, grasp dwell, settle.
- Demonstrated speed range: from the expert calibration sweep; the
  tighter clearance is expected to narrow the range relative to v1,
  which Gate B measures rather than assumes.
- Success predicate: peg center past a depth threshold along the
  slot axis, inside the slot cross-section, released, and settled
  within a final tilt tolerance derived from the clearance-to-length
  ratio, following the v1 derivation (2 c / L).
- Stage outcomes: acquired, lifted_clear, aligned, inserted,
  released, settled.
- Failure reasons: acquisition_failure, dropped_during_transport,
  alignment_failure, insertion_jam, release_failure, timeout.
- Initial conditions: planar start jitter in x and y, as v1; no yaw
  jitter in the first version.
- Evidence added: the tolerance-to-clearance ratio is roughly an
  order of magnitude tighter than ParcelStow's 10 mm per side, and
  there is no 90-degree reorientation, so success is bound by
  terminal alignment precision rather than by reorientation-phase
  transport. Jamming here is wedge-mediated two-point contact rather
  than misalignment at a wide entrance. Whether the v1 separation
  persists when the accuracy demand rather than the transport
  dynamics binds is exactly the cross-task question.
- xArm7 feasibility: high; an insertion scene already exists for the
  xArm7-L6 MuJoCo arena, and the hardware fixture is one machined
  block.
- Cost: one new environment configuration, geometry module, grasp
  bank entry, and IK trajectory; the phase clock, monitor pattern,
  drivers, and statistics are reused. Demonstrations under 1 h, ACT
  training about 45 min, evaluation about 2 h per policy at v1
  episode counts.

### Upright Placement

- Object: a tall rigid cuboid, approximately 40 x 40 x 140 mm and
  0.15 kg, starting on its side; a marked circular target region on
  the table. No receptacle.
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
through the shared `PhaseSchedule`. Its numeric constants stay
provisional until the kinematic probe, and no environment, scripted
expert, or gym registration exists yet.
