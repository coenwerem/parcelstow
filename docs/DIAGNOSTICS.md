# Failure-Localization Diagnostics

Every evaluated episode produces one JSON record. The fields below let an
analysis locate where a failure enters the manipulation, without rerunning
the simulator. `scripts/reproduce.py` drives the released analyses over
these records.

## Stage Outcomes

Booleans in task order, `acquired`, `lifted_clear`, `reoriented`,
`preinsert_reached`, `inserted`, `released`, `settled`, `task_success`,
plus `failure_reason` and `failure_detail` naming the first violated
predicate (for example `insertion_jam`, `insertion_misalignment`,
`excessive_inhand_slip`, `timeout`). The per-condition summary aggregates
each stage with Wilson intervals, so a stage-versus-speed plot shows which
stage separates two policies at a given execution speed.

## Hand-Object Motion

`max_hand_object_translation_m` and `max_hand_object_rotation_deg` track
the worst in-hand relative motion over the episode, with per-phase splits
`slip_reorient_*`, `slip_insert_*`, and the values at receptacle contact
`slip_*_at_receptacle_contact_*`. These separate transport slip from
contact-driven slip at insertion.

## Relative-Motion Handoff

`scripts/manipulation/stow_relative_handoff.py` implements the handoff reported
in the paper. Each policy controls acquisition. Once the parcel has been
acquired and lifted 40 mm, a shared controller preserves the policy's realized
hand shape and applies the expert's subsequent hand motion relative to the
policy's hand pose at handoff. The parcel remains a free rigid body, and the
controller receives no parcel-pose feedback.

The primary endpoint occurs at the first control step of insertion or at the
first earlier contact with the receptacle. It records whether the grasp retains
the parcel through free-space lift, reorientation, and transfer. The same
controller then continues through insertion, release, retreat, and settling to
provide the secondary full-task outcome. During release, the hand follows the
expert's opening motion; during retreat, the arm follows the expert-relative
retreat path. Episodes without a successful acquisition remain under the
policy's controller and record `relative_handoff=false`.

The released summary at
`experiments/paper/results/relative_handoff_summary.jsonl` therefore compares
the acquisitions produced by the policies under a shared downstream motion.
It does not isolate the handoff state from later interaction with the
receptacle.

`scripts/manipulation/stow_handoff.py` implements an earlier common-controller
ablation. It is not the relative-motion handoff reported in the paper.

## Acquisition-Time Force Closure

At acquisition, end of reorientation, and insertion start, the monitor
records the realized contact set (points, normals, forces) and scores it
with the Ferrari-Canny margin `epsilon_*` at the nominal friction,
computed in-repo by `mdp/ferrari_canny.py` (NumPy and SciPy only; `-1` is
the no-force-closure sentinel). Under the SciPy version used to produce the
records, the scorer reproduces all 6,322 recorded margins. Across Qhull
versions, 15 signs differ where the margin magnitude is below `1e-12`.

The released analysis (`scripts/reproduce.py certificate`) finds that no
acquired episode without force closure completes the task. This observation is
one-sided: the margin enters no success predicate, and the paper does not use
its continuous value to explain task success across execution speeds.

`epsilon_beta_*` additionally scores a risk-adjusted margin under friction
uncertainty (CVaR at beta 0.95). Computing it needs the optional
`firmgrasp` package (set `PARCELSTOW_FIRMGRASP` to its location, unset by
default and the field reads -1). The risk-adjusted margin concerns
friction uncertainty, a different axis from execution speed, and the
paper does not use it to explain temporal sensitivity.

## Actuation and Tracking

`max_joint_velocity_utilization`, `max_arm_velocity_utilization`,
`max_target_tracking_error_rad`, and `max_action_magnitude` record arm motion
and target tracking (`scripts/reproduce.py expert-ceiling`). Arm
joint-velocity utilization remains low through `r=2` while target-tracking
error against the commanded targets grows. The records provide no
evidence that arm joint-velocity saturation explains the expert's
observed success decrease at higher execution speeds.

## Physical-Integrity Checks

`tests/test_parcel_physics.py` (simulator-backed) asserts the physical
premises behind the records, no weld between hand and parcel, gravity
acts on the released parcel, insertion outside clearance or past the
orientation tolerance fails, scoring never mutates state, and per-env
phase and speedup factor stay independent.
