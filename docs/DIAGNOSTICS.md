# Failure-localization diagnostics

Every evaluated episode produces one JSON record. The fields below let an
analysis locate where a failure enters the manipulation, without rerunning
the simulator. `scripts/reproduce.py` drives the released analyses over
these records.

## Stage outcomes

Booleans in task order, `acquired`, `lifted_clear`, `reoriented`,
`preinsert_reached`, `inserted`, `released`, `settled`, `task_success`,
plus `failure_reason` and `failure_detail` naming the first violated
predicate (for example `insertion_jam`, `insertion_misalignment`,
`excessive_inhand_slip`, `timeout`). The per-condition summary aggregates
each stage with Wilson intervals, so a stage-versus-speed plot shows which
stage separates two policies at a given execution speed.

## Hand-object motion

`max_hand_object_translation_m` and `max_hand_object_rotation_deg` track
the worst in-hand relative motion over the episode, with per-phase splits
`slip_reorient_*`, `slip_insert_*`, and the values at receptacle contact
`slip_*_at_receptacle_contact_*`. These separate transport slip from
contact-driven slip at insertion.

## Relative-motion handoff

`scripts/manipulation/stow_relative_handoff.py` implements the
relative-motion handoff of the paper. The evaluated policy (expert or an
ACT training seed) runs its own controller through acquisition. At stable
handoff, after acquisition and before RELEASE, control passes to a shared
controller (`stow_relative_controller.py`) that commands the waist and arm
so the hand follows the expert's post-acquisition hand motion expressed
relative to the policy's own realized hand pose at handoff, while the hand
shape holds the policy's own realized grasp. This shared controller
supplies free-space manipulation, lift, reorientation, and transfer,
through the primary endpoint, the first control step of INSERT or the
first receptacle contact, whichever comes first. From RELEASE onward the
same shared controller supplies insertion, release, and retreat: the hand
follows the expert's opening motion and the arm follows the expert's
relative retreat path. Episodes in which the policy never acquires the
parcel run to their end under the policy's own controller and are marked
`relative_handoff` false. The released summaries under
`experiments/paper/results/relative_handoff_summary.jsonl` quantify how
well each policy's acquisition preserves the parcel under this shared
downstream motion.

`scripts/manipulation/stow_handoff.py` implements an earlier, separate
common-controller ablation (M12) that is not the relative-motion handoff
above and is not part of the paper's reported relative-motion handoff
results; its own docstring marks it as an ablation, not the main method.

## Acquisition-time force closure

At acquisition, end of reorientation, and insertion start, the monitor
records the realized contact set (points, normals, forces) and scores it
with the Ferrari-Canny margin `epsilon_*` at the nominal friction,
computed in-repo by `mdp/ferrari_canny.py` (numpy and scipy only, -1 is
the no-force-closure sentinel). The in-repo scorer reproduces all 6322
recorded margins of the released records exactly under the producing
scipy version, with 15 sign ties at |epsilon| under 1e-12 across qhull
versions. The released analysis (`scripts/reproduce.py certificate`)
shows the one-sided acquisition result of the paper, across the released
records no acquired episode lacking force closure completes the task. The
margin is a diagnostic only, it enters no success predicate, and the
paper does not present the continuous margin as an explanation of task
success across execution speeds.

`epsilon_beta_*` additionally scores a risk-adjusted margin under friction
uncertainty (CVaR at beta 0.95). Computing it needs the optional
`firmgrasp` package (set `PARCELSTOW_FIRMGRASP` to its location, unset by
default and the field reads -1). The risk-adjusted margin concerns
friction uncertainty, a different axis from execution speed, and the
paper does not use it to explain temporal sensitivity.

## Actuation and tracking

`max_joint_velocity_utilization`, `max_arm_velocity_utilization`,
`max_target_tracking_error_rad`, and `max_action_magnitude` support the
expert-ceiling attribution (`scripts/reproduce.py expert-ceiling`). Arm
joint-velocity utilization remains low through r=2 while target-tracking
error against the commanded targets grows. The records provide no
evidence that arm joint-velocity saturation explains the expert's
observed success decrease at higher execution speeds.

## Physical-integrity checks

`tests/test_parcel_physics.py` (simulator-backed) asserts the physical
premises behind the records, no weld between hand and parcel, gravity
acts on the released parcel, insertion outside clearance or past the
orientation tolerance fails, scoring never mutates state, and per-env
phase and speedup factor stay independent.
