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
each stage with Wilson intervals, so a stage-versus-rate plot shows which
stage separates two actors at a given rate.

## Hand-object motion

`max_hand_object_translation_m` and `max_hand_object_rotation_deg` track
the worst in-hand relative motion over the episode, with per-phase splits
`slip_reorient_*`, `slip_insert_*`, and the values at receptacle contact
`slip_*_at_receptacle_contact_*`. These separate transport slip from
contact-driven slip at insertion.

## Relative-motion handoff

The handoff drivers (`scripts/manipulation/stow_relative_handoff.py`,
`stow_handoff.py`) replay episodes where control of the free-space
transport switches between actors while the acquisition and insertion
segments stay with a reference. The released summaries under
`experiments/paper/results/relative_handoff_summary.jsonl` quantify how
well each actor preserves the parcel during free-space transport in
isolation.

## Realized-contact certificate

At acquisition, end of reorientation, and insertion start, the monitor
records the realized contact set (points, normals, forces) and scores it
with the Ferrari-Canny margin `epsilon_*` at the nominal friction. The
released analysis (`scripts/reproduce.py certificate`) shows the one-sided
acquisition result of the paper, across the released records no acquired
episode lacking force closure completes the task. The certificate is a
diagnostic only, it enters no success predicate, and the paper does not
present the continuous margin as an explanation of the task-rate envelope.

`epsilon_beta_*` additionally scores a risk-adjusted margin under friction
uncertainty (CVaR at beta 0.95). Computing it needs the optional
`firmgrasp` package (set `PARCELSTOW_FIRMGRASP` to its checkout, absent by
default and the field reads -1). The risk-adjusted margin concerns
friction uncertainty, a different axis from task rate, and the paper does
not use it to explain rate sensitivity.

## Actuation and tracking

`max_joint_velocity_utilization`, `max_arm_velocity_utilization`,
`max_target_tracking_error_rad`, and `max_action_magnitude` support the
expert-ceiling attribution, arm-speed utilization stays low through r 2
while tracking error against the commanded targets grows, so servo
tracking, not actuator saturation, bounds the expert
(`scripts/reproduce.py expert-ceiling`).

## Physical-integrity checks

`tests/test_parcel_physics.py` (simulator-backed) asserts the physical
premises behind the records, no weld between hand and parcel, gravity
acts on the released parcel, insertion outside clearance or past the
orientation tolerance fails, scoring never mutates state, and per-env
phase and rate stay independent.
