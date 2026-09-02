# Keyed-Peg Insertion Task Specification (Frozen Scientific Choices)

Status legend. FROZEN means the value is fixed and no learner result
may change it. Every value below derives from kinematic probes, grasp
synthesis feasibility, or expert-only calibration, never from a
learner outcome. The freeze log at the end records every change with
a date and a reason. The arXiv-v2 design note
([EXTENSION_PLAN.md](EXTENSION_PLAN.md)) records the increments and
their measured evidence; this document states the frozen values.

Document created 2026-09-01. Learner training and evaluation started
only after every entry below was frozen.

## 1. Scientific Question

The task extends the matched expert-learner evaluation across task
execution speeds to the tight-clearance containment regime: the
upright task's cuboid stood up and inserted into a square pocket with
3 mm of clearance per side, where failure is jam-mediated and the
terminal predicate is geometric containment. Faster execution raises
the in-hand pivot that arrival alignment depends on and the descent
excitation that wedging depends on; the two clocks run in opposite
directions with the cycle time, and the operating envelope measures
their competition. The endpoint is task success as a function of the
speedup factor r.

## 2. Object (FROZEN, Shared with the Upright Task)

The upright placement task's object, unchanged: rigid cuboid
55 x 55 x 180 mm, 0.120 kg, friction 0.5, restitution 0.0, one box
collider. One object across both new tasks holds the grasp synthesis,
the mass, and the aperture-floor rationale fixed
(TASK_SPEC_UPRIGHT.md section 2); a 160 mm variant returned no seated
force-closed grasp in synthesis.

## 3. Table, Robot, and Pocket Block (FROZEN)

Table and robot are the v1 scene (TASK_SPEC_UPRIGHT.md section 3);
the left arm stays at the arm-zero default. The pocket block is nine
kinematic slabs (the v1 receptacle pattern): a floor, four walls, and
four lead-in slabs, yaw-aligned with the goal at 45 deg.

- Cavity, 61 x 61 mm cross-section (object width plus 2 x 3 mm
  clearance), 60 mm deep; walls 30 mm thick, floor 10 mm.
- Block top at z = 0.820 m, 120 mm above the table (at 70 mm the
  waist roll saturates at the descent bottom, trajectory margin
  0.000).
- Lead-in funnel, four slanted slabs at 35 deg from the vertical,
  height 35 mm, widening the entry by 24.5 mm per side; the lead's
  outer edge at 55 mm from the cavity axis stays inside the 60.5 mm
  wall footprint, and the leads cover only the straight mouth edges
  (mitred corner pairs disturb the GPU contact pipeline, measured by
  scene bisection).
- Center at world (0.6187, 0.1273): along 0.28 m, lateral -0.10 m in
  the transport frame anchored at the start (d = rotz(45 deg) e_x),
  transport 0.297 m. At the probed upright target (0.527, 0.035),
  transport 0.180 m, the block's near face sits at the lying peg's
  far end and the grasp fingers extend past the object into the
  fixture's airspace (measured slab-distance traces, 18 of 20
  acquisition failures).

## 4. Start and Goal Poses (FROZEN)

Start, identical to the upright task: object center at
(0.35, 0.0, 0.7285), lying on a side face, long axis yawed +45 deg.
Goal, standing upright and seated in the pocket, cavity yaw equal to
the start yaw. Derived yaw tolerance of the containment,
2 c / a = 6.2 deg, stricter than the C4 symmetry requires; final tilt
tolerance 5 deg, fixed before any expert or learner ran.

## 5. Grasp Source (FROZEN, Provenance in assets/peg_insert_bank.json)

The upright task's synthesis record and 20 mm center-of-mass slide,
unchanged (TASK_SPEC_UPRIGHT.md section 5, contact centroid +52 mm).
Two peg-task additions to the bank builder, both from measured
mechanisms: a raised approach via (the pregrasp pose lifted 0.15 m,
hand open; the open fingers sweep the pocket's airspace on the direct
approach blend), and a multi-seed retry of any grid entry whose grasp
solve returns its seed unmoved (one entry admitted at 3.49 mm error
against 0.3 to 0.9 mm for its neighbors; every validation episode
drawing it seated badly and pivoted in the grasp). 49 of 49 grid
entries feasible, worst grasp error 1.74 mm.

## 6. Manipulation Geometry (FROZEN by Trajectory Build and Validation)

Lift straight up by 0.26 m: the transport corridor must clear the
lead-in tops, and the loaded arm carries the hand 20 to 22 mm below
the commanded transfer height (traced), so the 0.22 m corridor's
15 mm nominal clearance produced a mid-sweep strike on the near lead
(pocket-force spike with a 27 mm slip step at TRANSFER fraction 0.5
to 0.75); at 0.26 the nominal clearance is 55 mm. Reorient 90 deg at
the lift point, transfer at lift height to above the pocket, lower
to 10 mm above the pocket floor, release inside the guided cavity
(the v1 receptacle convention; commanding a full seat lowers the hand
onto the mouth hardware). Retreat backs the hand 0.10 m along the
reverse transport direction while climbing 0.12 m (the seated peg
stands 120 mm proud of the mouth, and a horizontal retreat levers it
out of the cavity with the open fingers). The trajectory build
carries a (-19.3, +1.1) mm world-frame compensation on the hand
targets, blended in over TRANSFER and held through the descent: with
the arm integrally on target, the object still arrives 19 mm past
the pocket center along x because it settles shifted in the grasp
relative to the synthesized hand-object transform (measured over 40
episodes, spread +-3 mm). Trajectory evidence at the frozen
configuration: 63 knots, all IK-verified within 2.0 mm and 0.7 deg,
minimum joint-limit margin 0.049 (the LIFT apex, waist roll).

## 7. Phase Sequence and Speedup Factor (FROZEN at Gate B)

Phases in order, with the nominal duration at unit rate,

    PARK            0.5 s   fixed
    APPROACH        2.5 s   fixed
    PREGRASP_DWELL  0.6 s   fixed
    CLOSE           1.5 s   fixed
    GRASP_DWELL     0.6 s   fixed
    LIFT            2.4 s   scaled by 1/r
    REORIENT        3.2 s   scaled by 1/r
    TRANSFER        3.2 s   scaled by 1/r
    INSERT          3.0 s   scaled by 1/r
    INSERT_DWELL    0.8 s   scaled by 1/r
    RELEASE         1.2 s   scaled by 1/r
    RETREAT         2.0 s   scaled by 1/r
    SETTLE          2.0 s   fixed (measurement window)

Cycle time at rate r equals 5.7 s + 15.8 s / r + 2.0 s. The geometric
path is a function of the phase index and in-phase fraction alone,
asserted by the pure tests. Two durations were re-anchored by this
task's Gate B calibration from the upright values: INSERT rose from
2.0 to 3.0 s (the 128 mm guided descent wedges at speed, 52 of 64
insertion jams at r = 1.5 with centered arrivals) and SETTLE rose
from 1.0 to 2.0 s (every r <= 1 timeout of the first sweep was a
seated upright peg whose settle window ran out). Episode length 45 s
(fallback); the per-environment task_complete termination ends the
episode at the cycle time.

Frozen expert calibration (64 episodes per speed, 10 mm jitter,
outputs of run_peg_expert.py --mode sweep), success 63, 62, 58, 53,
47, 48, 57, 58, 55 of 64 at r in {0.5, 0.75, 1.0, 1.25, 1.5, 1.75,
2.0, 2.5, 3.0}: at least 0.9 over [0.5, 1.0], a valley of 0.73 to
0.75 at r in [1.5, 1.75], and a recovery to 0.86 to 0.91 at r >= 2
(the time-driven pivot creep shrinks with the cycle while the descent
excitation grows with speed).

## 8. Physical Success (FROZEN)

Stage markers, each latched at the first step its condition holds
(peg_insert/mdp/monitor.py),

- acquired, object center at least 20 mm above its start height while
  the thumb distal phalanx and at least one other distal phalanx each
  read an object-filtered contact force above 1 N.
- lifted_clear, object center at least 60 mm above its start height.
- reoriented_upright, tilt under 15 deg while held, before RELEASE.
- aligned, object center within 15 mm of the pocket axis in the plane
  and tilt under 15 deg while held.
- inserted, base at least 40 mm below the pocket top with all four
  base corners inside the cavity cross-section, at or after INSERT.
- released, after inserted, the summed object-filtered distal contact
  force under 0.5 N for at least 0.1 s.
- settled, after released, inserted still true, linear speed under
  0.02 m/s and angular speed under 0.2 rad/s for a dwell of 0.4 s.
- task_success = inserted and released and settled and, at the
  episode end, all four base corners inside the cavity cross-section
  and the final tilt at most 5 deg.

The cross-section test encodes the derived yaw and offset tolerances;
no wrench-space quantity enters any predicate. A jam diagnostic
latches on sustained pocket contact force above 2 N during INSERT
while not inserted (10 consecutive steps); it attributes failures and
gates nothing.

Failure reason, the first category that applies in the order below,

    acquisition_failure       not acquired
    dropped_during_transport  acquired, contact lost before RELEASE
                              while the object is not inserted
    insertion_jam             not inserted, the jam diagnostic latched
    alignment_failure         not inserted without a latched jam, or
                              settled with a final tilt above 5 deg
    timeout                   inserted but not released, or released
                              but not settled
    other                     anything else (including leaving the
                              pocket after settling)

## 9. Slip Diagnostics (FROZEN, Diagnostics Only)

As the upright task (TASK_SPEC_UPRIGHT.md section 9), with the
per-phase splits at the end of REORIENT and at INSERT start, and the
pocket contact force recorded from the slab-filtered contact sensor
(peg_pocket_s). No slip or force threshold defines success.

## 10. Observation and Action Interface (FROZEN, the v1 Interface)

The 147-D state observation and 16-D joint-position action of the v1
interface, unchanged in layout (TASK_SPEC_UPRIGHT.md section 10).

## 11. Expert Construction (FROZEN in Method)

Acquisition from the bank's planar start-offset grid entry nearest
the realized start; a two-segment approach blend through the raised
via; cosine blends between phase targets; manipulation targets
interpolate the IK-verified trajectory knots
(assets/peg_insert_trajectory.json). The v1 integral sag correction
runs in the dwells and, for the arm, through TRANSFER: the gravity
droop of the loaded arm at the far-reach posture is stale by the
transfer posture, and closing it in free air recenters the descent;
the correction holds fixed through the contact-rich INSERT so the
integral never winds up against the funnel. After the gentle close to
the synthesized hand shape, the flexion joints except the pinky ramp
0.30 rad deeper across GRASP_DWELL and hold that squeeze through the
manipulation (raised from the upright's 0.20: the object pivots
quasistatically under its gravity moment through the free-air phases,
and arrivals pitched past about 15 deg wedge in the funnel; slowing
phases would lengthen the time under moment instead). The object is
never attached to anything.

## 12. Training Distribution (FROZEN at Gate B)

- Start jitter, uniform dx, dy in [-10, 10] mm, no yaw jitter.
- Speedup factor, uniform over [0.5, 1.0], the range over which the
  expert succeeds in at least 0.9 of its episodes.
- Demonstrations, complete expert episodes through SETTLE, admitted
  by physical task_success only, collected once with observation
  corruption on, and reused by every learner.

## 13. Evaluation Distribution (FROZEN at Gate B)

- The same jitter law as training, seeds fixed per speed
  (12345 + 1000 x speed index) and identical across policies.
- Speedup grid {0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5}, the
  upright grid, with r >= 1.25 above the demonstrated range so the
  matched evaluation measures the full non-monotonic envelope, 100
  episodes per policy and speed, corruption off.

## 14. Primary Metrics (FROZEN)

As the upright task (TASK_SPEC_UPRIGHT.md section 14): success and
stage probabilities per rate with Wilson intervals, failure reason
distributions, slip distributions, and the paired expert-learner
differences of plot_envelope.py.

## 15. Freeze Log

- 2026-09-01, block top raised 70 to 120 mm (waist roll saturation at
  the descent bottom), lead-in funnel added (45 N rim wedging at a
  1 mm, 1 deg arrival), release moved to 10 mm above the pocket floor
  (fingertip-fixture jams at the full seat), object set to the shared
  180 mm cuboid (160 mm synthesis returned empty), all before any
  learner ran.
- 2026-09-01, approach routed through a raised via (open fingers
  sweep the pocket's airspace on the direct blend, measured by
  slab-distance instrumentation).
- 2026-09-01, pocket moved from the probed upright target to
  (0.6187, 0.1273), transport 0.180 to 0.297 m (the grasp fingers
  extend past the lying peg's far end into the fixture at the grasp
  pose; acquisition 2 of 20 before, 20 of 20 after).
- 2026-09-01, lift raised 0.22 to 0.26 m (the loaded hand rides 20 to
  22 mm below the commanded transfer height and the peg base struck
  the near lead mid-sweep, traced), the slab-filtered pocket contact
  sensor restored (removed as bisection collateral in an earlier
  commit).
- 2026-09-01, arm integral correction extended through TRANSFER and
  the trajectory's (-19.3, +1.1) mm realized-grasp compensation added
  (droop closed, the object still arrives 19 mm past center along x),
  retreat rise 0.12 m added (horizontal retreat levers the seated peg
  out), squeeze overdrive 0.20 to 0.30 rad (quasistatic pivot creep
  past the funnel's pitch tolerance), bank grid outlier retry added
  (one entry solved at its seed, 3.49 mm error). Expert validation
  19 of 20 at r = 1 under 10 mm jitter.
- 2026-09-01 Gate B, lead height 20 to 35 mm (the capture edge
  measured at 12.8 mm success versus 17.2 mm jam arrivals against the
  14 mm per-side widening), INSERT 2.0 to 3.0 s (descent wedging at
  speed), SETTLE 1.0 to 2.0 s (seated timeouts), rate grid and
  training range [0.5, 1.0] frozen from the 64-episode sweep, all
  before any learner ran.
