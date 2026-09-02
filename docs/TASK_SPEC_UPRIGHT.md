# Upright Placement Task Specification (Frozen Scientific Choices)

Status legend. FROZEN means the value is fixed and no learner result
may change it. Every value below derives from kinematic probes, grasp
synthesis feasibility, or expert-only calibration, never from a
learner outcome. The freeze log at the end records every change with
a date and a reason. The historical development record
([EXTENSION_PLAN.md](development-history/EXTENSION_PLAN.md)) records the increments and
their measured evidence; this document states the frozen values.

Document created 2026-09-01. Learner training and evaluation started
only after every entry below was frozen.

## 1. Scientific Question

The task extends the matched expert-learner evaluation across task
execution speeds to a manipulation whose terminal predicate is
quasi-static stability rather than the geometric containment of
ParcelStow. A scripted expert reorients a tall cuboid to an upright pose on a marked
target region; the only environment contact is the table, so failure
cannot be jam-mediated, and faster execution raises the release
transients and the in-hand pivot that placement precision depends on.
The endpoint is task success as a function of the speedup factor r.

## 2. Object (Frozen, Chosen by Kinematic and Synthesis Criteria Only)

- Shape, rigid cuboid, extents 55 x 55 x 180 mm (x, y, z in the
  object frame, the long axis on z).
- Mass, 0.120 kg, the v1 parcel mass, holding object mass fixed
  across the task suite.
- Physics material, static and dynamic friction 0.5, restitution 0.0,
  the v1 values.
- Collision geometry, one box collider, no decorative mesh.
- Rationale, FRoGGeR returned no seated force-closed grasp for a
  cuboid of 40 or 50 mm width and three grasps at the v1-proven
  55 mm, so 55 mm is the aperture floor of the RealHand L6 for this
  synthesis; the 180 mm length admits the end-shifted grasp region
  the kinematic probe requires (section 5) with the contact span
  ending inside the shaft. Tipping angle of the resting cuboid,
  atan(27.5 / 90) = 17.0 degrees.

## 3. Table and Robot (Frozen, the v1 Scene)

Table center (0.55, 0.0, 0.68), size 0.81 x 1.092 x 0.04 m, top at
z = 0.70 m. Pelvis fixed at (0, 0, 0.75), identity orientation, the
robot faces +x. The left arm stays at the arm-zero default: any
static re-park shifts the waist gravity load and the resulting torso
sag enough to break the millimeter-margin open-loop acquisition
(measured, two park variants).

## 4. Start and Goal Poses (Frozen)

Object center at (0.35, 0.0, 0.7285), 1 mm above rest height, lying
on a 55 x 180 face with the long axis yawed +45 deg (the grasped +z
end points forward-left). Goal, standing upright with the base center
on the target region, goal yaw equal to the start yaw (the probe
found goal-yaw offsets drive the wrist yaw to its limit during
reorientation). Target region, a circle of radius 30 mm centered at
world (0.527, 0.035), on the robot's right of the transport axis and
0.207 m clear of the idle left hand (the left-side candidates put the
placement inside the idle hand's zone). Transport distance 0.180 m.

## 5. Grasp Source (Frozen, Provenance in assets/upright_place_bank.json)

The synthesis record is
assets/provenance/frogger_upright/scene_lab_x0.35_riser0/cuboid_180x55x55.json,
produced by the g1_l6 runner of the local FRoGGeR checkout at the v1
provenance commit with the v1 scene arguments (thorough grid,
risk_aware objective, clearance 5 mm, comfort naturalization, table
0.742 m in the synthesis scene, object at (0.35, 0.0), riser 0 mm).
The synthesis placed the five-contact grasp at +46 to +91 mm along
the shaft (centroid +72 mm) on its own; the bank slides it 20 mm
toward the center of mass along the constant cross-section (centroid
+52 mm) because the pinky contact at +91 mm sits on the shaft's end
edge and ejects the object axially under squeeze (measured in
validation traces). A shaft-centered grasp is kinematically
infeasible, the waist roll saturates while lowering (probe, minimum
margin 0.001). The bank re-solves the arm chain by damped least
squares IK at the start pose and fills the planar start-offset grid
of the v1 procedure, 49 of 49 entries feasible.

## 6. Manipulation Geometry (Frozen by the Probe and Validation)

Lift straight up by 0.18 m: an object that pivots in the grasp hangs
142 mm below the grasp point, and at this lift its hanging end clears
the table by 46 mm at the start of the reorientation. Reorientation,
90 deg about the horizontal axis at the lift point (the +z end rises).
Transfer at lift height to above the target, lower to the place pose,
which seats the base at its rest height: the object arrives pitched a
few degrees in the grasp, and seating presses the leading base edge
onto the table, which rights the object while it is still held
(releasing from a positive drop instead plants the tilted base off
center, 32 to 42 mm against the 30 mm radius). Release opens the hand
to the pregrasp shape; retreat backs the hand 0.10 m along the
reverse transport direction. Probe evidence at the frozen
configuration, all 38 knots within 3.6 mm and 0.9 deg, minimum
joint-limit margin 0.085 (worst at the end of LOWER, waist pitch);
trajectory evidence, 63 knots within 2.0 mm at margin 0.110.

## 7. Phase Sequence and Speedup Factor (Frozen after Expert-Only Calibration)

Phases in order, with the nominal duration at unit rate,

    PARK            0.5 s   fixed
    APPROACH        2.5 s   fixed
    PREGRASP_DWELL  0.6 s   fixed
    CLOSE           1.5 s   fixed
    GRASP_DWELL     0.6 s   fixed
    LIFT            2.4 s   scaled by 1/r
    REORIENT        3.2 s   scaled by 1/r
    TRANSFER        3.2 s   scaled by 1/r
    LOWER           2.0 s   scaled by 1/r
    PLACE_DWELL     0.8 s   scaled by 1/r
    RELEASE         1.2 s   scaled by 1/r
    RETREAT         2.0 s   scaled by 1/r
    SETTLE          1.0 s   fixed (measurement window)

Cycle time at rate r equals 5.7 s + 14.8 s / r + 1.0 s. The geometric
path is a function of the phase index and in-phase fraction alone, so
changing r changes only the timing, the v1 invariant, asserted by the
pure tests. The scaled nominals were set by the expert-only phase-schedule
calibration: at half these durations the expert's placement bias (the
in-hand pitch accumulated under the gravity moment of the end-shifted
grasp) exceeds the 30 mm target radius from r = 1. Episode length
45 s (fallback); the per-environment task_complete termination ends
the episode at the cycle time.

Frozen expert calibration (64 episodes per speed, 10 mm jitter,
outputs of run_upright_expert.py --mode sweep), success 55, 59, 62,
61, 63, 58, 26, 0, 0 of 64 at r in {0.5, 0.75, 1.0, 1.25, 1.5, 1.75,
2.0, 2.5, 3.0}: at least 0.9 over [0.75, 1.75], collapse above 1.75,
and a dip at 0.5 where the slow cycle gives the in-hand pivot more
time to creep.

## 8. Physical Success (Frozen)

Stage markers, each latched at the first step its condition holds
(upright_place/mdp/monitor.py),

- acquired, object center at least 20 mm above its start height while
  the thumb distal phalanx and at least one other distal phalanx each
  read an object-filtered contact force above 1 N.
- lifted_clear, object center at least 60 mm above its start height.
- reoriented_upright, tilt of the object z axis from the vertical
  under 15 deg while held, before RELEASE.
- placed, object center within 30 mm of the place pose and tilt under
  15 deg while held.
- released, after placed, the summed object-filtered distal contact
  force under 0.5 N for at least 0.1 s.
- settled, after released, tilt under 15 deg, base center inside the
  target region, linear speed under 0.02 m/s and angular speed under
  0.2 rad/s for a dwell of 0.4 s.
- task_success = placed and released and settled and, at the episode
  end, the base center inside the target region and the final tilt at
  most 5 deg.

The 5 deg final tilt tolerance is stricter than the 17.0 deg tipping
angle and was fixed before any expert or learner ran. No wrench-space
quantity enters any predicate.

Failure reason, the first category that applies in the order below,

    acquisition_failure       not acquired
    dropped_during_transport  acquired, contact lost before RELEASE
                              while the object is not placed
    placement_miss            held, never within the place tolerance
    timeout                   placed but not released, or released
                              but not settled
    tipped_after_release      released, final tilt above 5 deg or the
                              base center outside the target region
    other                     anything else

## 9. Slip Diagnostics (Frozen, Diagnostics Only)

At the acquired step the recorder stores the object pose in the hand
root frame; every later step reports the relative translation and
rotation against it, with per-phase splits at the end of REORIENT and
at LOWER start. Realized contact sets are recorded at acquisition, at
the end of REORIENT, and at LOWER start and scored with the in-repo
Ferrari-Canny margin as diagnostics (epsilon_lift, epsilon_reorient,
epsilon_lower). No slip or margin threshold defines success.

## 10. Observation and Action Interface (Frozen, the v1 Interface)

The 147-D state observation and 16-D joint-position action of the v1
interface, unchanged in layout: the object pose slice carries the
cuboid pose in the pelvis frame, task_phase is (k + f) / 13, and the
speedup factor r sits at index 146. Control at 50 Hz, physics at
200 Hz, the v1 PD gains.

## 11. Expert Construction (Frozen in Method)

Acquisition from the bank's planar start-offset grid entry nearest
the realized start, cosine blends between phase targets, the v1
integral sag correction during dwells. After the gentle close to the
synthesized hand shape, the flexion joints except the pinky ramp
0.20 rad deeper across GRASP_DWELL and hold that squeeze through the
manipulation (the pinky sits nearest the shaft end, where extra
flexion ejects the object). Manipulation targets interpolate the
IK-verified trajectory knots (assets/upright_place_trajectory.json).
The object is never attached to anything.

## 12. Training Distribution (Frozen Before Learner Training)

- Start jitter, uniform dx, dy in [-10, 10] mm, no yaw jitter.
- Speedup factor, uniform over [0.75, 1.75], the range over which the
  expert succeeds in at least 0.9 of its episodes.
- Demonstrations, complete expert episodes through SETTLE, admitted
  by physical task_success only, collected once with observation
  corruption on, and reused by every learner.

## 13. Evaluation Distribution (Frozen Before Learner Evaluation)

- The same jitter law as training, seeds fixed per speed
  (12345 + 1000 x speed index) and identical across policies.
- Speedup grid {0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5}, with
  r = 0.5 testing extrapolation below the demonstrated range and
  r >= 2.0 above it, 100 episodes per policy and speed, corruption
  off.

## 14. Primary Metrics (Frozen)

- P(task_success | rate) with Wilson 95 percent intervals, per policy.
- Stage completion probabilities per rate.
- Failure reason distribution per policy and rate.
- Slip distributions per policy and rate.
- The matched expert-learner difference at each speed under the
  paired draws, with the paired bootstrap of plot_envelope.py.

## 15. Freeze Log

- 2026-09-01, object extents revised 40 x 40 x 140 to 55 x 55 x 180
  on synthesis feasibility (no seated force-closed grasp below
  55 mm width); mass set to the v1 parcel's 0.120 kg.
- 2026-09-01, target moved from the provisional (0.457, 0.107) to
  (0.527, 0.035) on the robot's right of the transport axis (the
  idle left hand occupies the left-side zone) and lift raised from
  0.12 to 0.18 m (the pivot-to-hang end struck the table), both from
  measured validation mechanisms, before any learner ran.
- 2026-09-01, grasp slid 20 mm toward the center of mass (the pinky
  end-edge contact ejects the object axially) and the place pose
  seats the base at rest height (a positive release drop plants the
  pitched base off center), from validation traces, before any
  learner ran. Expert validation 20 of 20 at r = 0.5.
- 2026-09-01 expert-only phase-schedule calibration, scaled nominal durations doubled from the
  provisional values (the placement bias crosses the target radius
  from r = 1 under the fast nominal), episode fallback 45 s, rate
  grid and training range frozen from the 64-episode sweep, all
  before any learner ran.
