# ParcelStow Task Specification (Frozen Scientific Choices)

Status legend. FROZEN means the value is fixed and no learner result may
change it. PENDING(Mk) means the value waits for the named milestone, whose
inputs are kinematic or expert-only measurements, never learner outcomes.
The freeze log at the end records every change with a date and a reason.

Document created 2026-08-18 (M0). Learner training and evaluation start
only after every PENDING entry reads FROZEN.

## 1. Scientific Question

ParcelStow asks whether temporal robustness persists through imitation
of a dexterous behavior. A model-derived expert acquires a small
parcel and executes a full reorient-and-stow manipulation with real
contacts. DAgger, Diffusion Policy, and ACT imitate the same demonstration
set. The endpoint is task success as a function of the speedup factor r,
never an analytical grasp margin. Ferrari-Canny epsilon and the risk-adjusted
epsilon^(beta) on the realized contact set are diagnostics only. They enter
no success predicate, no termination, no expert acceptance rule, no rate
selection, and no action filter.

## 2. Object (Frozen, Chosen by Kinematic Criteria Only)

- Shape, rigid cuboid, extents 80 x 55 x 40 mm (x, y, z in the object frame
  and in the world frame at the start pose, the object rests on its 80 x 55
  face, 40 mm tall).
- Mass, 0.120 kg (density 682 kg/m^3, a filled small carton).
- Physics material, static and dynamic friction 0.5, restitution 0.0, the
  same values the cube protocol used and the friction the force-closure
  margin scorer takes as nominal.
- Collision geometry, one box collider, no decorative mesh.
- Rationale, the L6 pad aperture measured in the synthesis tool spans 38.8
  to 65.5 mm, so opposition across the 55 mm width sits inside the band with
  margin while 80 mm sits outside it. The synthesis tool returned a
  five-contact grasp across the 55 mm width on the first pass (section 9).
  The 80 mm length gives the insertion axis a meaningful depth and the
  40 mm height gives the reorientation a visible change of the supporting
  face. No learner outcome informed these numbers.

## 3. Table (Frozen, Unchanged from the Cube Tasks)

Table center (0.55, 0.0, 0.68), size 0.81 x 1.092 x 0.04 m, top at
z = 0.70 m. Pelvis fixed at (0, 0, 0.75), identity orientation, the robot
faces +x, the right arm hangs at -y.

## 4. Start Pose (Frozen)

Parcel center at (0.35, 0.0, 0.721), 1 mm above the table top, yawed by
+45 deg about the vertical (the 80 mm axis points forward-left as seen
from the robot). The kinematic probe chose both numbers. At the cube spot
(0.43, 0) the top-down grasp binds the wrist pitch limit (probe of
2026-08-18, start_pose_probe.json), at x = 0.35 the grasp of the frozen
record sits 0.22 from every limit, and the start yaw of +45 deg gives the
downstream reorientation the largest joint-limit margins (0.12) among the
probed yaws (outputs/paper/probe_r0g00b.json). Training and
evaluation add planar start jitter (sections 12 and 13). No learner
outcome informed these numbers.

## 5. Grasp Source (Frozen, Provenance in assets/gdf_bank_parcel.json)

The synthesis record is
assets/provenance/frogger_parcel/scene_lab_x0.35_riser0/parcel_80x55x40.json
(rank 0 of two returned grasps, the sibling rank 1 sits beside it),
produced by scripts/g1_l6_runner.py of the local frogger checkout at
commit 4705a49 on 2026-08-18 with the arguments recorded in the run.log of
that directory (thorough grid, risk_aware objective, clearance 5 mm,
comfort naturalization, the lab-equivalent scene, table 0.742 m relative
to the frogger pelvis, object at (0.35, 0.0), riser 0 mm, so the
finger-table clearance gate applies at the flush table of ParcelStow).
Every earlier record (frogger default scene, riser 15 mm, other object
positions) stays under assets/provenance/frogger_parcel/ for the audit
trail. The record stores X_WB, the hand-root
pose in the object frame, the eleven hand joint values, and the arm
configuration of the synthesis scene. The bank builder re-solves the arm
chain by damped least squares IK at the ParcelStow start pose, the same
procedure that built the cube bank. Yaw symmetry set of the resting
cuboid, {0, 180} degrees (C2 about the vertical axis), never intermediate
yaws.

## 6. Receptacle and Manipulation Geometry (Frozen by the Kinematic Probe)

Family C of IMPLEMENTATION_LOG.md, the parcel tilts by 90 deg about its
own width axis (Ry(-90) in the task frame, the task frame is the world
frame yawed by the start yaw of +45 deg), stands on its 55 x 40 mm end,
and slides forward-left along d = (0.707, 0.707, 0) into an open-front
receptacle whose entrance faces the robot. The reorientation runs at the lift
point (start plus 120 mm) while the parcel starts half of its travel toward
the pre-insertion pose (reorient_travel 0.5), the transfer completes the
travel. Total transport, 0.334 m. Palm behind and below the parcel center
(the record's hand root sits 72 mm toward the robot and 108 mm above the
flat parcel), fingers on the two 80 x 40 mm faces, no phalanx under the
parcel at release (release clearance +11 mm), so the release drops the
parcel 10 mm onto the receptacle floor and the hand withdraws straight back
along -d by 0.10 m.

Receptacle, five kinematic rigid boxes (floor, side_a, side_b, back, top), wall
thickness 20 mm, floor top 0.16 m above the table (a pedestal box from the
table to the floor top), yawed with the task frame. Interior, tight axis
vertical (parcel height 80 mm plus 2 x 10 mm, 100 mm), loose axis
horizontal across d (parcel width 55 mm plus 2 x 55 mm, 165 mm, the
phalanx bodies sit up to 45 mm outboard of the pad contact and the probe
measured penetration at 25 and 45 mm clearance), depth 40 mm plus 70 mm
slack (110 mm, the thumb tip frame reaches 25 mm past the leading face).
Entrance center at world (0.491, 0.141), the insert target puts the parcel
center 60 mm past the entrance plane (leading face 30 mm from the back
wall), the pre-insertion pose sits 30 mm outside the entrance. Predicate
inserted, center depth at least 40 mm (trailing face 20 mm inside) and
inside the interior box.

Probe evidence (outputs/paper/geometry_finalize.json), every
knot solves within 4 mm and 2 deg, joint-limit margin at least 0.12 over
the whole manipulation (the waist roll saturation of the earlier record
is gone), minimum hand-receptacle clearance +1.5 mm with 10 mm body
inflation, grasp margin 0.22.

Derived tolerances (frozen with the clearance, not tuned afterward). The
parcel of depth L_d = 40 mm along d in a slot of height 80 + 2 x 10 mm
fits while its pitch about the loose axis stays under about
2 c_tight / L_d = 0.5 rad, and its roll about d stays under about
2 c_tight / 55 mm = 0.36 rad. The final orientation tolerance of 10 deg
(section 8) is stricter than both and was fixed at M0.

Derived tolerances (frozen with the clearance, not tuned afterward). A
parcel of length L = 80 mm along the insertion axis and width w on the tight
axis fits a slot of width w + 2 c_tight only while its yaw about the slot
normal stays under about 2 c_tight / L = 0.2 rad = 11.5 deg. The final
orientation tolerance is therefore 10 deg (section 8).

## 7. Phase Sequence and Speedup Factor (Frozen, Speedup Grid Frozen at M6)

Phases in order, with the nominal duration at unit rate,

    PARK            0.5 s   fixed
    APPROACH        2.5 s   fixed
    PREGRASP_DWELL  0.6 s   fixed
    CLOSE           1.5 s   fixed
    GRASP_DWELL     0.6 s   fixed
    LIFT            1.2 s   scaled by 1/r
    REORIENT        1.6 s   scaled by 1/r
    TRANSFER        1.6 s   scaled by 1/r
    PREINSERT_DWELL 0.4 s   scaled by 1/r
    INSERT          1.0 s   scaled by 1/r
    INSERT_DWELL    0.4 s   scaled by 1/r
    RELEASE         0.6 s   scaled by 1/r
    RETREAT         1.0 s   scaled by 1/r
    SETTLE          0.6 s   fixed (measurement window)

The speedup factor r scales the manipulation segment (LIFT through RETREAT,
7.8 s at r = 1). The acquisition segment runs at fixed timing so the rate
axis isolates the downstream demand of the same geometric manipulation.
Cycle time at rate r equals 5.7 s + 7.8 s / r + 0.6 s. Phase progress is a
per-environment buffer (phase index k plus in-phase fraction f), advanced
each control step by dt / T_k(r). The geometric path is a function of
(k, f) alone, so changing r changes only the timing. Every learner observes
task_phase = (k + f) / 14 and task_rate = r. INSERT_DWELL entered the
schedule at M4 (freeze log), before any learner ran, after the first expert
sweep showed the arm servo lag releasing the parcel 3 cm short of the
insert pose at r >= 1.

Development slow speed for the M4 validation, r = 0.5. Frozen speedup grid
(M6, from the expert sweep of 64 episodes per speed at 1 cm jitter,
outputs/paper/expert/sweep_summary.jsonl), r in
{0.5, 1.0, 1.5, 2.0, 2.25, 2.5, 3.0}. The expert reads 63/64, 63/64,
64/64, 58/64 at 0.5, 1.0, 1.5, 2.0, then 19/64 at 2.5, 1/64 at 3.0, and 0
at 4 and 5, with arm velocity utilization under 0.3 and peak hand speed
under 0.5 m/s through r = 3, so the upper end stays a credible speed and no
actuator saturates. Cycle times, 21.9 s at 0.5, 14.1 s at 1.0, 10.2 s at
2.0, 8.9 s at 3.0. Episode length 30 s (fallback), the per-environment
task_complete termination ends the episode at the cycle time.

## 8. Physical Success (Frozen)

Stage markers, each latched at the first step its condition holds,

- acquired, parcel center at least 20 mm above its start height while the
  thumb distal phalanx and at least one other distal phalanx each read a
  parcel-filtered contact force above 1 N.
- lifted_clear, parcel center above table top + 60 mm (z >= 0.760 m).
- reoriented, geodesic angle between the parcel orientation and the frozen
  stow orientation under 15 deg at some step before insertion.
- preinsert_reached, parcel center within 30 mm of the pre-insertion object
  waypoint and orientation error under 15 deg.
- inserted, parcel center at least 50 mm past the entrance plane along the
  insertion axis and inside the interior cross-section box (front face at
  least 10 mm inside the receptacle).
- released, after inserted, the summed parcel-filtered distal contact force
  under 0.5 N for at least 0.1 s.
- settled, after released, the parcel remains inserted for a dwell of 0.4 s
  with linear speed under 0.02 m/s and angular speed under 0.2 rad/s.
- task_success = inserted and released and settled and, at the end of the
  settle window, the parcel center inside the interior box and the final
  orientation error under 10 deg.

None of these predicates reads epsilon, epsilon^(beta), the GDF field, or
any bank quantity. A regression test supplies a bogus negative force-closure
margin and asserts task_success is unchanged.

Failure reason, the first category that applies in the order below,

    acquisition_failure          not acquired
    dropped_during_lift          acquired, contact lost before REORIENT starts
    dropped_during_reorientation contact lost during REORIENT
    excessive_inhand_slip        held to insertion, but the hand-object drift
                                 exceeds the slot clearance (translation
                                 above 8 mm or rotation above 10 deg) before
                                 the parcel touches the receptacle, and
                                 insertion fails
    transfer_failure             held, not preinsert_reached
    insertion_misalignment       preinsert_reached, not inserted, parcel
                                 stopped outside the entrance plane with
                                 orientation error above 10 deg
    insertion_jam                preinsert_reached, not inserted, parcel
                                 stopped with parcel-receptacle contact above
                                 2 N for at least 0.2 s, or the hand-object
                                 drift exceeded the clearance only after the
                                 first receptacle contact (the receptacle
                                 moved the parcel in the hand)
    release_failure              inserted, not released
    timeout                      inserted and released, not settled
    other                        anything else

## 9. Slip Diagnostics (Frozen, Diagnostics Only)

At the acquired step the recorder stores T_HO, the parcel pose in the hand
root frame. Every later step reports the relative translation and rotation
against the stored transform. Logged per episode, the maximum translation
(m), the maximum rotation (deg), the transform at the end of REORIENT and
at INSERT start, the drop event, the active distal contact count, and the
contact force magnitudes. No slip threshold defines success.

## 10. Observation and Action Interface (Frozen)

Actions, absolute joint position targets for the 16 joints of CHAIN_ACTUATED
(waist yaw, roll, pitch, right shoulder pitch, roll, yaw, elbow, wrist roll,
pitch, yaw, thumb cmc roll, thumb cmc pitch, index, middle, ring, pinky mcp
pitch), scale 0.5 about the default pose, 50 Hz control at physics 200 Hz
(decimation 4), implicit PD actuators as in the cube tasks (arms stiffness
300, damping 10, hand stiffness 10, damping 0.2).

Observations of every learner, in order,
joint_pos_rel (51), joint_vel_rel (51), last_action (16), parcel pose in the
pelvis frame (position 3, quaternion wxyz 4), distal phalanx positions in
the pelvis frame (15), distal contact force magnitudes clipped and scaled
(5), task_phase (1), task_rate (1). Total 147. No GDF, no bank quantity, no
force-closure margin value. Corruption noise as in the distill task during
demonstration collection, off during evaluation.

## 11. Expert Construction (Frozen in Method)

Acquisition, pregrasp and grasp chain configurations from the parcel bank
entry nearest to the parcel start pose (planar grid, section 11.1), hand
shape from the bank, cosine blends between phase targets, integral sag
correction only during dwell segments (ki 0.08, clamp 0.35 rad), the same
scheme as scripts/vla/expert.py.

Manipulation, task-space object waypoints T_WO^d(k, f) along the frozen
path (lift by 80 mm, SLERP reorientation by 90 deg about the parcel center,
smooth translation to the pre-insertion pose, straight insertion along the
receptacle axis, release by opening the hand to the bank pregrasp shape, retreat
along the reverse of the insertion axis). Desired hand pose T_WH^d = T_WO^d
X_OH with X_OH the bank grasp transform (verified numerically by forward
kinematics of the solved grasp), solved by damped least squares IK on
rh_hand_base_link at fixed knots, stored in assets/parcel_stow_trajectory.json
with the desired hand pose, the solved joint target, the IK position and
orientation errors, and the joint-limit margin. Between knots the expert
interpolates joint targets. The object is never attached to the desired
transform.

### 11.1 Start-Pose Grid

The bank holds the acquisition knots re-solved on a planar grid of start
offsets, dx and dy in {-15, -10, -5, 0, 5, 10, 15} mm, and the LIFT knot per
entry. From REORIENT onward the joint path is the nominal one plus an offset
that decays to zero over the REORIENT phase.

## 12. Training Distribution (Frozen at M6)

- Start jitter, uniform dx, dy in [-10, 10] mm, no yaw jitter.
- Speedup factor, uniform over [0.5, 2.0], the range over which the expert
  succeeds in at least 0.9 of its episodes (its feasible range without the
  transition zone 2.25 to 3.0). Grid speeds 0.5 to 2.0 are therefore
  in-distribution for every learner and 2.25, 2.5, 3.0 test extrapolation
  beyond the demonstrated speed range for the learners while the expert
  itself degrades there.
- Demonstrations, complete expert episodes through SETTLE, admitted by
  physical task_success only, 300 expert episodes collected once, saved to
  outputs/paper/demos/expert_episodes.pt, and reused by every
  learner.
- Observation corruption on during collection, as in the cube protocol.

## 13. Evaluation Distribution (Frozen at M6)

- The same jitter law as training, seeds fixed per speed (12345 + 1000 x
  speed index) and identical across policies.
- Speedup grid {0.5, 1.0, 1.5, 2.0, 2.25, 2.5, 3.0}, 100 episodes per
  policy and speed in the final run, 32 environments per process.
- Corruption off.

## 14. Primary Metrics (Frozen)

- P(task_success | rate) with Wilson 95 percent intervals, per actor.
- Stage completion probabilities per rate (acquired, reoriented, inserted,
  released and settled).
- Failure reason distribution per actor and rate.
- Slip distributions (max translation, max rotation) per actor and rate.
- Secondary scalars, area under the success-versus-rate curve and the
  largest grid rate at which success stays at or above 0.8.
- Force-closure diagnostics, epsilon and epsilon^(beta) (beta 0.95, Gaussian
  friction prior with std 0.15 about 0.5) on the realized distal contact
  set at acquired, at the end of REORIENT, and at INSERT start, correlated
  with outcome by Spearman rank and by success stratified on margin
  quantile.

## 15. Freeze Log

- 2026-08-18 M0, sections 1 to 5, 7 (structure), 8, 9, 10, 11, 14 frozen.
  Section 6 receptacle geometry pending the probe. Sections 12 and 13
  pending the expert rate sweep.
- 2026-08-18 M3 (second pass), the record of the frozen grasp changed to
  the riser-0 lab-scene synthesis
  (assets/provenance/frogger_parcel/scene_lab_x0.35_riser0/parcel_80x55x40.json,
  five registered pads, eps_nominal 0.0019, eps_beta(0.95) 0.00078), start
  yaw +45 deg, shelf 0.16 m, lift 0.12 m, entrance (0.491, 0.141),
  c_tight 10 mm, wall thickness 20 mm (a 10 mm kinematic wall let a
  scripted parcel tunnel in the receptacle test), receptacle slabs are
  kinematic rigid bodies (the parcel-filtered contact sensor resolves them).
  Reason for the record change, the riser-15 record puts the thumb 8 mm
  below the flush table so the closing thumb jammed against the table and
  the parcel slid 5 cm in the hand at lift onset (measured, first M4 run,
  0 of 20). The physical validation with the riser-0 record passed 20 of 20
  and then 32 of 32 at r 0.5.
- 2026-08-18 M6, rate grid {0.5, 1.0, 1.5, 2.0, 2.25, 2.5, 3.0}, training
  rate range [0.5, 2.0], evaluation seeds, from the second expert sweep,
  before any learner ran (sections 7, 12, 13).
- 2026-08-18 M4, INSERT_DWELL 0.4 s (rate scaled) added between INSERT and
  RELEASE after the first rate sweep showed release before the arm reached
  the insert pose (depth 31 mm of 60 at r >= 1, then the parcel toppled).
  Batches run at 32 environments per process, at 50 and 64 environments
  the GPU PhysX pipeline produced env-index bands with different contact
  outcomes under identical inputs (documented in IMPLEMENTATION_LOG.md).
- 2026-08-18 M3, start pose moved from (0.43, 0) to (0.35, 0) with start
  yaw -45 deg (section 4), reason kinematic (wrist pitch limit at the cube
  spot, best reorientation margins at -45 deg). Section 6 frozen from
  the probe (family C, shelf 0.10 m, c_loose raised from 25 to 55 mm after
  the probe measured phalanx-wall penetration at 25 and 45 mm, depth slack
  raised from 40 to 70 mm and the back margin from 5 to 30 mm after the
  thumb tip frame reached the back wall, reorient_travel 0.5). All before
  any learner ran.
