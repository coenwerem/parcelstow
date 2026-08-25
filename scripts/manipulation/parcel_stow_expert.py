"""Scripted expert of the ParcelStow task, pure torch, importable without
the simulator.

The expert drives the 16 actuated joints of CHAIN_ACTUATED through the
phase schedule of the task (geometry.PHASES). Acquisition targets come from
the parcel bank (assets/gdf_bank_parcel.json), the pregrasp and grasp chain
configurations of the planar start-offset grid entry nearest to the actual
parcel start, with the bank hand shapes. Manipulation targets come from the
IK trajectory (assets/parcel_stow_trajectory.json), whose knots were solved
for the desired hand poses T_WH = T_WO X_OH along the frozen object path.
Between knots the expert interpolates joint targets linearly in the
in-phase fraction, and the phase fraction itself follows the cosine ease of
the object path inside the knots (the knots are dense enough that linear
joint interpolation between them tracks the eased task-space path).

The per-environment offset between the pose-conditioned lift knot and the
nominal lift knot decays to zero over the REORIENT phase, so the joint path
from TRANSFER onward is the nominal one for every start offset.

Phase targets,
  PARK             default pose
  APPROACH         cosine blend default -> pregrasp (hand open)
  PREGRASP_DWELL   pregrasp
  CLOSE            cosine blend pregrasp -> grasp (arm and hand)
  GRASP_DWELL      grasp
  LIFT             nominal lift knots plus decaying grid offset, hand grasp
  REORIENT         nominal knots plus decaying lift offset, hand grasp
  TRANSFER         nominal knots, hand grasp
  PREINSERT_DWELL  nominal knot, hand grasp
  INSERT           nominal knots, hand grasp
  INSERT_DWELL     insert knot, hand grasp (the arm settles before release)
  RELEASE          arm at the insert knot, hand cosine blend grasp -> open
  RETREAT          nominal retreat knots, hand open
  SETTLE           arm at the retreat end, hand open

The object is never attached to anything. Physics decides where it goes.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BANK_PATH = os.path.join(REPO, "assets", "gdf_bank_parcel.json")
TRAJECTORY_PATH = os.path.join(REPO, "assets", "parcel_stow_trajectory.json")

PHASE_NAMES = ["PARK", "APPROACH", "PREGRASP_DWELL", "CLOSE", "GRASP_DWELL", "LIFT", "REORIENT",
               "TRANSFER", "PREINSERT_DWELL", "INSERT", "INSERT_DWELL", "RELEASE", "RETREAT", "SETTLE"]
PH = {n: i for i, n in enumerate(PHASE_NAMES)}
DWELL_PHASES = ("PREGRASP_DWELL", "GRASP_DWELL", "PREINSERT_DWELL", "INSERT_DWELL", "SETTLE")
KI = 0.08
CORR_CLAMP = 0.35


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def _named_vector(names, *dicts):
    out = []
    for name in names:
        for d in dicts:
            if name in d:
                out.append(float(d[name]))
                break
        else:
            raise KeyError(f"joint {name} absent from the record")
    return out


def _ease(f):
    return 0.5 * (1.0 - torch.cos(math.pi * f.clamp(0.0, 1.0)))


class StowExpert:
    """Batched expert, targets as (E, 16) tensors in actuated order."""

    def __init__(self, actuated_names, bank=None, trajectory=None, device="cpu", candidate=0,
                 ki=KI, corr_clamp=CORR_CLAMP):
        self.names = list(actuated_names)
        self.device = device
        self.ki = ki
        self.corr_clamp = corr_clamp
        bank = bank if bank is not None else load_json(BANK_PATH)
        traj = trajectory if trajectory is not None else load_json(TRAJECTORY_PATH)
        self.bank = bank
        self.traj = traj
        chain = bank["chain_joint_names"]
        self.n_arm = len(chain)
        self.arm_idx = [self.names.index(n) for n in chain]
        hand_names = [n for n in self.names if n not in chain]
        self.hand_idx = [self.names.index(n) for n in hand_names]
        self.hand_open = torch.tensor(_named_vector(hand_names, bank["hand_pregrasp"]), dtype=torch.float32, device=device)
        self.hand_grasp = torch.tensor(_named_vector(hand_names, bank["hand_grasp"]), dtype=torch.float32, device=device)
        cand = bank["candidates"][candidate]
        self.q_pre_nom = torch.tensor(_named_vector(chain, cand["q_chain"]), dtype=torch.float32, device=device)
        self.q_grasp_nom = torch.tensor(_named_vector(chain, cand["q_chain_grasp"]), dtype=torch.float32, device=device)
        # grid entries
        entries = [e for e in bank["grid"]["entries"] if e["ok"]]
        self.grid_xy = torch.tensor([[e["dx"], e["dy"]] for e in entries], dtype=torch.float32, device=device)
        self.grid_pre = torch.tensor([_named_vector(chain, e["q_chain"]) for e in entries], dtype=torch.float32, device=device)
        self.grid_grasp = torch.tensor([_named_vector(chain, e["q_chain_grasp"]) for e in entries], dtype=torch.float32, device=device)
        self.grid_lift = torch.tensor([_named_vector(chain, e["q_chain_lift"]) for e in entries], dtype=torch.float32, device=device)
        # nominal knots per phase, (fractions tensor, q tensor (K, n_arm))
        self.knots = {}
        for k in traj["knots"]:
            self.knots.setdefault(k["phase"], []).append((float(k["f"]), _named_vector(chain, k["q_chain"])))
        self.phase_knots = {}
        for name, rows in self.knots.items():
            rows.sort(key=lambda r: r[0])
            fr = torch.tensor([r[0] for r in rows], dtype=torch.float32, device=device)
            qs = torch.tensor([r[1] for r in rows], dtype=torch.float32, device=device)
            self.phase_knots[name] = (fr, qs)
        self.q_lift_nom = self.phase_knots["LIFT"][1][-1]
        self.q_grasp_traj = self.phase_knots["LIFT"][1][0]
        self.q_insert_nom = self.phase_knots["INSERT"][1][-1]
        self.q_retreat_end = self.phase_knots["RETREAT"][1][-1]
        self.E = None

    # ------------------------------------------------------------------
    def reset(self, ids, start_xy_offset):
        """ids, environment indices, start_xy_offset (len(ids), 2), the
        parcel start offset from the nominal start in the world xy plane."""
        if self.E is None:
            raise RuntimeError("call allocate(E) first")
        ids = torch.as_tensor(list(ids), dtype=torch.long, device=self.device)
        if len(ids) == 0:
            return
        off = torch.as_tensor(start_xy_offset, dtype=torch.float32, device=self.device).view(-1, 2)
        d = (off.unsqueeze(1) - self.grid_xy.unsqueeze(0)).norm(dim=-1)
        j = d.argmin(dim=1)
        self.q_pre[ids] = self.grid_pre[j]
        self.q_grasp[ids] = self.grid_grasp[j]
        self.q_lift[ids] = self.grid_lift[j]
        self.grid_index[ids] = j
        self.corr[ids] = 0.0
        self.hand_corr_zeroed[ids] = False

    def allocate(self, E):
        self.E = E
        d = self.device
        self.q_pre = self.q_pre_nom.unsqueeze(0).repeat(E, 1)
        self.q_grasp = self.q_grasp_nom.unsqueeze(0).repeat(E, 1)
        self.q_lift = self.q_lift_nom.unsqueeze(0).repeat(E, 1)
        self.grid_index = torch.zeros(E, dtype=torch.long, device=d)
        self.corr = torch.zeros(E, len(self.names), device=d)
        self.hand_corr_zeroed = torch.zeros(E, dtype=torch.bool, device=d)

    # ------------------------------------------------------------------
    def _interp(self, name, f):
        """Linear interpolation of the nominal knots of a phase, f (E,) ->
        (E, n_arm)."""
        fr, qs = self.phase_knots[name]
        f = f.clamp(0.0, 1.0)
        idx = torch.searchsorted(fr, f, right=True).clamp(1, len(fr) - 1)
        f0 = fr[idx - 1]
        f1 = fr[idx]
        w = ((f - f0) / (f1 - f0).clamp(min=1e-6)).unsqueeze(1)
        return qs[idx - 1] * (1 - w) + qs[idx] * w

    def target(self, k, f, q_default):
        """Joint target (E, 16) for phase indices k (E,) and fractions f (E,)."""
        E = k.shape[0]
        out = q_default.clone()
        arm = torch.zeros(E, self.n_arm, device=self.device)
        hand = self.hand_open.unsqueeze(0).repeat(E, 1)
        s = _ease(f).unsqueeze(1)
        arm_default = q_default[:, self.arm_idx]
        hand_default = q_default[:, self.hand_idx]
        # acquisition
        arm = torch.where((k == PH["PARK"]).unsqueeze(1), arm_default, arm)
        hand = torch.where((k == PH["PARK"]).unsqueeze(1), hand_default, hand)
        m = (k == PH["APPROACH"]).unsqueeze(1)
        arm = torch.where(m, arm_default + s * (self.q_pre - arm_default), arm)
        hand = torch.where(m, hand_default + s * (self.hand_open - hand_default), hand)
        m = (k == PH["PREGRASP_DWELL"]).unsqueeze(1)
        arm = torch.where(m, self.q_pre, arm)
        m = (k == PH["CLOSE"]).unsqueeze(1)
        arm = torch.where(m, self.q_pre + s * (self.q_grasp - self.q_pre), arm)
        hand = torch.where(m, self.hand_open + s * (self.hand_grasp - self.hand_open), hand)
        m = (k == PH["GRASP_DWELL"]).unsqueeze(1)
        arm = torch.where(m, self.q_grasp, arm)
        hand = torch.where(m, self.hand_grasp, hand)
        # manipulation, nominal knots plus decaying grid offsets
        m = (k == PH["LIFT"]).unsqueeze(1)
        nom = self._interp("LIFT", f)
        off = (1 - s) * (self.q_grasp - self.q_grasp_traj) + s * (self.q_lift - self.q_lift_nom)
        arm = torch.where(m, nom + off, arm)
        hand = torch.where(m, self.hand_grasp, hand)
        m = (k == PH["REORIENT"]).unsqueeze(1)
        nom = self._interp("REORIENT", f)
        off = (1 - s) * (self.q_lift - self.q_lift_nom)
        arm = torch.where(m, nom + off, arm)
        hand = torch.where(m, self.hand_grasp, hand)
        for name in ("TRANSFER", "PREINSERT_DWELL", "INSERT", "INSERT_DWELL"):
            m = (k == PH[name]).unsqueeze(1)
            arm = torch.where(m, self._interp(name, f), arm)
            hand = torch.where(m, self.hand_grasp, hand)
        m = (k == PH["RELEASE"]).unsqueeze(1)
        arm = torch.where(m, self.q_insert_nom.unsqueeze(0).expand(E, -1), arm)
        hand = torch.where(m, self.hand_grasp + s * (self.hand_open - self.hand_grasp), hand)
        m = (k == PH["RETREAT"]).unsqueeze(1)
        arm = torch.where(m, self._interp("RETREAT", f), arm)
        m = (k == PH["SETTLE"]).unsqueeze(1)
        arm = torch.where(m, self.q_retreat_end.unsqueeze(0).expand(E, -1), arm)
        out[:, self.arm_idx] = arm
        out[:, self.hand_idx] = hand
        return out

    def act(self, k, f, q_default, q_measured):
        """Action (E, 16) with the integral sag correction of the dwell
        segments (arm and hand in the pregrasp and grasp dwells, arm only
        afterward, hand correction zeroed at RELEASE)."""
        q_t = self.target(k, f, q_default)
        dwell = torch.zeros_like(k, dtype=torch.bool)
        for name in DWELL_PHASES:
            dwell |= k == PH[name]
        early = (k == PH["PREGRASP_DWELL"]) | (k == PH["GRASP_DWELL"])
        upd = self.ki * (q_t - q_measured)
        new_corr = self.corr.clone()
        arm_mask = dwell.unsqueeze(1)
        new_corr[:, self.arm_idx] = torch.where(arm_mask, self.corr[:, self.arm_idx] + upd[:, self.arm_idx],
                                                self.corr[:, self.arm_idx])
        hand_mask = early.unsqueeze(1)
        new_corr[:, self.hand_idx] = torch.where(hand_mask, self.corr[:, self.hand_idx] + upd[:, self.hand_idx],
                                                 self.corr[:, self.hand_idx])
        zero_hand = (k >= PH["RELEASE"]).unsqueeze(1)
        new_corr[:, self.hand_idx] = torch.where(zero_hand, torch.zeros_like(new_corr[:, self.hand_idx]),
                                                 new_corr[:, self.hand_idx])
        self.corr = new_corr.clamp(-self.corr_clamp, self.corr_clamp)
        return to_action(q_t + self.corr, q_default), q_t


def to_action(q_target, q_default):
    """Invert the JointPositionAction map, q = 0.5 * a + q_default."""
    return 2.0 * (q_target - q_default)
