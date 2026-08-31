"""Minimal ParcelStow policy showing the evaluation interface.

The evaluator accepts any class reachable as module.path:ClassName whose
instances expose the actor interface of docs/POLICY_INTERFACE.md,

  name                       string stored in every episode record
  reset(ids, obs=None)       called with the env indices being reset
  act(obs) -> (action, q_t)  action (n, 16), q_t (n, 16) joint targets
                             or None when the policy has no explicit
                             joint-space plan

The observation is a (n, 147) torch tensor on the environment device,
slices documented in docs/POLICY_INTERFACE.md. The action is the
normalized joint-position command of the 16 actuated joints, the
environment applies target = 0.5 * action + q_default at 50 Hz.

The policy below commands the default posture, so it holds still and
fails every episode. It exists to show the integration point, replace
act() with your model call.

Run it on the frozen evaluation draws,

  python scripts/evaluate.py --actor examples.custom_policy:HoldPosturePolicy \
      --rates 1.0 --episodes 5 --num_envs 8
"""

import torch


class HoldPosturePolicy:
    name = "hold_posture"

    def __init__(self, base, checkpoint=None, num_envs=None):
        # base is the ManagerBasedRLEnv, checkpoint the --custom_ckpt path
        self.base = base
        self.n = num_envs or base.num_envs

    def reset(self, ids, obs=None):
        # per-env recurrent state would reset here
        pass

    @torch.no_grad()
    def act(self, obs):
        # obs[:, 146] holds the speedup factor r. A policy may condition
        # its action on this value.
        action = torch.zeros(self.n, 16, device=self.base.device)
        return action, None
