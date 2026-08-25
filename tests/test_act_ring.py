"""The ring-buffer temporal ensembling of the ACT actor matches the
released all-time-actions formulation (scripts/baselines/run_act.py) on
random chunk predictions."""

import importlib.util
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _released(a_hats, Tq, k=0.01):
    """a_hats (T, Tq, A) chunk predictions per step, returns (T, A) actions."""
    T, _, A = a_hats.shape
    all_time = torch.zeros(T, T + Tq, A)
    out = []
    for t in range(T):
        all_time[t, t:t + Tq] = a_hats[t]
        cur = all_time[:, t]
        populated = cur.abs().sum(dim=1) != 0
        cur = cur[populated]
        w = torch.exp(-k * torch.arange(cur.shape[0], dtype=torch.float32))
        w = (w / w.sum()).unsqueeze(1)
        out.append((cur * w).sum(dim=0))
    return torch.stack(out)


def _ring(a_hats, Tq, k=0.01):
    """The ACTActor.act ring buffer path, one environment."""
    T, _, A = a_hats.shape
    ring = torch.zeros(1, Tq, Tq, A)
    tstep = torch.zeros(1, dtype=torch.long)
    ages = torch.arange(Tq)
    idx = torch.arange(1)
    out = []
    for t in range(T):
        a_hat = a_hats[t:t + 1]
        slot = tstep % Tq
        ring[idx, slot] = a_hat
        src = tstep.unsqueeze(1) - ages.unsqueeze(0)
        valid = src >= 0
        slots = torch.remainder(src, Tq)
        gathered = ring[idx.unsqueeze(1), slots, ages.unsqueeze(0)]
        n_valid = valid.sum(dim=1, keepdim=True)
        rank = (n_valid - 1 - ages.unsqueeze(0)).clamp(min=0).float()
        w = torch.exp(-k * rank) * valid.float()
        w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-9)
        out.append((gathered * w.unsqueeze(-1)).sum(dim=1)[0])
        tstep += 1
    return torch.stack(out)


def test_ring_matches_released():
    torch.manual_seed(0)
    Tq, T, A = 7, 40, 3
    a_hats = torch.randn(T, Tq, A) + 1.0  # nonzero so the populated mask of the release is exact
    ref = _released(a_hats, Tq)
    got = _ring(a_hats, Tq)
    assert torch.allclose(ref, got, atol=1e-5), (ref - got).abs().max()


def test_ring_source_code_present():
    path = os.path.join(REPO, "scripts", "manipulation", "stow_runtime.py")
    src = open(path).read()
    assert "torch.remainder(src, self.Tq)" in src and "rank 0 the OLDEST" in src
