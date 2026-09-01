"""Task-level phase schedule for the speedup-factor clock.

A task defines its phases as (name, nominal_seconds, rate_scaled)
tuples, the shape of the ParcelStow geometry.PHASES table, and binds a
PhaseSchedule built from them to task_clock.SCHEDULE when its package
loads. The speedup factor r divides the durations of the rate-scaled
phases and leaves the fixed phases unchanged.
"""

import numpy as np


class PhaseSchedule:
    def __init__(self, phases):
        self.phases = [(str(n), float(d), bool(s)) for n, d, s in phases]
        self.names = [p[0] for p in self.phases]
        self.index = {n: i for i, n in enumerate(self.names)}
        self.nominal_durations = np.array([p[1] for p in self.phases])
        self.rate_scaled = np.array([p[2] for p in self.phases], dtype=bool)
        self.n_phases = len(self.phases)

    def phase_durations(self, rate):
        """(n_phases,) durations in seconds at the given speedup factor."""
        d = self.nominal_durations.copy()
        d[self.rate_scaled] /= float(rate)
        return d

    def cycle_time(self, rate):
        return float(self.phase_durations(rate).sum())
