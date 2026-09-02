"""MDP terms of the ParcelStow task."""

from isaaclab.envs.mdp import *  # noqa: F401, F403

from .guards import *  # noqa: F401, F403
from .contacts import fingertip_pos_b, tip_contact_forces, TIP_BODIES  # noqa: F401
from .task_clock import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .events import *  # noqa: F401, F403
from .terminations import *  # noqa: F401, F403

# Bind the frozen ParcelStow schedule to the task clock.
from parcelstow.phase_schedule import PhaseSchedule as _PhaseSchedule  # noqa: E402
from .. import geometry as _geometry  # noqa: E402
from . import task_clock as _task_clock  # noqa: E402

_task_clock.SCHEDULE = _PhaseSchedule(_geometry.PHASES)
