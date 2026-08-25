"""ParcelStow task package, dexterous parcel reorientation and stowing.
The task registers only when the frozen geometry asset exists, so the older
tasks import without error on a checkout without it."""

import os

from . import geometry as G

if os.path.exists(G.GEOMETRY_PATH):
    from . import parcel_stow_env_cfg  # noqa: F401
else:
    print(f"[WARN] parcel_stow, geometry asset {G.GEOMETRY_PATH} missing, task not registered")
