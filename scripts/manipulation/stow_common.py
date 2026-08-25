"""Loader shim, imports the ParcelStow geometry module by file path so the
manipulation scripts and the pure tests share one module without touching
the task package __init__ (which needs the simulator). Scripts that run
after the app launch may import the package module directly instead."""

import importlib.util
import os
import sys

_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "source", "parcelstow", "parcelstow",
    "tasks", "manager_based", "parcel_stow", "geometry.py"))
_NAME = "parcelstow.tasks.manager_based.parcel_stow.geometry"
if _NAME in sys.modules:
    _mod = sys.modules[_NAME]
else:
    _spec = importlib.util.spec_from_file_location(_NAME, _PATH)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_NAME] = _mod
    _spec.loader.exec_module(_mod)
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("__")})
