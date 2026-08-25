#!/bin/bash
# Run one Isaac command with the lab environment, logging to the given
# file. Usage, isaac_run.sh <logfile> <command...>
# Set ISAACLAB_VENV to the Isaac Lab virtual environment. When unset the
# script uses the already active Python environment.
LOG="$1"; shift
cd "$(dirname "$0")/.."
if [ -n "$ISAACLAB_VENV" ]; then
    source "$ISAACLAB_VENV/bin/activate"
fi
export OMNI_KIT_ACCEPT_EULA=YES WANDB_MODE=disabled
unset PYTHONPATH
mkdir -p "$(dirname "$LOG")"
# lowered priority and a bounded core set so the desktop stays usable,
# override with ISAAC_CORES (default 0-7) and ISAAC_NICE (default 10)
CORES="${ISAAC_CORES:-0-7}"
NICE="${ISAAC_NICE:-10}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
nice -n "$NICE" taskset -c "$CORES" "$@" > "$LOG" 2>&1
echo "exit $?" >> "$LOG"
