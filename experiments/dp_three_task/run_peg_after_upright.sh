#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

while tmux has-session -t iclr_dp_upright 2>/dev/null; do
  sleep 30
done

upright_results=outputs/iclr_dp/upright/results.jsonl
upright_diagnostic=outputs/iclr_dp/upright/diag.jsonl
if [[ ! -f "$upright_results" || ! -f "$upright_diagnostic" ]] ||
   ! rg -q '"stage": "diag_eval"' "$upright_results" ||
   [[ $(wc -l < "$upright_diagnostic") -ne 100 ]]; then
  printf '%s\n' 'Upright run did not complete its 100-episode diagnostic; peg training was not started.' >&2
  exit 1
fi

if [[ -e outputs/iclr_dp/peg ]]; then
  printf '%s\n' 'Peg output directory already exists; refusing to overwrite it.' >&2
  exit 1
fi

export ISAACLAB_VENV=/home/drce/ResearchProjects/IsaacLab/.venv
exec scripts/isaac_run.sh outputs/iclr_dp/peg_train.log \
  /home/drce/ResearchProjects/IsaacLab/.venv/bin/python \
  scripts/manipulation/run_stow_diffusion_policy.py \
  --task PegInsert-L6-Play-v0 \
  --demos /home/drce/ResearchProjects/parcelstow/outputs/peg/demos/expert_episodes.pt \
  --out_dir outputs/iclr_dp/peg --tag peg --epochs 300 \
  --diag_episodes 100 --num_envs 32 --eval_seed 42001
