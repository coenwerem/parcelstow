# DAgger Three-Task Extension Protocol

## Scope

Train one DAgger policy each for upright placement and keyed peg insertion. Parcel insertion already has a released DAgger policy. The extension completes the same three-method evaluation matrix used for ACT and Diffusion Policy; it does not change a task, expert, demonstration set, monitor, existing policy, or protected evaluation record.

The final checkpoint from one predeclared training run is the reported policy for each task. There is no seed search, checkpoint search, nominal-success threshold, retraining decision based on rollout performance, or substitution after final evaluation. Report both policies, including a policy with low or zero success.

## Inputs

Use the existing admitted expert demonstrations without recollection or additional filtering:

- Upright placement: `outputs/upright/demos/expert_episodes.pt`, demonstrated rate interval `[0.75, 1.75]`.
- Keyed peg insertion: `outputs/peg/demos/expert_episodes.pt`, demonstrated rate interval `[0.5, 1.0]`.

Record each demonstration file's SHA-256 digest before optimization. Require 147-dimensional observations and 16-dimensional normalized joint-position actions. Reject a demonstration set with different dimensions.

## Training

Use the released parcel-insertion DAgger configuration for both tasks:

- student: multilayer perceptron with widths 512, 256, and 128 and ELU activations;
- loss: mean squared error on the expert's 16-dimensional action;
- optimizer: Adam with learning rate `1e-3`;
- initial behavior-cloning fit: 40 epochs, batch size 4096;
- aggregation: four rounds of 100 complete student episodes;
- exploration: Gaussian action noise with standard deviation 0.1 on half of the parallel environments;
- labeling: the task-specific scripted expert labels every visited observation;
- refit: reinitialize and fit the same student on the complete aggregate for 40 epochs after each round;
- training seed: 1;
- parallel environments: 32;
- reset jitter: 10 mm;
- aggregation rates: sampled uniformly from the task's demonstrated rate interval.

The training command runs a 100-episode nominal diagnostic after behavior cloning and after each aggregation round. Diagnostics use development seed 42001. They verify execution and record failure modes; they do not select a checkpoint or authorize a new run. The final policy is `student_final.pt` after round four.

Use empty output directories:

- `outputs/iclr_dagger_three_task/upright`
- `outputs/iclr_dagger_three_task/peg`

Run upright placement first. Start keyed peg insertion only after upright training, its 100-episode diagnostic, and its checkpoint digest complete successfully.

## Commands

Upright placement:

```bash
ISAACLAB_VENV=../IsaacLab/.venv \
scripts/isaac_run.sh outputs/iclr_dagger_three_task/upright_train.log \
python scripts/manipulation/run_stow_distill.py \
  --task UprightPlace-L6-Play-v0 \
  --demos outputs/upright/demos/expert_episodes.pt \
  --out_dir outputs/iclr_dagger_three_task/upright \
  --rate_lo 0.75 --rate_hi 1.75 \
  --dagger_rounds 4 --dagger_episodes 100 \
  --diag_episodes 100 --diag_rate 1.0 \
  --num_envs 32 --jitter 0.01 --action_noise 0.1 \
  --epochs 40 --batch 4096 --lr 0.001 \
  --train_seed 1 --eval_seed 42001
```

Keyed peg insertion:

```bash
ISAACLAB_VENV=../IsaacLab/.venv \
scripts/isaac_run.sh outputs/iclr_dagger_three_task/peg_train.log \
python scripts/manipulation/run_stow_distill.py \
  --task PegInsert-L6-Play-v0 \
  --demos outputs/peg/demos/expert_episodes.pt \
  --out_dir outputs/iclr_dagger_three_task/peg \
  --rate_lo 0.5 --rate_hi 1.0 \
  --dagger_rounds 4 --dagger_episodes 100 \
  --diag_episodes 100 --diag_rate 1.0 \
  --num_envs 32 --jitter 0.01 --action_noise 0.1 \
  --epochs 40 --batch 4096 --lr 0.001 \
  --train_seed 1 --eval_seed 42001
```

## Protected Evaluation

Fix and record both final checkpoint hashes before evaluating either policy. Evaluate only the new DAgger actor; reuse the frozen expert, ACT, and Diffusion Policy records from `outputs/iclr_dp_three_task_20260919/raw/final`. For each task, use its eight registered rates, 200 episodes per rate, the final initial-condition bank, evaluation seed 73001, 10 mm reset jitter, no observation corruption, and no action noise.

Upright placement:

```bash
ISAACLAB_VENV=../IsaacLab/.venv \
scripts/isaac_run.sh outputs/iclr_dagger_three_task/upright_final.log \
python scripts/evaluate.py --task upright --actor dagger \
  --rates 0.5 0.75 1.0 1.25 1.5 1.75 2.0 2.5 \
  --episodes 200 --num_envs 32 --eval_seed 73001 --jitter 0.01 \
  --bank_role final --tag _protocol-1 \
  --dagger_ckpt outputs/iclr_dagger_three_task/upright/student_final.pt \
  --out_dir outputs/iclr_dagger_three_task/final/upright
```

Keyed peg insertion:

```bash
ISAACLAB_VENV=../IsaacLab/.venv \
scripts/isaac_run.sh outputs/iclr_dagger_three_task/peg_final.log \
python scripts/evaluate.py --task peg --actor dagger \
  --rates 0.5 0.75 1.0 1.25 1.5 1.75 2.0 2.5 \
  --episodes 200 --num_envs 32 --eval_seed 73001 --jitter 0.01 \
  --bank_role final --tag _protocol-1 \
  --dagger_ckpt outputs/iclr_dagger_three_task/peg/student_final.pt \
  --out_dir outputs/iclr_dagger_three_task/final/peg
```

Before analysis, require equality of the DAgger and frozen comparator initial-condition identifiers, initial object poses, reset seeds, rates, episode counts, and bank hashes at every task-rate condition.

## Reporting

Report every rate and policy. For each DAgger condition, report success count, Wilson 95% interval, signed DAgger-minus-expert paired difference, 20,000-resample paired bootstrap 95% interval, stage-completion counts, and terminal failure counts. Do not pool tasks, rates, policy seeds, or repeated expert rows as independent episodes. Do not apply a nominal-parity or noninferiority decision rule.
