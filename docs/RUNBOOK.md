# ParcelStow Reproduction Runbook

This runbook covers the three-task evidence in `data/manuscript_20260921/`:
scripted experts, ACT, Diffusion Policy, and DAgger. It separates
**recomputing results from saved records**, **evaluating saved checkpoints**, and
**training new replicas**. Only the first reproduces the exact saved counts.
Simulation and training reruns are new records and may differ numerically.

## 1. Which evidence to use

| Task | Demonstrated speedup factor | Final grid | Episodes per actor and factor | Final bank base seed |
|---|---|---|---:|---:|
| Parcel insertion | `[0.5, 2]` | `0.5 1 1.5 2 2.25 2.5 3` | 100 | 12345, recorded initial poses |
| Upright placement | `[0.75, 1.75]` | `0.5 0.75 1 1.25 1.5 1.75 2 2.5` | 200 | 73001, indexed bank |
| Keyed peg insertion | `[0.5, 1]` | `0.5 0.75 1 1.25 1.5 1.75 2 2.5` | 200 | 73001, indexed bank |

The per-factor seed is `base_seed + 1000 * factor_index`, with a zero-based
index in the **complete ordered grid**. Thus nominal final seeds are 13345 for
parcel and 75001 for upright/peg. A one-factor run with the same base seed does
not reproduce the nominal rows of the complete grid. Different factors are not
paired with one another.

| Task | Expert at `r=1` | Primary ACT at `r=1` | Highlighted factor | Expert | Primary ACT |
|---|---:|---:|---:|---:|---:|
| Parcel | 100/100 | 100/100 | 2 | 84/100 | 53/100 |
| Upright | 185/200 | 194/200 | 2 | 105/200 | 24/200 |
| Peg | 182/200 | 191/200 | 1.5 | 150/200 | 3/200 |

Expert-minus-ACT paired differences at the highlighted factors are 31, 40.5,
and 73.5 percentage points. Their pointwise 95% paired bootstrap intervals are
`[18,44]`, `[32,49]`, and `[67.5,79.5]`. Parcel `r=2` is demonstrated; upright
`r=2` and peg `r=1.5` are extrapolation. These are observed rankings, not tests
of population equivalence or architectural superiority.

## 2. File map and checksums

The code repository includes [data/manuscript_20260921](../data/manuscript_20260921):

| File | Purpose |
|---|---|
| `MANUSCRIPT_EVIDENCE.csv` | 136 canonical task × actor × factor conditions; checkpoint/dataset hashes, source line spans, success/stage/failure counts, audit status, evidence role and claim limits |
| `EXCLUDED_SERIES.csv` | excluded overlapping, superseded, development and pilot series, with reasons |
| `PAIRING_STATUS.csv` | 113 learner conditions; exact expert source, pairing key, paired count, mismatches and permitted inference |
| `FILES.json` | original source path → packaged path, packaged SHA-256, original SHA-256 and compression method |
| `records/S*.jsonl.gz` | 20 complete source series, without concatenation, filtering or relabeling |
| `ARTIFACT_INDEX.csv` | all 17 canonical learner checkpoints and five demonstration files, local/HF paths, sizes and SHA-256 |
| `provenance/act/` | six training configurations, candidate inventories and frozen selection reports |
| `provenance/` | task training protocols and DAgger validation/implementation-audit reports |

Original `.jsonl.gz` sources are copied byte-for-byte. Uncompressed `.jsonl`
sources are gzip-compressed for distribution; decompression must reproduce the
original source bytes and SHA-256. Canonical CSV files retain their original
paths and values. Resolve those paths through `FILES.json`, not by guessing a
filename under `outputs/`.

The HF copy is `evaluation/manuscript_20260921/`. Both copies contain the same
records and catalogs. Legacy source locations are listed below for the research
checkout; they need not exist on a fresh clone because the packaged records do.

| Evidence | Original location |
|---|---|
| Parcel expert/ACT/DP/DAgger | `data/records/{expert,act,dp,dagger}_episodes.jsonl.gz` |
| Parcel ACT replications/subsets | `data/records/replication/{act_seed2,act_seed3,act_n50,act_n100}.jsonl.gz` |
| Upright/peg expert, primary ACT, DP | `outputs/iclr_dp_three_task_20260919/raw/final/{upright,peg}/{expert,act,dp}_protocol-1.jsonl` |
| Upright/peg ACT seeds 1 and 2 | `outputs/act_comparability/TASK/seed-SEED/final/act_protocol-descriptive-20260911.jsonl` |
| Upright/peg DAgger | `outputs/iclr_dagger_three_task/final/{upright,peg}/dagger_protocol-1.jsonl` |
| Canonical selection and exclusions | `outputs/iclr_evidence_closeout_20260921/` |
| Original DAgger implementation diagnostics | `outputs/iclr_dagger_three_task/analysis/` |

### Demonstrations

| Task | Admitted / attempted episodes | Control steps | Training file | HF tensor file |
|---|---:|---:|---|---|
| Parcel | 297 / 300 | 196,822 | `outputs/paper/demos/expert_episodes.pt` | `demonstrations/expert_episodes.pt` |
| Upright | 315 / 330 | 297,985 | `outputs/upright/demos/expert_episodes.pt` | `demonstrations/upright_expert_episodes.pt` |
| Peg | 325 / 330 | 475,758 | `outputs/peg/demos/expert_episodes.pt` | `demonstrations/peg_expert_episodes.pt` |

The three full datasets total 937 admitted episodes and 970,565 control steps.
Training tensors contain `episodes` as `(observations, actions, success)` tuples,
with shapes `(T,147)` and `(T,16)`, plus `records`, `all_records`, `rate_spec`,
`jitter`, `seed`, `obs_dim`, `act_dim`, and `config`. Demonstration collection uses
observation corruption; evaluation disables it. `episodes` contains successful
attempts only; `all_records` preserves rejected attempts. Do not filter again.

Full-dataset SHA-256:

```text
parcel  5c443c62d21471897edad0132dbc46bc24f19ea63d77fd2b8f9c6e53bd605fa2
upright 2959a476b294bbd63fe76a4a7ab4e64e1a87dfadd7e146b132559b953acfbb07
peg     4512ae3a7b0b651faa380351ff727cdabe85ec1b132fdec0ac5f3ec0c0fff17d
```

HF also provides `data/{parcel_insertion,upright_placement,peg_insertion}_demonstrations.parquet`.
These expose the same demonstrations for inspection. Use the checksum-identified
`.pt` inputs with the supplied trainers; do not silently replace them with a
reconstructed Parquet conversion.

### ACT checkpoints for the reported comparisons

| Task | Seed / epoch | Local path | SHA-256 |
|---|---|---|---|
| Parcel ACT-A | 0 / 2000 | `outputs/paper/act/act_stow.pt` | `0d9400462d8ff4d3d99614d48a1d6c365de3e3df8faa9bf88f91677130a0d4af` |
| Upright | 0 / 2000 | `outputs/act_comparability/upright/seed-0/training/checkpoints/epoch-2000.pt` | `94888be60215900c6aff75d73ae8d4fd15535691ff16a7fcb545d34556e22fa7` |
| Peg | 0 / 1400 | `outputs/act_comparability/peg/seed-0/training/checkpoints/epoch-1400.pt` | `c9c1e8cd08dae4d5501dbbeceae933e69e2037633f31007e4257914d346a911d` |

Upright seed-1/2 checkpoints are epochs 1600/1300; peg seed-1/2 checkpoints are
2000/1600. The reported comparisons use seed 0; seeds 1 and 2 are replications.
Parcel `act_seed2.pt` and `act_seed3.pt` correspond to model seeds 1 and 2
(ACT-B/C), respectively.
DP uses seed 42 and epoch 300; upright/peg DAgger uses seed 1 and round four.
All hashes and download names are in `ARTIFACT_INDEX.csv`.

## 3. Reproduce saved results on CPU

From the code repository root, no Isaac Lab, GPU, checkpoints or demonstrations
are needed for the count audit:

```bash
python3 scripts/reproduce_manuscript.py \
  --output-dir outputs/reproduce/manuscript-counts
```

Expected validation: **136 conditions, 20 source series, 23,200 episodes;
104 verified paired learner conditions and nine standalone conditions**.
The command refuses an existing output directory. It verifies packaged/source
hashes, episode identities and counts, success/stage/failure counts, direct peg
monitor implications, and pairing metadata and discordant outcome counts.
It writes `success_counts.csv`, `paired_results.json`, and `validation.json`.

For the pointwise 20,000-resample paired bootstrap intervals, install NumPy and
use a different output directory:

```bash
python3 -m venv .venv-analysis
.venv-analysis/bin/python -m pip install numpy==1.26.4
.venv-analysis/bin/python scripts/reproduce_manuscript.py --bootstrap \
  --output-dir outputs/reproduce/manuscript-bootstrap
```

The sign is **expert minus learner**. Bootstrap seed is 0, with linear 2.5th/97.5th percentiles.
Standalone conditions receive separate Wilson binomial intervals and no paired
difference. Pairing is verified only within a factor. Parcel records lack
initial joint-state serialization and a bank digest; the pairing scope is the
recorded initial object poses and shared reset metadata. Do not pool seeds,
tasks, repeated expert series, or overlapping episodes.

The canonical rules prefer protected final over pilot records, corrected over
superseded records, and the declared checkpoint/bank. Larger sample sizes are
eligible only when protocol, checkpoint, bank and configuration are otherwise
identical. Selection never uses success rate. Floor-saturated nominal DAgger
supports failure characterization, not execution-speed degradation. The existing
90/100 nominal admission gate is unchanged; it is not a statistical equivalence
test; applicability to each evaluation is recorded in the evidence catalog.

Additional parcel diagnostics:

```bash
python3 scripts/reproduce.py envelope
python3 scripts/reproduce.py stages
python3 scripts/reproduce.py certificate
python3 scripts/reproduce.py certificate-oos
python3 scripts/reproduce.py expert-ceiling
```

They read `data/records/` and the parcel summaries. Some materialize
copies under `outputs/paper/` and write figures under `media/`; use a separate
checkout when generating these diagnostic figures. They are not the canonical three-task
selection command. Pin SciPy 1.15.3 when rescoring force-closure margins:
different Qhull versions can change signs at the numerical boundary.

## 4. Obtain checkpoints and tensors

Download the checkpoints and tensor datasets for all three tasks:

```bash
python3 scripts/download_artifacts.py --manuscript
python3 scripts/download_artifacts.py --manuscript --verify
```

To copy from a local Hugging Face checkout without network access:

```bash
python3 scripts/download_artifacts.py --manuscript \
  --local-hf ../parcelstow-hf
```

Existing matching files are skipped. A missing or corrupt file makes `--verify`
fail. Downloads are verified before installation, and mismatched existing
artifacts are never replaced. A tiny Git LFS pointer is not a usable checkpoint;
the local source must contain the actual tensor bytes. The manuscript bundle
contains 22 artifacts. Use `--revision HF_SHA` to pin a Hugging Face commit;
SHA-256 verification applies to every download.

For individual downloads, use artifact names from the index:

```bash
python3 scripts/download_artifacts.py --names \
  manuscript_upright_act_seed0 manuscript_peg_act_seed0 \
  manuscript_upright_dp_seed42 manuscript_peg_dp_seed42 \
  manuscript_upright_dagger_seed1 manuscript_peg_dagger_seed1
```

## 5. Environments and fresh output directories

Two recorded environments were used:

| Use | Python | PyTorch | NumPy | Other |
|---|---|---|---|---|
| Upright/peg ACT training (all seeds) | 3.12.3 | 2.8.0+cu128 | 2.3.5 | CUDA 12.8, RTX 5070 Ti |
| Simulator evaluation; parcel ACT, DP and DAgger drivers | 3.11.14 | 2.7.0+cu128 | 1.26.4 | Isaac Sim 5.1.0, Isaac Lab 0.54.2, SciPy 1.15.3, diffusers 0.30.3 |

The six ACT `run_config.json` files record exact training commands, versions,
partitions and validation starts. Do not assume the simulator Python was used
for those offline trainers. Execution revisions are recorded with the evaluation data. Record the source revision and local changes for every
new reproduction.

Set these paths to your Isaac Lab and offline ACT environments:

```bash
export ISAACLAB_VENV=/path/to/IsaacLab/.venv
SIM_PY="$ISAACLAB_VENV/bin/python"
ACT_PY=/path/to/act-training-env/bin/python
export OMNI_KIT_ACCEPT_EULA=YES WANDB_MODE=disabled
unset PYTHONPATH
"$SIM_PY" -m pip install -e 'source/parcelstow[all]'
"$SIM_PY" -m pip install diffusers==0.30.3 scipy==1.15.3
```

For a new offline ACT environment, use Python 3.12.3 with PyTorch 2.8.0+cu128
and NumPy 2.3.5. Point `ACT_PY` to that environment; the trainer does not import
Isaac Lab. Verify dependencies before a long run:

```bash
"$ACT_PY" -c 'import sys,torch,numpy; print(sys.version,torch.__version__,numpy.__version__,torch.version.cuda)'
"$SIM_PY" -c 'import sys,torch,numpy,scipy,diffusers; print(sys.version,torch.__version__,numpy.__version__,scipy.__version__,diffusers.__version__)'
```

Every simulator or training rerun must use a fresh root:

```bash
RUN="outputs/reproduction_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "$RUN"
mkdir "$RUN/logs"
git rev-parse HEAD > "$RUN/code_head.txt"
git diff --binary > "$RUN/working_tree.patch"
"$SIM_PY" -m pip freeze > "$RUN/simulator_packages.txt"
"$ACT_PY" -m pip freeze > "$RUN/act_training_packages.txt"
```

Keep untracked source dependencies with the run as well; `git diff` does not
capture them. Never target existing `outputs/paper`, `outputs/act_comparability`,
`outputs/iclr_*`, or the packaged records with a new training/evaluation command.
`scripts/isaac_run.sh` sets four CPU threads, `nice=10`, and CPU cores `0-7`;
`ISAAC_CORES` and `ISAAC_NICE` can override resource allocation. Logs belong
outside a trainer's initially empty output directory. The training/evaluation
recipes use `/usr/bin/time -p`; its `real`, `user`, and `sys` times are captured
in each run log. ACT also records elapsed seconds in `metrics.jsonl`; the other
trainers record training durations in `results.jsonl`.

## 6. Evaluate the frozen checkpoints

These commands evaluate the frozen checkpoints in new simulator runs. Upright
and peg use the declared final-bank configuration. The current parcel driver
uses an indexed bank whose draws differ from the saved parcel records, even
with the same base seed. Use Section 3
to reproduce the saved counts. Do not select checkpoints from these new outcomes.

### Parcel: seven factors, 100 episodes each

```bash
scripts/isaac_run.sh "$RUN/logs/parcel_eval.log" \
  /usr/bin/time -p "$SIM_PY" scripts/evaluate.py --task parcel --actor expert act dp dagger \
  --rates 0.5 1 1.5 2 2.25 2.5 3 \
  --episodes 100 --num_envs 32 --eval_seed 12345 --jitter 0.01 \
  --act_ckpt outputs/paper/act/act_stow.pt \
  --dp_ckpt outputs/paper/dp/dp_stow.pt \
  --dagger_ckpt outputs/paper/dagger/student_final.pt \
  --out_dir "$RUN/evaluation/parcel" --tag _replica
```

The parcel driver uses indexed initial conditions and has no `--bank_role`
flag. Its output can append to existing paths, so use a fresh `RUN` directory.
Pair expert and learner episodes within the new run, using recorded initial conditions.
The saved parcel DP conditions and DAgger at `r=2,2.5` fail full initial-pose
pairing and retain their standalone statistical treatment.

### Upright and peg: eight factors, 200 episodes each

```bash
for task in upright peg; do
  if [ "$task" = upright ]; then epoch=2000; else epoch=1400; fi
  scripts/isaac_run.sh "$RUN/logs/${task}_eval.log" \
    /usr/bin/time -p "$SIM_PY" scripts/evaluate.py --task "$task" --actor expert act dp dagger \
    --rates 0.5 0.75 1 1.25 1.5 1.75 2 2.5 \
    --episodes 200 --num_envs 32 --eval_seed 73001 --jitter 0.01 \
    --bank_role final --tag _replica \
    --act_ckpt "outputs/act_comparability/$task/seed-0/training/checkpoints/epoch-$epoch.pt" \
    --dp_ckpt "outputs/iclr_dp_three_task_20260919/raw/checkpoints/dp_$task.pt" \
    --dagger_ckpt "outputs/iclr_dagger_three_task/$task/student_final.pt" \
    --out_dir "$RUN/evaluation/$task"
done
```

Here `final` selects the existing final-bank configuration and enforces its
full ordered grid, 200 episodes, base seed 73001, jitter 0.01, zero action noise,
no observation corruption, and no rate-feature intervention. The driver refuses
existing target records. It writes `ACTOR_replica.jsonl` and
`summary_replica.jsonl`; inspect `checkpoint_sha256`, `bank_role`,
`initial_condition_id`, `initial_condition_bank_sha256`, per-factor seed and
configuration in the episode records. Do not sum repeated expert evaluations.

For ACT seeds 1 and 2, use the exact checkpoint in `ARTIFACT_INDEX.csv`, a distinct
output directory and the same factor grid/bank. Keep each training seed separate.

## 7. Reproduce training procedures

These recipes create **new replicas**. Reuse the frozen demonstrations for a
training reproduction. Collecting new demonstrations changes the input dataset.
Save new checkpoints and hashes separately; never replace a canonical checkpoint
with one that scores better. Reproduction commands do not promise bitwise-identical
training or simulation across CUDA, physics or numerical-library versions.

### Parcel ACT

The parcel trainer uses all 297 episodes and a padded-loss reduction.
The upright/peg trainer uses a different loss reduction; use the task-specific recipe.

```bash
scripts/isaac_run.sh "$RUN/logs/parcel_act_train.log" \
  /usr/bin/time -p "$SIM_PY" scripts/manipulation/run_stow_act.py \
  --task ParcelStow-L6-Distill-Play-v0 \
  --demos outputs/paper/demos/expert_episodes.pt \
  --out_dir "$RUN/training/parcel/act" --tag stow \
  --epochs 2000 --batch 8 --lr 0.00001 --chunk_size 100 \
  --kl_weight 10 --hidden_dim 512 --dim_feedforward 3200 --temporal_agg 1 \
  --model_seed 0 --num_envs 32 --jitter 0.01 \
  --diag_episodes 50 --diag_rate 1 --eval_seed 12345
```

Output is `act_stow.pt`, `results.jsonl` and diagnostic records under the new
training directory. The diagnostic follows training and does not select an epoch.
For ACT-B/C, set `--model_seed 1/2`, use distinct output directories and tags
`seed2/seed3`. For the 50/100-demonstration ablations, download `demos_n50` and
`demos_n100` or reproduce their rate-stratified subsets:

```bash
"$ACT_PY" scripts/manipulation/subsample_stow_demos.py \
  --demos outputs/paper/demos/expert_episodes.pt --sizes 50 100 \
  --out_dir "$RUN/subsets"
```

Train each subset with the parcel ACT recipe, model seed 0, a distinct directory,
and the appropriate `--demos` and `--tag n50` / `n100`. Compare the subset checksum
against `ARTIFACT_INDEX.csv` before calling it the frozen input.

### Upright/peg ACT: training, candidate evaluation, then selection

Train complete episodes with an 80/20 split, valid-element L1, training-only
normalization, split seed 20260904 and validation-start seed 20260905. The primary
model seed is 0; seeds 1 and 2 are separate replications. The retained candidates
are the five lowest deterministic validation L1 checkpoints plus epoch 2000.

```bash
for task in upright peg; do
  for seed in 0 1 2; do
    /usr/bin/time -p "$ACT_PY" scripts/train_act.py --task "$task" --model_seed "$seed" \
      --demos "outputs/$task/demos/expert_episodes.pt" \
      --out_dir "$RUN/training/$task/seed-$seed" \
      --epochs 2000 --batch 8 --lr 0.00001 --weight_decay 0.0001 \
      --chunk_size 100 --kl_weight 10 --hidden_dim 512 --dim_feedforward 3200 \
      --temporal_agg 1 --split_seed 20260904 --validation_start_seed 20260905 \
      --validation_fraction 0.2 --validation_samples_per_episode 4 \
      --checkpoint_every 100 --keep_top 5 \
      > "$RUN/logs/${task}_act_seed${seed}_train.log" 2>&1
  done
done
```

For a new trained seed, evaluate **every retained candidate** on the nominal
100-episode development bank. The following example is upright seed 0; repeat
with explicit task and seed values, never with a final-bank score:

```bash
task=upright
seed=0
training="$RUN/training/$task/seed-$seed"
development="$RUN/development/$task/seed-$seed"
scripts/isaac_run.sh "$RUN/logs/${task}_seed${seed}_dev_expert.log" \
  /usr/bin/time -p "$SIM_PY" scripts/evaluate.py --task "$task" --actor expert \
  --rates 1 --episodes 100 --num_envs 32 --eval_seed 42001 --jitter 0.01 \
  --bank_role development --tag _expert --out_dir "$development"
for checkpoint in "$training"/checkpoints/epoch-*.pt; do
  epoch=${checkpoint##*/epoch-}
  epoch=${epoch%.pt}
  epoch=$((10#$epoch))
  scripts/isaac_run.sh "$RUN/logs/${task}_seed${seed}_dev_${epoch}.log" \
    /usr/bin/time -p "$SIM_PY" scripts/evaluate.py --task "$task" --actor act \
    --rates 1 --episodes 100 --num_envs 32 --eval_seed 42001 --jitter 0.01 \
    --bank_role development --tag "_epoch-$epoch" --act_ckpt "$checkpoint" \
    --out_dir "$development"
done
"$ACT_PY" scripts/select_act_checkpoint.py --task "$task" \
  --training_dir "$training" --evaluation_dir "$development" \
  --expert_record "$development/expert_expert.jsonl" \
  --output "$RUN/${task}_seed${seed}_selection.json"
```

Selection order: greatest success count, greatest lexicographic task-stage vector,
lowest deterministic validation L1, earliest epoch. Freeze the selected hash before
final evaluation. The selector requires execution-time checkpoint hash fields in development
records. The bundled selection manifests identify the checkpoints for the
reported results; use those manifests directly when evaluating the saved models.

### Diffusion Policy

All tasks use a state-based conditional 1D U-Net (256/512/1024), horizon 16,
two observation steps, eight executed actions, 100 DDPM steps, AdamW, EMA power
0.75, model seed 42 and the fixed final epoch 300. No nominal-success selection.
The generic driver selects the task-specific expert and monitor from `--task`.

```bash
for task in parcel upright peg; do
  case "$task" in
    parcel) gym=ParcelStow-L6-Distill-Play-v0; demos=outputs/paper/demos/expert_episodes.pt; tag=stow; diag_n=50; diag_seed=12345 ;;
    upright) gym=UprightPlace-L6-Play-v0; demos=outputs/upright/demos/expert_episodes.pt; tag=upright; diag_n=100; diag_seed=42001 ;;
    peg) gym=PegInsert-L6-Play-v0; demos=outputs/peg/demos/expert_episodes.pt; tag=peg; diag_n=100; diag_seed=42001 ;;
  esac
  scripts/isaac_run.sh "$RUN/logs/${task}_dp_train.log" \
    /usr/bin/time -p "$SIM_PY" scripts/manipulation/run_stow_diffusion_policy.py \
    --task "$gym" --demos "$demos" --out_dir "$RUN/training/$task/dp" --tag "$tag" \
    --epochs 300 --batch 256 --lr 0.0001 --horizon 16 --n_obs_steps 2 \
    --n_action_steps 8 --num_inference_steps 100 --model_seed 42 \
    --num_envs 32 --jitter 0.01 --diag_episodes "$diag_n" --diag_rate 1 --eval_seed "$diag_seed"
done
```

Outputs are `dp_stow.pt`, `dp_upright.pt`, or `dp_peg.pt` under the specified new
directory, with training/diagnostic logs. Evaluation loads the saved EMA weights.

### DAgger

The upright/peg configuration is a 512/256/128 ELU MLP, 40-epoch initial fit,
four rounds of 100 complete learner episodes, and a reinitialized 40-epoch fit
on all accumulated expert labels after each round. Exploration uses action
noise standard deviation 0.1 in half the environments; nominal diagnostics are
noise-free and do not select a checkpoint. Labels correspond to the state before
the learner action. The final `student_final.pt` is tensor-equivalent to round
four even if serialization bytes differ.

```bash
for task in upright peg; do
  if [ "$task" = upright ]; then
    gym=UprightPlace-L6-Play-v0; lo=0.75; hi=1.75
  else
    gym=PegInsert-L6-Play-v0; lo=0.5; hi=1.0
  fi
  scripts/isaac_run.sh "$RUN/logs/${task}_dagger_train.log" \
    /usr/bin/time -p "$SIM_PY" scripts/manipulation/run_stow_distill.py \
    --task "$gym" --demos "outputs/$task/demos/expert_episodes.pt" \
    --out_dir "$RUN/training/$task/dagger" --rate_lo "$lo" --rate_hi "$hi" \
    --dagger_rounds 4 --dagger_episodes 100 --diag_episodes 100 --diag_rate 1 \
    --num_envs 32 --jitter 0.01 --action_noise 0.1 --epochs 40 --batch 4096 \
    --lr 0.001 --train_seed 1 --eval_seed 42001
done
```

For the parcel four-round recipe, use the same options with
`--task ParcelStow-L6-Distill-Play-v0`, parcel demonstrations, `--rate_lo 0.5
--rate_hi 2`, `--diag_episodes 50 --eval_seed 12345`, and a separate parcel
output directory. The parcel training configuration uses seed 1. Its saved state dictionary and
evaluation records do not bind all training metadata at execution time, so the
evidence catalog marks the model seed unknown. Use `student_final.pt`, as
identified by the artifact index.

The upright/peg implementation audit passed observation/action layout, phase
and speedup-factor inputs, expert-label timing, reset behavior, checkpoint loading,
demonstration-state errors and closed-loop divergence checks. Low success alone
was not taken as evidence of a valid implementation. The audit and monitor-rule
resolution are packaged under `provenance/`; parcel does not inherit that audit.

## 8. Optional demonstration recollection

Recollection is a new dataset; exact admitted counts and tensor hashes can differ.
Use the distributed inputs for the canonical training recipe. These commands
document the original attempt counts, ranges, seed and jitter:

```bash
"$SIM_PY" scripts/manipulation/run_stow_expert.py --mode demos \
  --episodes 300 --num_envs 32 --seed 1 --rate_lo 0.5 --rate_hi 2 --jitter 0.01 \
  --out "$RUN/collection/parcel.jsonl" --demo_out "$RUN/collection/parcel.pt"
"$SIM_PY" scripts/manipulation/run_upright_expert.py --mode demos \
  --episodes 330 --num_envs 32 --seed 1 --rate_lo 0.75 --rate_hi 1.75 --jitter 0.01 \
  --out "$RUN/collection/upright.jsonl" --demo_out "$RUN/collection/upright.pt"
"$SIM_PY" scripts/manipulation/run_peg_expert.py --mode demos \
  --episodes 330 --num_envs 32 --seed 1 --rate_lo 0.5 --rate_hi 1 --jitter 0.01 \
  --out "$RUN/collection/peg.jsonl" --demo_out "$RUN/collection/peg.pt"
```

The `demos` mode enables observation corruption internally. The expert uses its
own controller inputs; the recorded 147-vector is the learner observation.
Store all attempt records, not just successful demonstrations.

## 9. Validate a new reproduction before comparison

Retain the command, source revision and patch, versions, input/output SHA-256,
full episode records, summaries, checkpoint selection and elapsed compute time.
Verify the 147-input/16-action contract: joint positions/velocities, previous
action, object pose, fingertips/forces, phase, and speedup factor; action is
`q_target = q_default + 0.5 * action` at 50 Hz. Reset policy buffers and expert
correction state at episode boundaries.

For each factor, check unique episode IDs, the requested count, checkpoint and
bank identities, initial poses, seeds, noise settings, physical configuration,
and agreement between episode and summary counts. Fail on a mismatch before
computing differences. The frozen-input reproducer validates only the packaged
canonical files; it is not a generic admission tool for a new evaluation bank.

Peg physical events are recorded independently except for these implications:

```text
lifted_clear -> acquired
reoriented_upright -> acquired
aligned -> acquired
released -> inserted
settled -> released
task_success -> inserted and released and settled
```

Do not require `inserted -> aligned`, `aligned -> reoriented_upright`, or
`inserted -> acquired`. A peg can enter the pocket without having met the
stricter held-alignment condition. For upright, `tipped_after_release` is a
terminal category that also includes leaving the target; inspect `failure_detail`
before describing it as physical tipping.

Use paired differences only when complete episode-level pairing is verified.
Use explicitly unpaired differences with separate binomial intervals only when
distribution matching is established without episode pairing. Otherwise report
standalone rates. Do not interpret floor-saturated nominal policies as measuring
execution-speed degradation. Illustrative videos and their separate reset seeds
never add episodes to the canonical success-rate estimates.
