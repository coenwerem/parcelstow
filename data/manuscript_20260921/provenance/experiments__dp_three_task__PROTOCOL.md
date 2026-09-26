# Diffusion Policy Extension Protocol

## Scope

Train one state-based Diffusion Policy checkpoint each for upright placement and
keyed peg insertion. Parcel insertion already has a released Diffusion Policy
baseline. This extension adds a second learner architecture to both additional
tasks; it does not change any task, expert, demonstration, monitor, or ACT result.

The selected checkpoint for each new task is the final epoch of one predeclared
training run. There is no checkpoint or seed search. A nominal development
evaluation checks that the checkpoint can be loaded and executed; its result
does not authorize retraining or model selection. Report both tasks, including
poor or failed policies.

## Inputs and Training

Use the existing successful expert episodes at
`outputs/upright/demos/expert_episodes.pt` and
`outputs/peg/demos/expert_episodes.pt`, without recollection or filtering.
Record each file's SHA-256 digest before training. Each task uses the released
parcel Diffusion Policy configuration: state observations, 16-dimensional
actions, a `ConditionalUnet1D` with dimensions 256/512/1024, horizon 16, two
observation steps, eight executed actions, 100 DDPM training and inference
steps, batch 256, AdamW learning rate `1e-4`, EMA power 0.75, and 300 epochs.
The one predeclared model seed is 42. Normalization uses the task's admitted
demonstrations, as in the released parcel baseline. No validation partition or
nominal-success threshold is used for checkpoint selection.

Run one task per process on the Linux Isaac Lab environment. Training output
directories must be empty. The checkpoint is `dp_upright.pt` or `dp_peg.pt`.
Record the code revision, dependency versions, demonstration digest, training
command, checkpoint digest, and run log. A failed technical run may be debugged
before final evaluation, but all attempts and changes must be retained.

## Evaluation

The training command runs a 100-episode diagnostic at `r=1` on development seed
42001. This is not final evidence. Once both checkpoint digests are fixed,
evaluate expert, the preselected ACT seed 0, and Diffusion Policy on each
task's full registered rate grid, 200 episodes per actor and rate, with 10 mm
planar jitter, no observation corruption, no action noise, and the indexed
initial-condition bank generated from base seed 73001. Do not use final
outcomes to change a checkpoint, seed, policy input, task, or rate grid.

The new Diffusion Policy results are a post hoc architectural extension to the
already inspected ACT final study. Do not present this as an untouched
pre-registered confirmatory test, or treat its pointwise intervals as
simultaneous inference. Expert and learners are paired within each rate;
different rates use different banks and are not paired across rates.

The preselected ACT comparator is seed 0 in each task's frozen
`selection-20260911.json` manifest. For upright placement it is epoch 2000
(`94888be60215900c6aff75d73ae8d4fd15535691ff16a7fcb545d34556e22fa7`);
for keyed peg insertion it is epoch 1400
(`c9c1e8cd08dae4d5501dbbeceae933e69e2037633f31007e4257914d346a911d`).
These checkpoint digests must match the files loaded for final evaluation.

Report per-rate success counts, demonstrated-rate boundaries, signed
learner-minus-expert differences with 20,000-resample paired bootstrap 95%
intervals, and stage/failure counts. Do not pool repeated expert rows across
actors as independent episodes. No noninferiority or nominal-parity threshold
is part of this protocol.
