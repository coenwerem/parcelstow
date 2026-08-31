# Frozen Derived Analyses of the Paper

Summary files behind the paper's tables, produced from the released
episode records (data/records/) by the analysis scripts under
scripts/manipulation/. Per-episode records, videos, checkpoints, and
demonstrations are packaged separately, see docs/DATA_AND_CHECKPOINTS.md.

- eval_summary.jsonl, one row per actor and rate of the final evaluation
  (100 episodes each, stage fractions with Wilson intervals, failure
  reasons, slip and utilization distributions).
- certificate_analysis.json and .txt, Spearman, AUROC, and quartile
  strata of the realized margins against outcome
  (analyze_stow_certificate.py).
- certificate_oos_analysis.json and .txt, the out-of-sample certificate
  validation across actors (analyze_certificate_oos.py).
- expert_ceiling_analysis.json and .txt, the tracking-accuracy
  attribution of the expert rate limit (analyze_expert_ceiling.py).
- handoff_summary.jsonl, the common-controller handoff (50 episodes per
  actor and rate).
- relative_handoff_summary.jsonl and .csv, the relative-motion handoff
  (stow_relative_handoff.py, 100 episodes per actor and rate, expert and
  the three ACT seeds), retention to pre-insertion, d_p and d_R, endpoint
  reasons, kinematic residuals, and the secondary insertion outcome
  (summarize_relative_handoff.py).
- act_multiseed_summary.json and .csv, the ACT training-seed replication
  (act_seed1 model_seed 0, act_seed2 model_seed 1, act_seed3 model_seed 2)
  at r 1.0, 1.5, 2.0 on the draws of the final evaluation, per seed and
  rate with the training logs, the r 1 diagnostics, and the across-seed
  mean, min, max, and individual values (summarize_act_multiseed.py).
- act_demoscale_*, the demonstration-scaling runs, ACT retrained on
  rate-stratified 50 and 100 demonstration subsets and evaluated on the
  frozen draws (subsample_stow_demos.py, summaries per n).
- rate_conditioned_margin_analysis.json and .txt, the rate-conditioned
  analysis of the nominal realized-contact margin at lift with
  epsilon^(beta) as the secondary domain-alignment result
  (analyze_rate_conditioned_margin.py).
- expert_sweep_summary.jsonl, the expert calibration behind the frozen
  rate grid.
- act_results.jsonl, act_seed2_results.jsonl, act_seed3_results.jsonl,
  dp_results.jsonl, dagger_results.jsonl, dagger_results_ext.jsonl,
  demos_summary.jsonl, the training driver logs.
