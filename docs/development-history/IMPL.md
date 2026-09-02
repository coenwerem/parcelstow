# Historical Standalone Release Working Notes

> **Historical record.** This previously ignored file preserves the v1.0.0
> extraction, verification, and release decisions. Its blocker tables and
> status statements are obsolete. Use the repository README and current files
> under `docs/` for adopter instructions.

Local implementation log for the public-artifact effort. This file stays
out of Git (ignored via .gitignore). Last update 2026-08-25 (evening,
blocker-closing pass).

## Blocker Status Table (Current)

| # | blocker | status |
|---|---|---|
| 1 | firmgrasp private dependency | RESOLVED, Ferrari-Canny vendored in-repo, epsilon_beta optional behind `PARCELSTOW_FIRMGRASP` |
| 2 | FRoGGeR grasp-bank provenance | DEMOTED TO DOCUMENTED PROVENANCE, frozen construction input, docs/ASSET_PROVENANCE.md, private repo coenwerem/frogger commit 4705a49 recorded, no upstream base commit determinable |
| 3 | G1-L6 asset regeneration provenance | RESOLVED, merge_g1_l6_urdf.py recovered into scripts/assets/ (parameterized), regenerates the shipped merged URDF byte-identically from upstream Unitree HEAD plus the shipped L6 URDFs |
| 4 | robot mesh redistribution licensing | RESOLVED, all 167 G1 meshes byte-identical to unitreerobotics/unitree_ros commit 4ddbf6d (BSD-3-Clause), all L6 URDFs and meshes byte-identical to linker-bot/linkerhand-urdf commit 075cc7d (Apache-2.0), license texts under assets/LICENSES/, mapping in assets/NOTICE.md, the hand is LinkerHand L6 v3.1 (the RealHand name in older notes is a misnomer) |
| 5 | top-level license | RESOLVED, Apache-2.0 LICENSE plus NOTICE separating third-party components, compatibility audit MIT + BSD-3 + Apache-2.0, no conflict |
| 6 | external artifact hosting | STAGED PRIVATE, all 11 artifacts uploaded to the PRIVATE HF dataset cenwerem/parcelstow with MANIFEST.json, manifest carries hf_path plus resolve url per artifact, download plus sha256 plus model-construction validated from a fresh clone, PUBLIC visibility awaits user approval |
| 7 | learned-baseline dependency declaration | RESOLVED, setup.py extras (analysis, diffusion, hub, all), scipy in install_requires, tested-version table in docs/REPRODUCING_THE_PAPER.md incl. the SciPy 1.15.3 / Qhull boundary-tie note |
| 8 | pyright sibling path | RESOLVED, marked developer-local in pyproject.toml |

## Final Paper-Artifact Consistency Decisions (2026-08-25, Closing Pass)

- Public product name is RealHand L6 (manufacturer rebrand,
  realhand.com), historical description provenance stays LinkerHand /
  linkerhand in upstream repo names, filenames, paths, package ids,
  licenses, and frozen metadata. Clarification blurbs sit in
  docs/ASSET_PROVENANCE.md and assets/NOTICE.md, README names the hand
  once. No technical path or frozen record renamed.
- Paired bootstrap protocol is 20000 resamples
  (plot_envelope.paired_gap_ci default). Rerun reproduces gap 0.31 and
  95% interval [0.18, 0.44] unchanged at the reported rounding.
- Expert-only calibration has ELEVEN candidate rates through r=6
  (expert_sweep_summary.jsonl, 64 episodes each, counts 63/63/63/64/64/
  58/19/1/0/0/3). The learned-policy grid stays the seven rates ending
  at r=3. No learner evaluation exists or will exist at r=6.
- Reproducibility contract is numbers-first, released records plus
  analysis code reproduce the reported numerical results, principal
  evaluation plots regenerate too, camera-ready figure generation is NOT
  part of the release contract. REPRODUCING_THE_PAPER.md carries the
  quantity-to-record-to-command audit table, reproduce.py gained the
  certificate-oos target and materializes the replication records.
- FIRMGrasp unchanged, private, optional epsilon_beta hook only, the
  appendix negative control rests on the frozen analysis.
- No-force-closure population accounting (2026-08-25). The manuscript's
  414 is the six-actor population (expert, ACT-A, DAgger, DP, ACT-B,
  ACT-C) counting acquired episodes with epsilon_lift <= 0 over every
  evaluated rate, per-file 0 + 42 + 57 + 150 + 63 + 102 = 414, zero
  successes, exactly reproducible from the shipped records. The
  manuscript's 79 is the two demonstration-scaling checkpoints, n50 15
  plus n100 64, zero successes, exactly reproducible. The ACT-A
  replication rerun contributes 8 further such episodes (zero
  successes) and sits outside both populations by design, the rerun
  re-evaluates the already-counted ACT-A checkpoint as a draw-identity
  control, not a seventh actor. Two of the 414 (dagger r0.5 ep44, dp
  r2.5 ep95) record epsilon_lift exactly -0.0, zero margin, no force
  closure, included under the <=0 definition, excluded by a
  sentinel-only ==-1 count. The 499 in the 2026-08-25 closeout chat was
  a sentinel-only aggregate over all nine record files (501 under the
  manuscript's <=0 definition, 414 + 79 + 8), it appears in no public
  document, and zero successes holds under every definition and
  population.

Release gate, every item is closed except the deliberately user-gated
ones, HF repo visibility flip, GitHub remote/public, v0.1.0 tag, arXiv
id. Frozen-asset integrity note, the extraction's rename pass had
rewritten provenance path strings inside gdf_bank_parcel.json,
parcel_stow_geometry.json, and parcel_stow_trajectory.json, restored to
their original bytes on 2026-08-25 (commit ba5f947) and verified by
sha256 against the source repo.

## Standalone Criterion

Every dependency must be third-party, open-source external work, or a
public project of the author. Nothing may depend on the private
g1_locomanip_lab repo, private local checkouts, or files existing only on
the primary machine.

## Current Blockers, Enumerated

1. **firmgrasp (private local checkout). RESOLVED 2026-08-25.**
   The plain Ferrari-Canny epsilon is now computed in-repo by
   `mdp/ferrari_canny.py` (grasp wrench space from linearized cone edges,
   inscribed ball via scipy ConvexHull, -1 no-force-closure sentinel),
   ported from the firmgrasp contacts path with the author's approval.
   Validation, the vendored scorer recomputed all 6322 recorded contact
   sets of the frozen records, ZERO mismatches under the Isaac venv scipy
   (the producing environment), 15 boundary sign ties at |epsilon| under
   1e-12 under system scipy (qhull version float ties, not divergence).
   `epsilon_beta` stays optional behind `PARCELSTOW_FIRMGRASP`, and the
   `~/ResearchProjects/firmgrasp` expanduser fallback is GONE, local runs
   wanting epsilon_beta must set the env var now. firmgrasp itself stays
   unpublished by the author's decision (focus moved to learning-facing
   work). New simulations on a fresh clone now produce the certificate
   with no private dependency. Scoring needs scipy, which the Isaac Lab
   environment ships.

2. **frogger local checkout behind the grasp bank.**
   The frozen bank `assets/gdf_bank_parcel.json` was synthesized by
   "frogger scripts/g1_l6_runner.py (local checkout)" (stamp inside
   `build_parcel_bank.py`), a local fork/driver over the public FRoGGeR.
   The bank and the frogger record ship frozen, so nothing at runtime
   needs frogger, but regenerating the bank does. Resolutions, publish
   the g1_l6 runner (or fold it into the public repo) and cite upstream
   FRoGGeR, or declare the bank a frozen input with provenance only.

3. **Robot-asset regeneration tooling left in the private repo.**
   `assets/g1_l6/g1_29dof_l6_both.urdf` and the USDs were produced by
   `merge_g1_l6_urdf.py` (old repo, scripts/tools, excluded) plus the
   Isaac URDF importer. The merged asset ships frozen, its regeneration
   path does not. Resolution, copy `merge_g1_l6_urdf.py` into the public
   repo (author's own code) and document the importer step, or declare
   the asset frozen. `assets/g1_l6/usd_both/config.yaml` still carries
   the old machine's absolute paths as importer provenance.

4. **Mesh licensing unverified.**
   The G1 meshes originate from Unitree's description files and the L6
   meshes from LinkerHand. Redistribution licenses for both mesh sets
   are unverified, and the repo states none. Publishing without a
   verified license line for the meshes risks a takedown. Resolution,
   verify both upstream licenses, add attribution and license text under
   assets/, only then publish.

5. **No top-level LICENSE.**
   The repository itself declares no license (`setup.py` says Apache-2.0
   from the template). Without a LICENSE file nobody can legally reuse
   the code. Decision pending with the author.

6. **External artifacts exist only on this machine.**
   The 2.2 GB of checkpoints, demonstrations, and videos are inventoried
   with sha256 in `artifacts/manifest.json`, every `url` is null. Until
   hosted (GitHub Releases, Hugging Face, or Zenodo all fit the file
   sizes), Tier 2/3 depend on the primary machine. The videos bundle is
   already built at `outputs/release/parcelstow_videos.tar.gz`.

7. **Undeclared Python dependencies for the learned baselines.**
   The extension's `setup.py` declares only psutil. Evaluating DP needs
   `diffusers`, the analysis tier needs numpy and matplotlib, everything
   simulator-side assumes the Isaac Lab environment (torch, gymnasium).
   Resolution, add a requirements listing (or extras) and name the
   tested Isaac Lab / Isaac Sim versions in the README.

8. **Dev-tooling paths assume a sibling IsaacLab checkout.**
   Root `pyproject.toml` pyright config points at `../IsaacLab/.venv`.
   Harmless for users, but worth a comment or removal before publishing.

Non-blockers checked and cleared, Isaac Sim/Isaac Lab and isaaclab_assets
are acceptable third-party dependencies (EULA acceptance via
OMNI_KIT_ACCEPT_EULA), third_party/act and third_party/diffusion_policy
ship with upstream LICENSE and NOTICE, no supported code references the
old repo or absolute local paths, and the frozen records' embedded
old-machine paths are documented provenance, not dependencies.

## Progress Log

- 2026-08-25, extraction, rename, and adoption restructuring complete,
  five commits on main, all verification green (30/30 Isaac tests, 16
  pure tests, checkpoint loads, driver end-to-end check, Tier-0
  reproduction incl. the exact [0.18, 0.44] gap interval). Full report
  below.
- 2026-08-25, blocker 1 resolved, Ferrari-Canny scorer vendored into
  `mdp/ferrari_canny.py`, validated exactly against all 6322 recorded
  contact sets, epsilon_beta optional behind `PARCELSTOW_FIRMGRASP`,
  Isaac suite rerun after the rewire (commit 70f295d).
- 2026-08-25 evening, blocker-closing pass. Frozen asset JSONs restored
  to original bytes (ba5f947). Provenance verified by sha256 against
  upstream clones, license structure added (5ebec69). merge script
  recovered and shown to regenerate the merged URDF byte-identically.
  All 11 artifacts uploaded to the PRIVATE HF dataset
  cenwerem/parcelstow, manifest carries hf_path and resolve urls,
  download_artifacts.py goes through huggingface_hub with sha256
  verification and a --names selector. Clean-environment validation,
  fresh clone plus uv venv with numpy and matplotlib runs all four
  reproduce targets and reproduces the [0.18, 0.44] gap interval, then
  downloads ACT-A, DAgger, demonstrations, and the videos bundle with
  matching sha256, and the downloaded ACT-A constructs StateACT and
  produces a (1, 100, 16) action chunk. Dependency extras and the
  tested-version table added, pyright block marked developer-local.
  Isaac suite 30/30 after everything.
- Awaiting user decisions ONLY, HF repo visibility flip to public,
  GitHub remote creation and public visibility, v0.1.0 tag, arXiv id
  into CITATION.cff and README.

---

# Release Report, 2026-08-25

The extraction and adoption-oriented restructuring are done. The repo
sits at `~/ResearchProjects/parcelstow`, five commits on `main`, no
remote, not public. Full report below.

## 1. Final Tree

```
parcelstow/
  README.md  CITATION.cff  pyproject.toml  .gitignore  .gitattributes
  .pre-commit-config.yaml  .github/workflows/ci.yml
  source/parcelstow/            Isaac Lab extension (config, setup.py)
    parcelstow/robots.py
    parcelstow/tasks/manager_based/parcel_stow/   env cfg, geometry, mdp/, agents/
  scripts/
    run_task.py  evaluate.py  plot_envelope.py  reproduce.py
    download_artifacts.py  isaac_run.sh
    manipulation/               validated drivers and analyzers (28 files)
  examples/custom_policy.py
  docs/  BENCHMARK.md  TASK_SPEC.md  POLICY_INTERFACE.md
         REPRODUCING_THE_PAPER.md  DATA_AND_CHECKPOINTS.md  DIAGNOSTICS.md
  data/records/                 episode records (gz) + summaries + replication/
  experiments/paper/results/    frozen derived analyses
  artifacts/manifest.json       external-artifact inventory with sha256
  media/                        envelope figure, stages figure, r=2 comparison gif+mp4
  assets/                       g1_l6 (URDF, meshes, usd_both), linkerhand, frozen JSONs
  third_party/{act,diffusion_policy}
  tests/                        pure + simulator-backed suites
  outputs/                      gitignored, local 3.2 GB artifact copy at outputs/paper
```

419 tracked files, 196 MB tracked (165 MB is the robot asset).

## 2. Copied, Omitted, Renamed vs. the Extraction Plan

Renames beyond the plan. `experiments/wrl_functional` became
`experiments/paper`, its checks moved to `tests/`, `TASK_SPEC.md` moved
to `docs/`, and `isaac_run.sh` moved to `scripts/` with the hardcoded
machine paths replaced by `ISAACLAB_VENV`. Additions forced by
correctness. `parcel_stow` imported symbols from the dropped
`gdf_reach`/`stand_reach` packages, so the extraction vendored them
verbatim (`mdp/contacts.py`, `mdp/guards.py`, `CHAIN_ACTUATED` and the
table constants into the env cfg, and a flattened
`agents/rsl_rl_ppo_cfg.py` because all eight drivers resolve the hydra
`rsl_rl_cfg_entry_point`). `state_act.py` (the ACT architecture) lived in
the excluded `scripts/baselines` and moved into `scripts/manipulation`.
The robot USDs, gitignored in the old repo and therefore machine-local,
now ship in Git (`usd_both`, 41 MB, self-contained) since the task cannot
spawn without them, and the unused right-arm-only USD variant was
dropped. Omissions per instruction. All internal figure builders
(`make_stow_figures.py`, replication and rollout figure scripts) and the
internal figure PDFs under `results/figures/` stayed behind,
`IMPLEMENTATION_LOG.md` and the internal experiment README stayed
behind. One functional deviation. The gym registration's RL entry point
now names the vendored `ParcelStowPPORunnerCfg` (same field values, new
`experiment_name`), which no released workflow reads.

## 3. Test Results

`pytest tests/ -q` gives 16 passed, 14 skipped (simulator marks).
`pytest tests/ --isaac -q` gives 30/30 passed in 194 s on the RTX 5070
Ti. The end-to-end driver check through `scripts/evaluate.py` ran the
expert 4/4 successes at r=1 and the example custom policy 0/4 with
`acquisition_failure`, both recorded correctly. Ruff passes on the new
public scripts. Pre-existing cosmetic USD visual-prim warnings appear
identically in the old repo's logs.

## 4. Checkpoint Compatibility

ACT-A, Diffusion Policy, and DAgger checkpoints all load on CPU and in
the simulator run, and all are plain state dicts with normalization
stats, so the package rename cannot affect them. The frozen episode
records embed old `outputs/wrl_functional/...` checkpoint paths in their
`checkpoint` fields, left untouched and documented as provenance in
`docs/DATA_AND_CHECKPOINTS.md`.

## 5. README Draft

`~/ResearchProjects/parcelstow/README.md`. It opens with the
one-sentence definition, the r=2 comparison gif, the numeric table
(100/100 both at r=1, expert 84/100 and ACT-A 53/100 at r=2, gap
interval [0.18, 0.44]), the envelope figure, and a navigation table,
then "What can I do with ParcelStow?", the four quickstart tiers, the
policy section, installation, a repository map, and the citation block
with a pending arXiv identifier. Every number in it was verified against
`eval_summary.jsonl`, and the fresh paired-bootstrap implementation
reproduces the interval exactly.

## 6. Supported Public Commands

```
python scripts/run_task.py                       first simulator run, 5 expert episodes
python scripts/evaluate.py --actor <a> --rates .. --episodes ..
python scripts/plot_envelope.py [--gap A B --gap_rate r]
python scripts/reproduce.py {envelope|stages|certificate|expert-ceiling|all}
python scripts/download_artifacts.py {--paper|--demo|--all|--verify}
```

`run_task.py` and `evaluate.py` are thin wrappers re-execing the
validated `eval_stow_policies.py` driver, so no experiment logic was
rewritten. The README separates these from `scripts/manipulation/`
(drivers, documented as internal).

## 7. Policy Adapter

The de facto actor interface already existed in `stow_runtime.py`, so it
is documented as the contract, no new one was invented. Three members,
`name`, `reset(ids, obs=None)`, `act(obs) -> (action, q_target_or_None)`,
over the frozen 147-D observation (rate at index 146) and 16-D action at
50 Hz. `load_actor` gained one additive branch accepting
`module.path:ClassName`, `evaluate.py` passes `--custom_ckpt` through,
and `examples/custom_policy.py` plus `docs/POLICY_INTERFACE.md` (with
the exact slice table, cross-checked against the frozen spec's
51/51/16/7/15/5/1/1 layout) complete the path. The scientific interface
is unchanged.

## 8. Artifact Inventory

In Git, 7 MB of episode records (`data/records/`, gz), 6 MB of frozen
derived analyses, small frozen geometry/trajectory/grasp JSONs.
External, inventoried in `artifacts/manifest.json` with byte sizes and
sha256, all verified locally by `download_artifacts.py --verify`. ACT-A
291 MB, DP 556 MB, DAgger 1 MB, 297 demonstrations 129 MB, ACT-B/C
291 MB each, n50/n100 checkpoints 291 MB each, demo subsets 22+43 MB,
videos bundle 73 MB (`outputs/release/parcelstow_videos.tar.gz`, already
built). Total external about 2.2 GB. Classified as disposable and
excluded, DAgger training intermediates, traces, probe dumps, logs, and
regenerable figure outputs.

## 9. Reproducible without Isaac

Operating-envelope figure and table, the paired-bootstrap gap interval,
the stage-versus-rate figure, the force-closure certificate analysis,
the out-of-sample certificate analysis, the expert-ceiling attribution,
the rate-conditioned margin analysis, and the multiseed, demo-scaling,
and handoff summaries, all from records in Git. The CI workflow runs
lint, pure tests, manifest validation, and four reproduction targets on
every push.

## 10. Still Requiring Isaac and a GPU

Any new evaluation (including a user policy), demonstration collection,
ACT/DP/DAgger retraining, the handoff diagnostics reruns, video
recording, and the simulator-backed physics tests. All documented with
rough runtimes in `docs/REPRODUCING_THE_PAPER.md`.

## 11. Unresolved Hosting and Licensing

- No LICENSE file exists. The old extension's `setup.py` declares
  Apache-2.0 from the Isaac Lab template, but no license was added
  without the author's choice. `third_party/act` and
  `third_party/diffusion_policy` carry their upstream LICENSE and NOTICE
  files. The Unitree/LinkerHand robot meshes and URDFs ship without a
  stated license, worth checking before publication.
- All manifest `url` fields are null pending the hosting choice (about
  2.2 GB, GitHub Releases caps files at 2 GB each so every individual
  file fits, Hugging Face or Zenodo also work). Filling the URLs and
  rerunning `--verify` is the only step left there.
- `extension.toml` has an empty `repository` field awaiting the GitHub
  URL.

## 12. Remaining Old-Name and Absolute-Path References

All deliberate. The frozen episode records and derived analyses contain
old `outputs/wrl_functional` paths and the old machine prefix in
provenance fields (records stay unmutated per the frozen-science rule).
`assets/g1_l6/usd_both/config.yaml` is the URDF-to-USD importer's
generated stamp with old absolute paths, inert at runtime.
`mdp/metrics.py` retained `~/ResearchProjects/firmgrasp` as the guarded
fallback behind the `PARCELSTOW_FIRMGRASP` env var at report time
[superseded, the fallback is removed and epsilon is computed in-repo,
see the blocker table]. `docs/TASK_SPEC.md`
mentions `scripts/vla/expert.py` once as method provenance inside the
frozen spec. Supported code, docs, and configs are otherwise free of
`g1_locomanip_lab`, `wrl_functional`, and absolute paths.

## 13. Code-Paper Discrepancies

None found. The records reproduce every number specified, expert 100/84
and ACT-A 100/53 at r=1/r=2, the gap interval [0.18, 0.44] exactly,
ACT-B/C degradation, the no-force-closure one-sided result via the
certificate analyzer, and the expert-ceiling attribution (arm
utilization at most 0.17 through r=2 while orientation error crosses the
10-degree tolerance).

## 14. Recommended About Text and Release Tag

About, `Isaac Lab benchmark for task-rate robustness evaluation in
learned contact-rich manipulation.` First tag, `v0.1.0` (matches
`CITATION.cff` and the extension version), created after the artifact
URLs and the license are filled so the tagged state is the fully
reproducible one.

Nothing was pushed or published, and the source repo is untouched. The
natural next steps are the license choice, artifact hosting, and the
GitHub remote.
