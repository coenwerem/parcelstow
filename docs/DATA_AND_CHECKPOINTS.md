# Data and checkpoints

## Shipped in the repository

| path | content | size |
|---|---|---|
| `data/records/eval_summary.jsonl` | per-condition summary, 4 actors x 7 rates x 100 episodes | 76 KB |
| `data/records/{expert,act,dagger,dp}_episodes.jsonl.gz` | complete episode-level evaluation records of the main comparison | 4.4 MB |
| `data/records/replication/` | episode records of the ACT seed replication (A rerun, B, C) and the 50/100-demonstration scaling runs | 2.6 MB |
| `experiments/paper/results/` | frozen derived analyses of the paper, certificate, handoff, expert sweep, multiseed and scaling summaries | 6 MB |
| `assets/parcel_stow_geometry.json` | frozen receptacle and path geometry | small |
| `assets/parcel_stow_trajectory.json` | frozen IK trajectory of the expert | small |
| `assets/gdf_bank_parcel.json` | frozen grasp bank entry with provenance | small |

The episode records are the paper's raw evaluation output, unmodified.
Field paths inside them (for example `checkpoint`) refer to the layout of
the machine where they were produced and stay untouched for provenance.

## External artifacts

Demonstrations, checkpoints, and rollout videos exceed sensible Git sizes
and ship as release artifacts, inventoried with sizes and sha256 in
[`artifacts/manifest.json`](../artifacts/manifest.json). Fetch them with

```bash
python scripts/download_artifacts.py --paper   # demos + all checkpoints
python scripts/download_artifacts.py --demo    # ACT-A checkpoint + videos
python scripts/download_artifacts.py --verify  # verify local files
```

Until hosting is finalized the manifest carries `url: null` entries, the
script reports them and continues. Every artifact arrives at its manifest
path under `outputs/`, where the drivers expect it.

## Regenerating instead of downloading

Everything external regenerates from the repository with Isaac Lab and a
GPU, [REPRODUCING_THE_PAPER.md](REPRODUCING_THE_PAPER.md) lists the
commands and rough runtimes.
