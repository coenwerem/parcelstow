# Data and Checkpoints

## Released with the Main Modules in the Repository

| path | content | size |
|---|---|---|
| `data/records/eval_summary.jsonl` | per-condition summary, 4 actors x 7 rates x 100 episodes | 76 KB |
| `data/records/{expert,act,dagger,dp}_episodes.jsonl.gz` | complete episode-level evaluation records of the main comparison | 4.4 MB |
| `data/records/replication/` | episode records of the ACT seed replication (A rerun, B, C) and the 50/100-demonstration scaling runs | 2.6 MB |
| `data/records/upright/` | upright placement expert and ACT records and summaries, 8 rates x 100 episodes | 2.6 MB |
| `data/records/peg/` | keyed peg insertion expert and ACT records and summaries, 8 rates x 100 episodes | 2.1 MB |
| `experiments/paper/results/` | frozen derived analyses of the paper, certificate, handoff, expert sweep, multiseed and scaling summaries | 6 MB |
| `assets/parcel_stow_geometry.json` | frozen receptacle and path geometry | small |
| `assets/parcel_stow_trajectory.json` | frozen IK trajectory of the expert | small |
| `assets/gdf_bank_parcel.json` | frozen grasp bank entry with provenance | small |

The episode records are the paper's raw evaluation output, unmodified.
Field paths inside them (for example `checkpoint`) refer to the layout of
the machine where they were produced and stay untouched for provenance.

## External Artifacts

Demonstrations, checkpoints, and rollout videos exceed sensible Git sizes
and live in the Hugging Face dataset repository
[`cenwerem/parcelstow`](https://huggingface.co/datasets/cenwerem/parcelstow),
inventoried with sizes, sha256, and hosted paths in
[`artifacts/manifest.json`](../artifacts/manifest.json). Fetch them with

```bash
python scripts/download_artifacts.py --paper     # demos + all checkpoints
python scripts/download_artifacts.py --demo      # ACT-A checkpoint + videos
python scripts/download_artifacts.py --names X   # one specific artifact
python scripts/download_artifacts.py --verify    # verify local files
python scripts/download_artifacts.py --task parcel
python scripts/download_artifacts.py --task upright
python scripts/download_artifacts.py --task peg
```

With the `huggingface_hub` package installed (`pip install
huggingface_hub`, or the `[hub]` extra) downloads go through the Hub
client, resumable and cached, otherwise the script falls back to the
plain resolve URL. Every download is verified against the manifest
sha256 and arrives at its manifest path under `outputs/`, where the
drivers expect it.

Parcel ACT-A was trained on 297 successful demonstrations over `r` in
`[0.5, 2.0]` and supplies the primary nominally matched comparison. The
upright ACT checkpoint was trained on 315 successful demonstrations over
`[0.75, 1.75]`; it succeeds in 39/100 nominal episodes versus 92/100 for the
expert. The peg ACT checkpoint was trained on 325 successful demonstrations
over `[0.5, 1.0]`; it succeeds in 76/100 nominal episodes versus 94/100 for the
expert and has 0/100 acquisitions at every evaluated `r >= 1.5`. The upright
and peg ACT results are not primary matched comparisons.

## Regenerating Instead of Downloading

Everything external regenerates from the repository with Isaac Lab and a
GPU, [REPRODUCING_THE_PAPER.md](REPRODUCING_THE_PAPER.md) lists the
commands and rough runtimes.
