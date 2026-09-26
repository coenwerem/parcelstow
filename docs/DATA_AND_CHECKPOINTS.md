# Data and Checkpoints

Use the `manuscript` bundle for the checkpoints and tensor datasets behind the
three-task results:

```bash
python3 scripts/download_artifacts.py --manuscript
python3 scripts/download_artifacts.py --manuscript --verify
```

Add `--local-hf ../parcelstow-hf` to copy from a local Hugging Face checkout.
The downloader verifies SHA-256 before installing a file and refuses to replace
a mismatched existing file.

| Files | Content |
|---|---|
| [Artifact index](../data/manuscript_20260921/ARTIFACT_INDEX.csv) | paths, sizes, and hashes for 17 checkpoints and five tensor datasets |
| [Evaluation inventory](../data/manuscript_20260921/MANUSCRIPT_EVIDENCE.csv) | 136 task × actor × speedup-factor conditions from 20 source series |
| [Pairing status](../data/manuscript_20260921/PAIRING_STATUS.csv) | episode-level pairing and permitted comparisons |
| [Record guide](../data/manuscript_20260921/README.md) | packaged record paths and CPU reproduction |
| [Download manifest](../artifacts/manifest.json) | artifact locations and checksums |

The [runbook](RUNBOOK.md#2-file-map-and-checksums) maps each task and method to
its checkpoint, demonstration dataset, and evaluation bank. Its commands specify
the complete settings for training and evaluating the policies.

The [expert–ACT comparison video](https://huggingface.co/datasets/cenwerem/parcelstow/resolve/main/videos/expert_act_comparison.mp4)
shows all three tasks with the checkpoints reported in the results.

Download the comparison video and the three ACT checkpoints shown in it with:

```bash
python3 scripts/download_artifacts.py --demo
```
