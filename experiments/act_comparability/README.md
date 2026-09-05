# ACT Comparability Study

This directory records the diagnosis and bounded development experiments for
the upright-placement and keyed-peg-insertion ACT policies. It does not change
the released task definitions or the immutable parcel-insertion v1.0.0
release.

- [Pipeline Audit](AUDIT.md) traces the current data, model, training, and
  evaluation paths and separates observations from proposed explanations.
- [Experimental Protocol](PROTOCOL.md) defines the data partitions,
  checkpoint selection, indexed evaluation banks, stopping rules, and retained
  evidence for new ACT policies.

Development outputs belong under `outputs/act_comparability/`, which is
excluded from Git. Small configuration and result summaries may be copied into
this directory only after their provenance and checksums have been recorded.
