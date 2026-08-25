# Asset provenance

Provenance of every frozen construction input shipped in `assets/`.
None of these files needs regeneration to run or evaluate ParcelStow,
the benchmark treats them as frozen inputs. `assets/NOTICE.md` maps each
family to its upstream license.

## Robot asset, G1 with LinkerHand L6 hands

Pipeline behind the frozen runtime asset,

```
Unitree G1 description                LinkerHand L6 description
robots/g1_description/                L6/right, L6/left
g1_29dof_rev_1_0.urdf + meshes       linkerhand_l6v3.1_{right,left}.urdf + meshes
        \                                  /
         scripts/assets/merge_g1_l6_urdf.py
                      |
         assets/g1_l6/g1_29dof_l6_both.urdf   (merged URDF)
                      |
         Isaac Lab UrdfConverter (Isaac Sim URDF importer)
                      |
         assets/g1_l6/usd_both/               (frozen USD, the runtime asset)
```

Verification (2026-08-25), all 167 shipped G1 mesh files are
byte-identical (sha256) to
[unitreerobotics/unitree_ros](https://github.com/unitreerobotics/unitree_ros)
`robots/g1_description/meshes` at commit `4ddbf6d`, and all shipped L6
URDFs and meshes are byte-identical to
[linker-bot/linkerhand-urdf](https://github.com/linker-bot/linkerhand-urdf)
`L6/right` and `L6/left` at commit `075cc7d`. The hand is the LinkerHand
L6 (v3.1 URDFs). The merged URDFs are transformed, the merge strips the
stock Unitree rubber-hand links, prefixes mesh paths, and adds two fixed
mount joints at the wrist yaw links (mount transforms validated in a
MuJoCo graft before the merge).

`assets/g1_l6/usd_both/config.yaml` and `.asset_hash` are files the
Isaac Lab UrdfConverter generated during the one-time URDF-to-USD
conversion (2026-08-02). The absolute paths inside `config.yaml` name
the machine and layout where the conversion ran, no ParcelStow workflow
reads the file, and it stays byte-identical as generated provenance.
Rerunning the conversion is never required, the checked-in USD is the
runtime asset.

## Grasp bank

`assets/gdf_bank_parcel.json` is the frozen five-contact grasp of the
benchmark, one entry with the grasp transform X_OH, joint targets, and
synthesis scores. Its `synthesis` stamp records the producing tool, a
private driver (`scripts/g1_l6_runner.py`) inside a private repository
(`coenwerem/frogger`, commit `4705a49`, 2026-08-18) building on the
public FRoGGeR grasp synthesizer
([alberthli/frogger](https://github.com/alberthli/frogger), MIT License,
Copyright (c) 2023 Albert H. Li). The private repository shares no git
history with upstream, so no upstream base commit can be stated
reliably, and the public upstream alone does not reproduce the bank.

ParcelStow does not require FRoGGeR at runtime, and the benchmark's
scientific claim does not depend on reproducing the grasp synthesis, the
bank is a frozen construction input. `assets/provenance/frogger_parcel/`
preserves the raw synthesizer records (per-scene grasp candidates,
object mesh, run log) behind the frozen entry. Path strings inside these
records and inside the bank's `source_record`/`geometry_file` fields
name the producing machine's layout and stay unmodified.

## Geometry and trajectory

`assets/parcel_stow_geometry.json` (receptacle and path geometry) and
`assets/parcel_stow_trajectory.json` (IK-solved expert knots) were
produced in-repo by `scripts/manipulation/probe_stow_geometry.py
--finalize` and `scripts/manipulation/build_stow_trajectory.py` under
the frozen task specification (docs/TASK_SPEC.md), and are frozen. Their
embedded absolute paths are provenance from the producing machine.
