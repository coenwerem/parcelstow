# Asset Attribution

Verified 2026-08-25 by sha256 comparison against the upstream
repositories, docs/ASSET_PROVENANCE.md holds the full provenance chain.
ParcelStow refers to the hand as the RealHand L6, the manufacturer's
current branding, while the redistributed description assets originate
from the manufacturer's legacy LinkerHand repository and retain the
historical LinkerHand naming in upstream files, paths, and metadata.

| asset family | upstream | license | copyright | state |
|---|---|---|---|---|
| `g1_l6/meshes/g1/` (167 mesh files) | [unitreerobotics/unitree_ros](https://github.com/unitreerobotics/unitree_ros), `robots/g1_description/meshes`, commit `4ddbf6d` | BSD-3-Clause ([text](LICENSES/UNITREE_BSD-3-CLAUSE.txt)) | HangZhou YuShu TECHNOLOGY CO.,LTD. (Unitree Robotics) | byte-identical, renamed into the `g1/` subtree |
| `linkerhand_l6/`, `linkerhand_l6_left/` (URDFs and meshes) | [linker-bot/linkerhand-urdf](https://github.com/linker-bot/linkerhand-urdf), `L6/right` and `L6/left`, commit `075cc7d` | Apache-2.0 ([text](LICENSES/LINKERHAND_APACHE-2.0.txt)) | LinkerBot (LinkerHand) | byte-identical |
| `g1_l6/meshes/l6r/`, `g1_l6/meshes/l6l/` | same as above | Apache-2.0 | LinkerBot (LinkerHand) | byte-identical copies placed by the merge script |
| `g1_l6/g1_29dof_l6_both.urdf`, `g1_29dof_l6_right.urdf` | derived from `g1_29dof_rev_1_0.urdf` (Unitree) plus the two L6 URDFs (LinkerHand) | BSD-3-Clause + Apache-2.0 | as above | transformed, merged by `scripts/assets/merge_g1_l6_urdf.py` |
| `g1_l6/usd_both/` | generated from the merged URDF by the Isaac Lab UrdfConverter | derivative of the above | as above | generated, frozen runtime asset |
| `gdf_bank_parcel.json`, `provenance/frogger_parcel/` | synthesized outputs, see docs/ASSET_PROVENANCE.md | n/a (generated data) | project | frozen construction input |
| `parcel_stow_geometry.json`, `parcel_stow_trajectory.json` | produced by `scripts/manipulation/probe_stow_geometry.py` and `build_stow_trajectory.py` | project (Apache-2.0) | project | frozen |

ParcelStow modifications to upstream robot descriptions, mesh files are
unmodified, the merged URDFs strip the stock Unitree rubber-hand links,
rewrite mesh paths into per-source subtrees, and attach the L6 hands to
the wrist yaw links with fixed mount joints.
