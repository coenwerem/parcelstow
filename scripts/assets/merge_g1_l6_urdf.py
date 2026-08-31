"""Merge the Unitree G1 29-DoF URDF with LinkerHand L6 hands on both wrists.

The script follows the same procedure Isaac Lab documents for the provided
G1 plus Inspire hand asset, a URDF-level merge with fixed wrist joints,
then a single URDF-to-USD conversion. The stock rubber-hand links that the
Unitree URDF fixes to each wrist are removed first. Mount transforms come
from the validated MuJoCo graft in MuJoCoDex, rh_root and lh_root under
the wrist yaw links. MuJoCo quaternions are scalar-first and URDF rpy is
extrinsic xyz, so the conversion goes through scipy with explicit
reordering.
"""

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--g1_urdf", type=Path, required=True,
                 help="path to g1_29dof_rev_1_0.urdf in a unitree_ros checkout "
                      "(robots/g1_description), meshes/ expected beside it")
_ap.add_argument("--assets", type=Path,
                 default=Path(__file__).resolve().parents[2] / "assets",
                 help="assets directory holding linkerhand_l6 and linkerhand_l6_left")
_args = _ap.parse_args()

G1_URDF = _args.g1_urdf
G1_MESHES = G1_URDF.parent / "meshes"
ASSETS = _args.assets
OUT_DIR = ASSETS / "g1_l6"

RUBBER_LINKS = ("left_rubber_hand", "right_rubber_hand")

HANDS = (
    (
        ASSETS / "linkerhand_l6" / "linkerhand_l6v3.1_right.urdf",
        "l6r",
        "rh_hand_base_link",
        "right_wrist_yaw_link",
        (0.0415, -0.003, 0.0),
        (0.5, 0.5, 0.5, 0.5),
    ),
    (
        ASSETS / "linkerhand_l6_left" / "linkerhand_l6v3.1_left.urdf",
        "l6l",
        "lh_hand_base_link",
        "left_wrist_yaw_link",
        (0.0415, 0.003, 0.0),
        (0.5, -0.5, 0.5, -0.5),
    ),
)

def rewrite_mesh_paths(root: ET.Element, prefix: str) -> None:
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename")
        if fn and fn.startswith("meshes/"):
            mesh.set("filename", fn.replace("meshes/", f"meshes/{prefix}/", 1))

def quat_wxyz_to_rpy(quat_wxyz) -> np.ndarray:
    q = np.asarray(quat_wxyz)[[1, 2, 3, 0]]
    return R.from_quat(q).as_euler("xyz")

def strip_rubber_hands(g1: ET.Element) -> int:
    removed = 0
    for link in list(g1.findall("link")):
        if link.get("name") in RUBBER_LINKS:
            g1.remove(link)
            removed += 1
    for joint in list(g1.findall("joint")):
        child = joint.find("child")
        if child is not None and child.get("link") in RUBBER_LINKS:
            g1.remove(joint)
    return removed

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mesh_sources = [("g1", G1_MESHES)] + [
        (prefix, urdf.parent / "meshes") for urdf, prefix, *_ in HANDS
    ]
    for prefix, src in mesh_sources:
        dst = OUT_DIR / "meshes" / prefix
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    stale = OUT_DIR / "meshes" / "l6"
    if stale.exists():
        shutil.rmtree(stale)

    g1 = ET.parse(G1_URDF).getroot()
    rewrite_mesh_paths(g1, "g1")
    assert strip_rubber_hands(g1) == 2
    g1_links = {l.get("name") for l in g1.findall("link")}

    for urdf, prefix, root_link, wrist_link, pos, quat in HANDS:
        hand = ET.parse(urdf).getroot()
        rewrite_mesh_paths(hand, prefix)

        assert wrist_link in g1_links
        children = {j.find("child").get("link") for j in hand.findall("joint")}
        actual_root = ({l.get("name") for l in hand.findall("link")} - children).pop()
        assert actual_root == root_link, actual_root
        overlap = g1_links & {l.get("name") for l in hand.findall("link")}
        assert not overlap, overlap

        rpy = quat_wxyz_to_rpy(quat)
        mount = ET.SubElement(
            g1, "joint", name=f"{prefix}_hand_mount_joint", type="fixed"
        )
        ET.SubElement(
            mount,
            "origin",
            xyz=" ".join(f"{v:.6f}" for v in pos),
            rpy=" ".join(f"{v:.6f}" for v in rpy),
        )
        ET.SubElement(mount, "parent", link=wrist_link)
        ET.SubElement(mount, "child", link=root_link)
        for elem in list(hand):
            g1.append(elem)
        g1_links |= {l.get("name") for l in g1.findall("link")}

    g1.set("name", "g1_29dof_l6_both")
    out = OUT_DIR / "g1_29dof_l6_both.urdf"
    ET.ElementTree(g1).write(out, xml_declaration=True, encoding="utf-8")
    n_joints = len(g1.findall("joint"))
    n_mimic = len([j for j in g1.findall("joint") if j.find("mimic") is not None])
    print(f"wrote {out}, joints {n_joints}, mimic {n_mimic}")

if __name__ == "__main__":
    main()
