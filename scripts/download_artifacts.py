"""Fetch and verify the demonstrations, checkpoints, and videos listed in
artifacts/manifest.json.

Artifacts live in the Hugging Face dataset repository named by the
manifest's hf_repo field. With the huggingface_hub package installed the
script downloads through hf_hub_download (resumable, cached, works on a
private repository after hf auth login), otherwise it falls back to the
plain resolve url, which needs the repository to be public. Every
download is verified against the manifest sha256.

Bundles,
  --paper   demonstrations, checkpoints, and subsets behind the paper
  --demo    the three reported ACT checkpoints and comparison video
  --all     every artifact
  --task    released ACT checkpoint and demonstrations for one task
  --manuscript  canonical September 21 checkpoints and demonstrations
  --local-hf DIR  copy from a local HF checkout without network access
  --verify  no downloads; verify selected files (all when no selection is given)

Files already present with a matching checksum are skipped.

Run,
  python scripts/download_artifacts.py --paper
  python scripts/download_artifacts.py --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.request

from task_registry import ALIASES, get_task

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MANIFEST = os.path.join(REPO, "artifacts", "manifest.json")


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def check(name, art):
    path = os.path.join(REPO, art["path"])
    if not os.path.exists(path):
        return "missing"
    if os.path.getsize(path) != art["bytes"]:
        return "size mismatch"
    if art.get("sha256") and sha256_of(path) != art["sha256"]:
        return "checksum mismatch"
    return "ok"


def fetch(name, art, manifest, local_hf=None, revision="main"):
    path = os.path.join(REPO, art["path"])
    if os.path.exists(path):
        raise FileExistsError(f"Refusing to replace a mismatched existing artifact: {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    repo_id = art.get("hf_repo") or manifest.get("hf_repo")
    hf_path = art.get("hf_path")
    fd, temporary = tempfile.mkstemp(prefix=".download-", dir=os.path.dirname(path))
    os.close(fd)
    try:
        if local_hf:
            source = os.path.join(local_hf, hf_path)
            print(f"[copy] {name} <- {source}")
            shutil.copyfile(source, temporary)
        elif repo_id and hf_path:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError:
                url = f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{hf_path}"
                urllib.request.urlretrieve(url, temporary)
            else:
                print(f"[fetch] {name} <- hf:{repo_id}/{hf_path}@{revision}")
                got = hf_hub_download(repo_id=repo_id, filename=hf_path,
                                      repo_type=manifest.get("hf_repo_type", "dataset"), revision=revision)
                shutil.copyfile(got, temporary)
        else:
            urllib.request.urlretrieve(art["url"], temporary)
        if os.path.getsize(temporary) != art["bytes"] or sha256_of(temporary) != art["sha256"]:
            raise ValueError(
                f"Downloaded artifact failed verification: {name}; a local LFS pointer is not a tensor file"
            )
        # Exclusive creation prevents replacing an artifact created concurrently.
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--manuscript", action="store_true")
    ap.add_argument("--local-hf", default=None, help="local HF checkout; no network")
    ap.add_argument("--revision", default="main", help="HF revision or commit; checksums always enforced")
    ap.add_argument("--task", choices=ALIASES,
                    help="fetch the selected task's ACT checkpoint and demonstrations")
    ap.add_argument("--names", nargs="+", default=None,
                    help="specific artifact names from the manifest")
    args = ap.parse_args()

    with open(MANIFEST) as f:
        m = json.load(f)
    names = set()
    if args.all:
        names = set(m["artifacts"])
    if args.manuscript:
        names |= set(m["bundles"]["manuscript"])
    if args.paper:
        names |= set(m["bundles"]["paper"])
    if args.demo:
        names |= set(m["bundles"]["demo"])
    if args.task:
        spec = get_task(args.task)
        names |= {spec.checkpoint_artifact, spec.demonstration_artifact}
    if args.names:
        names |= set(args.names)
    if args.verify and not names:
        names = set(m["artifacts"])
    if not names:
        ap.print_help()
        return 2

    failures = 0
    for name in sorted(names):
        art = m["artifacts"][name]
        status = check(name, art)
        if status == "ok":
            print(f"[ok] {name} at {art['path']}")
            continue
        if args.verify:
            print(f"[{status}] {name} at {art['path']}")
            failures += 1
            continue
        if not art.get("url") and not art.get("hf_path"):
            print(f"[no host yet] {name}, url unset in artifacts/manifest.json")
            failures += 1
            continue
        fetch(name, art, m, args.local_hf, args.revision)
        status = check(name, art)
        print(f"[{status}] {name}")
        failures += status != "ok"
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
