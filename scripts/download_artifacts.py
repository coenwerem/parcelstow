"""Fetch and verify the external release artifacts listed in
artifacts/manifest.json.

Artifacts live in the Hugging Face dataset repository named by the
manifest's hf_repo field. With the huggingface_hub package installed the
script downloads through hf_hub_download (resumable, cached, works on a
private repository after hf auth login), otherwise it falls back to the
plain resolve url, which needs the repository to be public. Every
download is verified against the manifest sha256.

Bundles,
  --paper   demonstrations, checkpoints, and subsets behind the paper
  --demo    the ACT-A checkpoint and the rollout videos
  --all     every artifact
  --verify  no downloads, verify size and sha256 of local files

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
import urllib.request

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


def fetch(name, art, manifest):
    path = os.path.join(REPO, art["path"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    repo_id = manifest.get("hf_repo")
    hf_path = art.get("hf_path")
    if repo_id and hf_path:
        try:
            from huggingface_hub import hf_hub_download

            print(f"[fetch] {name} <- hf:{repo_id}/{hf_path}")
            got = hf_hub_download(
                repo_id=repo_id,
                filename=hf_path,
                repo_type=manifest.get("hf_repo_type", "dataset"),
            )
            shutil.copyfile(got, path)
            return
        except ImportError:
            pass
    print(f"[fetch] {name} <- {art['url']}")
    urllib.request.urlretrieve(art["url"], path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--names", nargs="+", default=None,
                    help="specific artifact names from the manifest")
    args = ap.parse_args()

    with open(MANIFEST) as f:
        m = json.load(f)
    names = set()
    if args.all or args.verify:
        names = set(m["artifacts"])
    if args.paper:
        names |= set(m["bundles"]["paper"])
    if args.demo:
        names |= set(m["bundles"]["demo"])
    if args.names:
        names |= set(args.names)
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
            failures += status != "missing"
            continue
        if not art.get("url") and not art.get("hf_path"):
            print(f"[no host yet] {name}, url unset in artifacts/manifest.json")
            failures += 1
            continue
        fetch(name, art, m)
        status = check(name, art)
        print(f"[{status}] {name}")
        failures += status != "ok"
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
