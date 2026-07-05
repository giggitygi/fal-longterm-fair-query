from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from fal_experiment.data import get_dataset  # noqa: E402


CIFAR10_ARCHIVE = "cifar-10-python.tar.gz"
CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"
CIFAR10_MIRROR_URL = "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/torchvision")
    parser.add_argument("--datasets", nargs="+", default=["FashionMNIST", "MNIST", "CIFAR10"])
    return parser.parse_args()


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_cifar10_archive(root: Path) -> None:
    archive_path = root / CIFAR10_ARCHIVE
    if archive_path.exists() and md5sum(archive_path) == CIFAR10_MD5:
        return

    tmp_path = archive_path.with_suffix(".tar.gz.part")
    print(f"Downloading CIFAR10 mirror to {archive_path}", flush=True)
    with urlopen(CIFAR10_MIRROR_URL, timeout=60) as response, tmp_path.open("wb") as output:
        total = int(response.headers.get("Content-Length", "0"))
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            downloaded += len(chunk)
            if total:
                percent = downloaded * 100 / total
                print(f"\rCIFAR10 mirror download: {percent:5.1f}%", end="", flush=True)
    print("", flush=True)

    actual_md5 = md5sum(tmp_path)
    if actual_md5 != CIFAR10_MD5:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"CIFAR10 md5 mismatch: expected {CIFAR10_MD5}, got {actual_md5}")
    tmp_path.replace(archive_path)


def main() -> None:
    args = parse_args()
    root = ROOT / args.root
    root.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        print(f"Downloading {name} to {root}", flush=True)
        if name == "CIFAR10":
            ensure_cifar10_archive(root)
        get_dataset(name, root, train=True, download=True)
        get_dataset(name, root, train=False, download=True)
        print(f"{name} ready.", flush=True)
    print("Dataset download complete.", flush=True)


if __name__ == "__main__":
    main()
