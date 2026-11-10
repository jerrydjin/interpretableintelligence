"""Locate the real LIBERO-plus assets folder inside an extracted zip and place it.

The official assets.zip is NOT rooted at `assets/`. All 457k entries live under
  inspire/hdd/project/embodied-multimodality/public/syfei/libero_new/release/dataset/LIBERO-plus-0/assets/
so unzipping into libero/libero/ leaves libero/libero/assets empty.
"""
from __future__ import annotations
import os, shutil, zipfile
from pathlib import Path

# README / inspected zip children of the real assets root
ASSET_DIRS = (
    "articulated_objects",
    "new_objects",
    "scenes",
    "stable_hope_objects",
    "stable_scanned_objects",
    "textures",
    "turbosquid_objects",
)
ASSET_FILES = ("wall.xml", "serving_region.xml", "wall_frames.stl")
KNOWN_ASSET = "wall.xml"  # asserted after install


def is_assets_root(p: Path) -> bool:
    p = Path(p)
    if not p.is_dir():
        return False
    n_dirs = sum(1 for d in ASSET_DIRS if (p / d).is_dir())
    n_files = sum(1 for f in ASSET_FILES if (p / f).is_file())
    return n_dirs >= 2 and n_files >= 1


def find_assets_root(extract_root: Path) -> Path:
    extract_root = Path(extract_root)
    # Fast path: common layouts
    for cand in (
        extract_root / "assets",
        extract_root,
    ):
        if is_assets_root(cand):
            return cand
    # Walk: first directory that looks like the assets root
    for dirpath, dirnames, filenames in os.walk(extract_root):
        p = Path(dirpath)
        if is_assets_root(p):
            return p
        # prune obvious non-asset branches a bit? keep simple
    raise FileNotFoundError(
        f"could not find LIBERO assets root under {extract_root} "
        f"(looked for dirs {ASSET_DIRS} and files {ASSET_FILES})"
    )


def place_assets(src_root: Path, dest: Path) -> Path:
    """Move or symlink src_root into dest (the path LIBERO expects)."""
    src_root = Path(src_root).resolve()
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        if dest.is_symlink() or dest.is_file():
            dest.unlink()
        elif dest.is_dir():
            # If dest is already a valid assets root, keep it
            if is_assets_root(dest) and dest.resolve() == src_root:
                return dest
            shutil.rmtree(dest)
    # Prefer symlink (saves a second copy of 6.4 GB); fall back to move
    try:
        dest.symlink_to(src_root, target_is_directory=True)
    except OSError:
        shutil.move(str(src_root), str(dest))
    if not is_assets_root(dest):
        raise RuntimeError(f"placed assets at {dest} but markers missing")
    return dest


def extract_zip_and_place(zip_path: Path, extract_dir: Path, dest: Path) -> Path:
    zip_path = Path(zip_path)
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"not a complete zip: {zip_path} size={zip_path.stat().st_size}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    root = find_assets_root(extract_dir)
    return place_assets(root, dest)
