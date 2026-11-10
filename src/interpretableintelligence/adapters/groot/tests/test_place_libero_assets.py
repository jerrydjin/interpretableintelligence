"""Synthetic zip with the REAL nested layout (no 6.4 GB download)."""
import sys, zipfile, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from place_libero_assets import find_assets_root, extract_zip_and_place, is_assets_root, KNOWN_ASSET

PREFIX = "inspire/hdd/project/embodied-multimodality/public/syfei/libero_new/release/dataset/LIBERO-plus-0/assets/"

def _write_tree(z: zipfile.ZipFile, prefix: str):
    for d in ("scenes", "articulated_objects", "textures", "new_objects"):
        z.writestr(prefix + d + "/placeholder.txt", b"ok")
    z.writestr(prefix + "wall.xml", b"<wall/>")
    z.writestr(prefix + "serving_region.xml", b"<sr/>")
    z.writestr(prefix + "wall_frames.stl", b"solid")

def test_nested_real_layout():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        zp = td / "assets.zip"
        with zipfile.ZipFile(zp, "w") as z:
            _write_tree(z, PREFIX)
        dest = td / "LIBERO-plus" / "libero" / "libero" / "assets"
        ex = td / "extract"
        placed = extract_zip_and_place(zp, ex, dest)
        assert is_assets_root(placed)
        assert (placed / "wall.xml").is_file()
        assert (placed / "scenes").is_dir()
        assert placed.resolve() == dest.resolve() or dest.resolve() == placed.resolve()
        print("OK nested layout ->", dest, "wall.xml", (dest / "wall.xml").exists())

def test_top_level_assets():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        zp = td / "assets.zip"
        with zipfile.ZipFile(zp, "w") as z:
            _write_tree(z, "assets/")
        dest = td / "dest_assets"
        placed = extract_zip_and_place(zp, td / "ex", dest)
        assert (placed / KNOWN_ASSET).is_file()
        print("OK top-level assets/")

def test_incomplete_zip_rejected():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        zp = td / "assets.zip"
        zp.write_bytes(b"PK\x03\x04incomplete")
        try:
            extract_zip_and_place(zp, td / "ex", td / "dest")
        except (ValueError, zipfile.BadZipFile) as e:
            print("OK incomplete rejected:", type(e).__name__)
        else:
            raise AssertionError("incomplete zip should fail")

if __name__ == "__main__":
    test_nested_real_layout()
    test_top_level_assets()
    test_incomplete_zip_rejected()
    print("ALL_PLACE_TESTS_PASSED")
