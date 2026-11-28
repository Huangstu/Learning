"""Inventory OpenSID files without changing the dataset (Python standard library)."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"Directory does not exist: {root}")
    files = []
    counts = Counter()
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            relative = path.relative_to(root)
            label = relative.parts[0] if len(relative.parts) > 1 else "__unassigned__"
            counts[label] += 1
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            files.append({"path": relative.as_posix(), "folder_label": label,
                          "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    report = {"root": str(root), "image_count": len(files), "folder_counts": dict(counts),
              "paper_expected_image_count": 9194, "paper_expected_class_count": 8,
              "matches_paper_counts_only": len(files) == 9194 and len(counts) == 8,
              "note": "Count agreement alone does not verify labels, official splits, or provenance.",
              "files": files}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"images": len(files), "classes": dict(counts)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
