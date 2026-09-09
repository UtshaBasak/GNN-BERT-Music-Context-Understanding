#!/usr/bin/env python
"""Build the submission archive from exactly what git tracks.

    python scripts/make_submission_zip.py

Taking the file list from ``git ls-files`` rather than walking the directory is
the point: the archive then cannot pick up an untracked scratch file, a stale
checkpoint or a local cache, and it cannot silently miss something that was
committed after the last build. It also prints what it excluded, so an omission
is a visible decision rather than an accident.

Re-run this after compiling the report PDF and committing it, or the archive
will be missing the one deliverable a grader looks for first.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import get_logger, project_root  # noqa: E402

LOGGER = get_logger("gbmc.submission")

#: quarantined synthetic-smoke artifacts; see that directory's README
EXCLUDE_PREFIXES = ("results/_synthetic_smoke/",)

#: the five items the brief requires, and where each lives
REQUIRED = {
    "1. source code": "src/train.py",
    "2. graph samples": "data/processed/sample_graphs/",
    "3. tables + plots": "results/metrics.json",
    "4. report PDF": "report/final_report.pdf",
    "5. demo notebook": "notebooks/demo_context.ipynb",
}


def tracked(root: Path) -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=root,
                         capture_output=True, text=True, check=True)
    return out.stdout.split()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=None,
                        help="output path (default: ../SUBMISSION/submission_<repo>.zip)")
    args = parser.parse_args(argv)

    root = project_root()
    files = tracked(root)
    if not files:
        LOGGER.error("git ls-files returned nothing; is this a repository?")
        return 1

    out = Path(args.out) if args.out else (
        root.parent / "SUBMISSION" /
        f"submission_CSE425_{root.name.replace('-', '_')}.zip")
    out.parent.mkdir(parents=True, exist_ok=True)

    kept, skipped, absent = [], [], []
    for rel in files:
        if rel.startswith(EXCLUDE_PREFIXES):
            skipped.append(rel)
            continue
        if not (root / rel).is_file():
            absent.append(rel)
            continue
        kept.append(rel)

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for rel in kept:
            z.write(root / rel, f"{root.name}/{rel}")

    print(f"{len(kept)} files -> {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    if skipped:
        print(f"\nexcluded on purpose ({len(skipped)}):")
        for rel in skipped:
            print(f"   {rel}")
    if absent:
        print(f"\nWARNING: tracked but not on disk ({len(absent)}):")
        for rel in absent:
            print(f"   {rel}")

    print("\nrequired items:")
    ok = True
    for label, probe in REQUIRED.items():
        present = (root / probe).exists()
        ok &= present
        print(f"   {'yes' if present else 'NO '}  {label:<22} {probe}")
    if not ok:
        print("\nAt least one required item is absent. The report PDF is built on "
              "Overleaf; see report/README.md, then re-run this script.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
