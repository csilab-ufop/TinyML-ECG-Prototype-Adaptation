#!/usr/bin/env python3
"""Create or patch the generated ModusToolbox dual-core replay project."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


MTB = Path(__file__).resolve().parent
PROJECT_NAME = "psoc6_dual_core_ecg"
SOURCE_FILES = (
    "../../../../app/source/ecg_head_adaptation_app.c",
    "../../../../common/source/backbone_runtime.c",
    "../../../../common/source/personalization.c",
)
INCLUDE_DIRS = (
    "../../../../app/include",
    "../../../../common/include",
    "../../../../generated",
)


def make_assignment(name: str, values: tuple[str, ...]) -> str:
    continuation = " " + chr(92) + "\n        "
    return name + "=" + continuation.join(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify without changing files")
    parser.add_argument("--workspace", type=Path, default=MTB / "build_workspace")
    parser.add_argument(
        "--tools-dir",
        type=Path,
        default=Path(os.environ.get("MTB_TOOLS_DIR", "/opt/Tools/ModusToolbox/tools_3.8")),
    )
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    project = workspace / PROJECT_NAME
    if not project.is_dir() and not args.check:
        creator = args.tools_dir / "project-creator" / "project-creator-cli"
        if not creator.is_file():
            raise SystemExit(f"ModusToolbox project creator not found: {creator}")
        workspace.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                str(creator), "--board-id", "CY8CPROTO-063-BLE",
                "--app-id", "mtb-example-psoc6-dual-cpu-empty-app",
                "--user-app-name", PROJECT_NAME, "--target-dir", str(workspace),
            ],
            check=True,
        )

    makefile = project / "proj_cm4" / "Makefile"
    main_c = project / "proj_cm4" / "main.c"
    if not makefile.is_file() or not main_c.is_file():
        raise SystemExit(f"Incomplete generated project: {project}")

    current = makefile.read_text(encoding="utf-8")
    updated = current
    for name, values in (("SOURCES", SOURCE_FILES), ("INCLUDES", INCLUDE_DIRS)):
        block = make_assignment(name, values)
        if block in updated:
            continue
        original = f"{name}=\n"
        if updated.count(original) != 1:
            raise SystemExit(f"Cannot safely locate {name} in {makefile}")
        updated = updated.replace(original, block + "\n", 1)

    expected_main = (MTB / "cm4_main.c").read_bytes()
    main_matches = main_c.read_bytes() == expected_main
    if args.check:
        if updated != current or not main_matches:
            raise SystemExit("Generated CM4 project differs from the published integration")
        print(f"Verified ModusToolbox integration: {project}")
        return

    if updated != current:
        makefile.write_text(updated, encoding="utf-8")
    if not main_matches:
        shutil.copyfile(MTB / "cm4_main.c", main_c)
    print(f"Prepared ModusToolbox replay project: {project}")


if __name__ == "__main__":
    main()
