"""Local launcher for the Next.js browser UI."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    if shutil.which("npm") is None:
        print("Next.js UI needs Node.js and npm on PATH.", file=sys.stderr)
        return 2

    if not (ROOT / "node_modules" / "next").exists():
        print("Next.js UI dependencies are missing.", file=sys.stderr)
        print("Run `npm install` and `npm run dev`.", file=sys.stderr)
        return 2

    return subprocess.call(["npm", "run", "dev"], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
