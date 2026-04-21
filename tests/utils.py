from __future__ import annotations

import shutil
import subprocess
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATA = shutil.which("stata-mp")
SPUR_COMMIT = "e694f9b09f04657554321ce90e03190280464792"
SPUR_STATA = f"https://raw.githubusercontent.com/pdavidboll/SPUR/{SPUR_COMMIT}/"
STATA_CACHE_ROOT = PROJECT_ROOT / ".pytest_cache" / "stata_spur" / SPUR_COMMIT


def stata_path(path: Path) -> str:
    return path.resolve().as_posix()


def ensure_spur_stata_installed() -> Path:
    assert STATA is not None, "stata-mp not found"

    root = STATA_CACHE_ROOT
    plus = root / "plus"
    personal = root / "personal"
    plus.mkdir(parents=True, exist_ok=True)
    personal.mkdir(parents=True, exist_ok=True)

    moremata_ok = (plus / "l" / "lmoremata.mlib").exists()
    spur_ok = (plus / "s" / "spurtransform.ado").exists()
    if moremata_ok and spur_ok:
        return root

    script = textwrap.dedent(
        f"""
        clear all
        set more off

        sysdir set PLUS "{stata_path(plus)}"
        sysdir set PERSONAL "{stata_path(personal)}"

        ssc install moremata, replace
        net install spur, replace from("{SPUR_STATA}")
        """
    )
    execute_stata_command(script, root)
    return root


def execute_stata_command(script: str, cwd: Path) -> None:
    assert STATA is not None, "stata-mp not found"
    result = subprocess.run(
        [STATA, "-q"],
        cwd=cwd,
        input=script + "\nexit, clear\n",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr.strip() or result.stdout.strip()
