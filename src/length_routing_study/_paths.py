from __future__ import annotations

import os
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT     = PACKAGE_ROOT.parent
PROJECT_ROOT = SRC_ROOT.parent

DEFAULT_BASELINE_ROOT = Path(os.environ.get("LRS_BASELINE_ROOT",  "/root/test"))
DEFAULT_FLEX_ROOT     = Path(os.environ.get("LRS_FLEX_ROOT",      "/root/FlexPrefill"))
DEFAULT_PBS_ROOT      = Path(os.environ.get("LRS_PBS_ROOT",       "/root/pbs-attn-src"))


def _prepend(path: Path) -> None:
    s = str(path.resolve())
    if path.exists() and s not in sys.path:
        sys.path.insert(0, s)


def ensure_project_src_path() -> None:
    _prepend(SRC_ROOT)


def ensure_external_paths(*, include_baseline: bool = True) -> None:
    ensure_project_src_path()
    if include_baseline:
        _prepend(DEFAULT_BASELINE_ROOT)
    _prepend(DEFAULT_FLEX_ROOT)
    _prepend(DEFAULT_PBS_ROOT)
