#!/usr/bin/env python3
"""
Calibrate the theory cost model against empirical sweep results.

Reads a ``LengthStudyResult`` JSON produced by ``run_length_study.py``
and outputs a ``CalibratedProfile`` JSON with fitted PBS / Flex coefficients.

Usage
-----
  python scripts/calibrate_cost_model.py \
      --input results/length_study_latest.json \
      --output results/calibrated_profile.json \
      --num-layers 36 \
      --block-size 128
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from length_routing_study.calibration import calibrate_from_study


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",      type=Path, required=True,
                    help="Path to length_study_*.json from run_length_study.py")
    ap.add_argument("--output",     type=Path, default=None,
                    help="Where to write calibrated_profile.json (default: next to input)")
    ap.add_argument("--num-layers", type=int,  default=36)
    ap.add_argument("--block-size", type=int,  default=128)
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        study_data = json.load(f)

    profile = calibrate_from_study(
        study_data,
        num_layers=args.num_layers,
        block_size=args.block_size,
    )
    profile.print_summary()

    out = args.output or args.input.parent / "calibrated_profile.json"
    profile.save(out)

    # Print how the fitted values compare to the default qwen2 profile
    from length_routing_study.cost_model import get_profile
    default = get_profile("qwen2")
    print("\n── Comparison to default qwen2 profile ─────────────────────────")
    print(f"  pbs_kernel  default={default.pbs_kernel_ms_per_active_block_per_layer:.4e}  "
          f"fitted={profile.pbs_kernel_ms_per_active_block_per_layer:.4e}")
    print(f"  pbs_select  default={default.pbs_select_ms_per_block_pair_per_layer:.4e}  "
          f"fitted={profile.pbs_select_ms_per_block_pair_per_layer:.4e}")
    print(f"  flex_kernel default={default.flex_kernel_ms_per_active_block_per_layer:.4e}  "
          f"fitted={profile.flex_kernel_ms_per_active_block_per_layer:.4e}")
    print(f"  flex_mask   default={default.flex_mask_ms_per_block_log_per_layer:.4e}  "
          f"fitted={profile.flex_mask_ms_per_block_log_per_layer:.4e}")


if __name__ == "__main__":
    main()
