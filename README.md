# length-routing-study

Length-aware sparse attention routing: compare **PBS-Attn**, **FlexPrefill**, and an adaptive **Ours** router (mask + PBS backends) on controlled QKV data from 4K–128K tokens.

## What this repo contains

- **`src/length_routing_study/`** — Router (`length_router.py`), sparsity estimator, cost model, empirical PBS/Flex wrappers, block-sparse mask execution (`selection_strategies.py`).
- **`scripts/`** — Eval drivers: `run_comprehensive_eval.py`, `run_variant_comparison.py`, QKV generators (`generate_high_sparsity_qkv.py`, `generate_lm_patterns_qkv.py`).
- **`data/cache/`** — QKV tensors + `manifest.json` (see [`data/README.md`](data/README.md)). **Tensor files are gitignored**; regenerate on your machine.
- **`results/`** — Published metrics JSONs + [`results/README.md`](results/README.md).

## Environment

- Python 3.10+, PyTorch 2.x with CUDA, **NVIDIA GPU** (benchmarks used A800 80GB).
- Optional local checkouts (for full kernel runs): PBS-Attn, FlexPrefill — paths are resolved in `length_routing_study/_paths.py` / env vars:

| Variable | Typical use |
|----------|----------------|
| `LRS_PBS_ROOT` | PBS-Attn source tree |
| `LRS_FLEX_ROOT` | FlexPrefill source tree |

Install the package in editable mode:

```bash
cd length-routing-study
pip install -e .
```

## Quick start

1. **Generate QKV cache** (required before eval; ~tens of GB disk):

   ```bash
   python scripts/generate_high_sparsity_qkv.py --out-dir data/cache
   python scripts/generate_lm_patterns_qkv.py --output-dir data/cache
   ```

2. **Run comprehensive eval** (writes `results/comprehensive_eval.json`):

   ```bash
   export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
   python scripts/run_comprehensive_eval.py --cache-dir data/cache
   ```

3. **Optional: fixed-parameter + adaptive comparison** (writes `results/variant_comparison.json`):

   ```bash
   python scripts/run_variant_comparison.py \
     --cache-dir data/cache \
     --ours-json results/comprehensive_eval.json
   ```

## Paper / report tables

The latest bundled numbers are in `results/comprehensive_eval.json` and `results/variant_comparison.json`. Re-run the scripts above to refresh after code or data changes.

## License

See project root or upstream dependencies for license terms of bundled ideas (PBS / Flex attention implementations are external).
