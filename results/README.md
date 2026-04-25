# Results (canonical)

| File | Description |
|------|-------------|
| `comprehensive_eval.json` | Full eval: PBS fixed, Flex fixed, Ours (adaptive router), 8 scenarios × 6 lengths (4K–128K). |
| `variant_comparison.json` | Fixed-parameter PBS/Flex variants + built-in adaptive vs. Ours (same dataset). |

Regenerate with:

```bash
python scripts/run_comprehensive_eval.py --cache-dir data/cache
python scripts/run_variant_comparison.py --cache-dir data/cache --ours-json results/comprehensive_eval.json
```

GPU and random seed are recorded inside the JSON metadata where present.
