# Test data (QKV cache)

The evaluation uses **48** cached tensors: **8 scenarios** × **6 sequence lengths** (4096 … 131072), naming:

`qkv_len_{L}_{scenario}_seed_42.pt`

| Scenarios | Description |
|-----------|-------------|
| `sparse_low`, `sparse_med`, `sparse_high` | K-norm–controlled sparsity (uniform-style). |
| `lm_sink_local`, `lm_sparse_global`, `lm_hierarchical`, `lm_local_periodic`, `lm_mixed` | Realistic LLM-style attention (block shared-context injection). |

`manifest.json` lists all files. **`.pt` files are not in git** (too large); generate locally:

```bash
# Sparse scenarios
python scripts/generate_high_sparsity_qkv.py \
  --lengths 4096 8192 16384 32768 65536 131072 \
  --out-dir data/cache

# LM-style scenarios (merges into data/cache/manifest.json)
python scripts/generate_lm_patterns_qkv.py \
  --lengths 4096 8192 16384 32768 65536 131072 \
  --output-dir data/cache
```
