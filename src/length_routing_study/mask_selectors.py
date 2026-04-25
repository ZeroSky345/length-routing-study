import math

import torch

from .selector_base import BaseSelector, SelectionResult


def _causal_mask(q_len: int, k_len: int, device: torch.device) -> torch.Tensor:
    return torch.arange(k_len, device=device).view(1, 1, 1, k_len) <= torch.arange(
        q_len, device=device
    ).view(1, 1, q_len, 1)


def _sparsity(mask: torch.Tensor) -> float:
    density = mask.count_nonzero().item() / float(mask.numel())
    return 1.0 - density


def finalize_causal_mask(mask: torch.Tensor) -> torch.Tensor:
    row_any = mask.any(dim=-1)
    need = ~row_any
    if not need.any():
        return mask
    out = mask.clone()
    q_len = out.shape[-2]
    q_idx = torch.arange(q_len, device=mask.device)
    out[:, :, q_idx, q_idx] |= need
    return out


def finalize_selection(result: SelectionResult) -> SelectionResult:
    if result.mask is None:
        return result
    m = finalize_causal_mask(result.mask)
    return SelectionResult(mask=m, sparsity=_sparsity(m), layout=result.layout)


def _causal_window_stats(q_len: int, k_len: int, window_size: int) -> tuple[float, float]:
    if q_len != k_len or q_len <= 0 or k_len <= 0:
        return 0.0, 1.0
    span = min(int(window_size), k_len) + 1
    dense_causal = q_len * (q_len + 1) // 2
    if q_len <= span:
        kept = dense_causal
    else:
        kept = span * q_len - (span * (span - 1) // 2)
    sparsity = 1.0 - (kept / float(q_len * k_len))
    ratio = float(kept) / float(max(1, dense_causal))
    return float(sparsity), float(ratio)


def _window_for_target_causal_density(q_len: int, target_ratio: float, min_w: int, max_w: int) -> int:
    if q_len <= 1:
        return 0
    lo = max(0, int(min_w))
    hi = max(lo, min(int(max_w), q_len - 1))
    target = max(0.0, min(1.0, float(target_ratio)))
    if target <= 0.0:
        return lo
    if target >= 1.0:
        return hi
    best_w = lo
    best_gap = float('inf')
    left, right = lo, hi
    while left <= right:
        mid = (left + right) // 2
        _, ratio = _causal_window_stats(q_len, q_len, mid)
        gap = abs(ratio - target)
        if gap < best_gap:
            best_gap = gap
            best_w = mid
        if ratio < target:
            left = mid + 1
        else:
            right = mid - 1
    for cand in (max(lo, best_w - 1), best_w, min(hi, best_w + 1)):
        _, ratio = _causal_window_stats(q_len, q_len, cand)
        gap = abs(ratio - target)
        if gap < best_gap:
            best_gap = gap
            best_w = cand
    return int(best_w)


def _chunked_topk_indices(q: torch.Tensor, k: torch.Tensor, topk: int, chunk_size: int = 256) -> torch.Tensor:
    b, h, q_len, d = q.shape
    out = torch.empty((b, h, q_len, topk), dtype=torch.long, device=q.device)
    scale = max(d, 1) ** 0.5
    kt = k.transpose(-1, -2).float()
    for start in range(0, q_len, chunk_size):
        end = min(start + chunk_size, q_len)
        q_chunk = q[:, :, start:end, :].float()
        scores = torch.matmul(q_chunk, kt) / scale
        idx = torch.topk(scores, k=topk, dim=-1).indices
        out[:, :, start:end, :] = idx
    return out


class DenseSelector(BaseSelector):
    name = "dense_no_select"

    def select(self, q: torch.Tensor, k: torch.Tensor) -> SelectionResult:
        return SelectionResult(mask=None, sparsity=0.0, layout={"type": "dense", "causal_density_ratio": 1.0})


class FixedWindowSelector(BaseSelector):
    name = "fixed_window"

    def __init__(self, window_size: int = 512) -> None:
        self.window_size = window_size

    def _resolve_window(self, q_len: int, k_len: int) -> int:
        return int(min(self.window_size, k_len))

    def select(self, q: torch.Tensor, k: torch.Tensor) -> SelectionResult:
        q_len = q.shape[-2]
        k_len = k.shape[-2]
        w = self._resolve_window(q_len, k_len)
        if q_len == k_len:
            sparsity, ratio = _causal_window_stats(q_len, k_len, w)
        else:
            ratio = 1.0
            sparsity = 0.0
        return SelectionResult(mask=None, sparsity=float(sparsity), layout={"type": "causal_window", "window_size": w, "causal_density_ratio": ratio})


class TopKScoreSelector(BaseSelector):
    name = "topk_score"

    def __init__(self, keep_ratio: float = 0.2) -> None:
        self.keep_ratio = keep_ratio
        self.chunk_size = 256

    def select(self, q: torch.Tensor, k: torch.Tensor) -> SelectionResult:
        b, h, q_len, _ = q.shape
        k_len = k.shape[-2]
        topk = max(1, int(k_len * self.keep_ratio))
        idx = _chunked_topk_indices(q, k, topk=topk, chunk_size=self.chunk_size)
        mask = torch.zeros((b, h, q_len, k_len), dtype=torch.bool, device=q.device)
        mask.scatter_(-1, idx, True)
        mask = mask & _causal_mask(q_len, k_len, q.device)
        return SelectionResult(mask=mask, sparsity=_sparsity(mask))


class FixedTopKSelector(BaseSelector):
    name = "fixed_topk"

    def __init__(self, topk: int = 256) -> None:
        self.topk = topk
        self.chunk_size = 256

    def select(self, q: torch.Tensor, k: torch.Tensor) -> SelectionResult:
        b, h, q_len, _ = q.shape
        k_len = k.shape[-2]
        topk = min(max(1, self.topk), k_len)
        idx = _chunked_topk_indices(q, k, topk=topk, chunk_size=self.chunk_size)
        mask = torch.zeros((b, h, q_len, k_len), dtype=torch.bool, device=q.device)
        mask.scatter_(-1, idx, True)
        mask = mask & _causal_mask(q_len, k_len, q.device)
        return SelectionResult(mask=mask, sparsity=_sparsity(mask))


class VerticalOnlySelector(BaseSelector):
    name = "vertical_only"

    def __init__(self, keep_columns: int = 256) -> None:
        self.keep_columns = keep_columns

    def select(self, q: torch.Tensor, k: torch.Tensor) -> SelectionResult:
        b, h, q_len, _ = q.shape
        k_len = k.shape[-2]
        key_score = k.float().norm(dim=-1)
        topk = min(max(1, self.keep_columns), k_len)
        col_idx = torch.topk(key_score, k=topk, dim=-1).indices
        mask = torch.zeros((b, h, q_len, k_len), dtype=torch.bool, device=q.device)
        expand_idx = col_idx.unsqueeze(2).expand(b, h, q_len, topk)
        mask.scatter_(-1, expand_idx, True)
        mask = mask & _causal_mask(q_len, k_len, q.device)
        return SelectionResult(mask=mask, sparsity=_sparsity(mask))


class SlashOnlySelector(BaseSelector):
    name = "slash_only"

    def __init__(self, offsets: tuple[int, ...] = (0, 128, 256, 512)) -> None:
        self.offsets = offsets

    def select(self, q: torch.Tensor, k: torch.Tensor) -> SelectionResult:
        b, h, q_len, _ = q.shape
        k_len = k.shape[-2]
        q_idx = torch.arange(q_len, device=q.device).view(1, 1, q_len, 1)
        mask = torch.zeros((b, h, q_len, k_len), dtype=torch.bool, device=q.device)
        for off in self.offsets:
            k_idx = q_idx - off
            valid = (k_idx >= 0) & (k_idx < k_len)
            mask.scatter_(-1, torch.clamp(k_idx, 0, k_len - 1), valid)
        mask = mask & _causal_mask(q_len, k_len, q.device)
        return SelectionResult(mask=mask, sparsity=_sparsity(mask))


class VerticalSlashSelector(BaseSelector):
    name = "vertical_slash"

    def __init__(self, keep_columns: int = 128, offsets: tuple[int, ...] = (0, 64, 128, 256)) -> None:
        self.vertical = VerticalOnlySelector(keep_columns=keep_columns)
        self.slash = SlashOnlySelector(offsets=offsets)

    def select(self, q: torch.Tensor, k: torch.Tensor) -> SelectionResult:
        m1 = self.vertical.select(q, k).mask
        m2 = self.slash.select(q, k).mask
        mask = m1 | m2
        return SelectionResult(mask=mask, sparsity=_sparsity(mask))


class QueryAwareFullBlockSelector(BaseSelector):
    name = "query_aware_full_block"

    def __init__(self, block_size: int = 128, blocks_per_query_block: int = 4) -> None:
        self.block_size = block_size
        self.blocks_per_query_block = blocks_per_query_block

    def select(self, q: torch.Tensor, k: torch.Tensor) -> SelectionResult:
        b, h, q_len, d = q.shape
        k_len = k.shape[-2]
        q_blocks = (q_len + self.block_size - 1) // self.block_size
        k_blocks = (k_len + self.block_size - 1) // self.block_size
        q_pad = q_blocks * self.block_size - q_len
        k_pad = k_blocks * self.block_size - k_len
        q_p = torch.nn.functional.pad(q, (0, 0, 0, q_pad))
        k_p = torch.nn.functional.pad(k, (0, 0, 0, k_pad))
        q_blk = q_p.view(b, h, q_blocks, self.block_size, d).mean(dim=3)
        k_blk = k_p.view(b, h, k_blocks, self.block_size, d).mean(dim=3)
        blk_scores = torch.matmul(q_blk.float(), k_blk.transpose(-1, -2).float()) / max(d, 1) ** 0.5
        topk = min(max(1, self.blocks_per_query_block), k_blocks)
        blk_idx = torch.topk(blk_scores, k=topk, dim=-1).indices
        q_ids = torch.arange(q_len, device=q.device).view(1, 1, q_len, 1)
        q_blk_id = (q_ids // self.block_size).expand(b, h, q_len, 1)
        selected = blk_idx.gather(2, q_blk_id.expand(b, h, q_len, topk))
        k_base = selected * self.block_size
        token_offsets = torch.arange(self.block_size, device=q.device).view(1, 1, 1, 1, self.block_size)
        token_ids = (k_base.unsqueeze(-1) + token_offsets).view(b, h, q_len, -1)
        token_ids = torch.clamp(token_ids, 0, k_len - 1)
        mask = torch.zeros((b, h, q_len, k_len), dtype=torch.bool, device=q.device)
        mask.scatter_(-1, token_ids, True)
        mask = mask & _causal_mask(q_len, k_len, q.device)
        return SelectionResult(mask=mask, sparsity=_sparsity(mask))


class LengthBasedHybridSelector(BaseSelector):
    name = "length_based_hybrid"

    def __init__(self) -> None:
        self.short = FixedWindowSelector(window_size=256)
        self.mid = VerticalSlashSelector(keep_columns=128, offsets=(0, 64, 128, 256))
        self.long = TopKScoreSelector(keep_ratio=0.1)

    def select(self, q: torch.Tensor, k: torch.Tensor) -> SelectionResult:
        q_len = q.shape[-2]
        if q_len <= 1024:
            return self.short.select(q, k)
        if q_len <= 4096:
            return self.mid.select(q, k)
        return self.long.select(q, k)


class AdaptiveFractionWindowSelector(BaseSelector):
    name = "adaptive_fraction_window"

    def __init__(self, fraction: float, min_w: int, max_w: int) -> None:
        self.fraction = float(fraction)
        self.min_w = int(min_w)
        self.max_w = int(max_w)

    def _resolve_window(self, q_len: int, k_len: int) -> int:
        w = max(self.min_w, min(q_len, int(q_len * self.fraction)))
        return int(min(w, self.max_w, k_len))

    def select(self, q: torch.Tensor, k: torch.Tensor) -> SelectionResult:
        q_len = q.shape[-2]
        k_len = k.shape[-2]
        w = self._resolve_window(q_len, k_len)
        if q_len == k_len:
            sparsity, ratio = _causal_window_stats(q_len, k_len, w)
        else:
            ratio = 1.0
            sparsity = 0.0
        return SelectionResult(mask=None, sparsity=float(sparsity), layout={"type": "causal_window", "window_size": int(w), "causal_density_ratio": ratio})


class SqrtWindowSelector(BaseSelector):
    name = "sqrt_window"

    def __init__(self, coeff: float, min_w: int, max_w: int) -> None:
        self.coeff = float(coeff)
        self.min_w = int(min_w)
        self.max_w = int(max_w)

    def _resolve_window(self, q_len: int, k_len: int) -> int:
        w = int(round(self.coeff * math.sqrt(max(1.0, float(q_len)))))
        w = max(self.min_w, min(self.max_w, w))
        return int(min(w, k_len))

    def select(self, q: torch.Tensor, k: torch.Tensor) -> SelectionResult:
        q_len = q.shape[-2]
        k_len = k.shape[-2]
        w = self._resolve_window(q_len, k_len)
        if q_len == k_len:
            sparsity, ratio = _causal_window_stats(q_len, k_len, w)
        else:
            ratio = 1.0
            sparsity = 0.0
        return SelectionResult(mask=None, sparsity=float(sparsity), layout={"type": "causal_window", "window_size": int(w), "causal_density_ratio": ratio})


class TargetDensityWindowSelector(BaseSelector):
    name = "target_density_window"

    def __init__(self, target: float, min_w: int, max_w: int) -> None:
        self.target = float(target)
        self.min_w = int(min_w)
        self.max_w = int(max_w)

    def _resolve_window(self, q_len: int, k_len: int) -> int:
        return _window_for_target_causal_density(q_len, self.target, self.min_w, min(self.max_w, k_len - 1))

    def select(self, q: torch.Tensor, k: torch.Tensor) -> SelectionResult:
        q_len = q.shape[-2]
        k_len = k.shape[-2]
        w = self._resolve_window(q_len, k_len)
        if q_len == k_len:
            sparsity, ratio = _causal_window_stats(q_len, k_len, w)
        else:
            ratio = 1.0
            sparsity = 0.0
        return SelectionResult(mask=None, sparsity=float(sparsity), layout={"type": "causal_window", "window_size": int(w), "causal_density_ratio": ratio, "target_causal_density_ratio": self.target})


class ProgressiveSqrtTopKSelector(BaseSelector):
    name = "progressive_sqrt_topk"

    def __init__(self, coeff: float, lo: float, hi: float) -> None:
        self.coeff = float(coeff)
        self.lo = float(lo)
        self.hi = float(hi)
        self.chunk_size = 256

    def select(self, q: torch.Tensor, k: torch.Tensor) -> SelectionResult:
        b, h, q_len, _ = q.shape
        k_len = k.shape[-2]
        ratio = self.coeff / max(1.0, math.sqrt(float(q_len)))
        ratio = max(self.lo, min(self.hi, ratio))
        topk = max(1, int(k_len * ratio))
        idx = _chunked_topk_indices(q, k, topk=topk, chunk_size=self.chunk_size)
        mask = torch.zeros((b, h, q_len, k_len), dtype=torch.bool, device=q.device)
        mask.scatter_(-1, idx, True)
        mask = mask & _causal_mask(q_len, k_len, q.device)
        return SelectionResult(mask=mask, sparsity=_sparsity(mask))


class TierRouterAlphaSelector(BaseSelector):
    name = "length_tier__alpha"

    def __init__(self) -> None:
        self.a = FixedWindowSelector(window_size=256)
        self.b = VerticalSlashSelector(keep_columns=128, offsets=(0, 64, 128, 256))
        self.c = TopKScoreSelector(keep_ratio=0.12)

    def select(self, q: torch.Tensor, k: torch.Tensor) -> SelectionResult:
        q_len = q.shape[-2]
        if q_len <= 1024:
            return self.a.select(q, k)
        if q_len <= 4096:
            return self.b.select(q, k)
        return self.c.select(q, k)


class TierRouterBetaSelector(BaseSelector):
    name = "length_tier__beta"

    def __init__(self) -> None:
        self.a = FixedWindowSelector(window_size=512)
        self.b = FixedWindowSelector(window_size=1024)
        self.c = TopKScoreSelector(keep_ratio=0.15)

    def select(self, q: torch.Tensor, k: torch.Tensor) -> SelectionResult:
        q_len = q.shape[-2]
        if q_len <= 2048:
            return self.a.select(q, k)
        if q_len <= 8192:
            return self.b.select(q, k)
        return self.c.select(q, k)


class TierRouterGammaSelector(BaseSelector):
    name = "length_tier__gamma"

    def __init__(self) -> None:
        self.a = TopKScoreSelector(keep_ratio=0.22)
        self.b = FixedWindowSelector(window_size=512)
        self.c = FixedTopKSelector(topk=320)

    def select(self, q: torch.Tensor, k: torch.Tensor) -> SelectionResult:
        q_len = q.shape[-2]
        if q_len <= 1024:
            return self.a.select(q, k)
        if q_len <= 4096:
            return self.b.select(q, k)
        return self.c.select(q, k)


def _attach_variant(selector: BaseSelector, variant_name: str, params: dict) -> BaseSelector:
    selector.name = variant_name
    setattr(selector, "variant_params", params)
    return selector


def _effective_window_signature(selector: BaseSelector, eval_lengths):
    if isinstance(selector, DenseSelector):
        return ("dense",)
    if hasattr(selector, "_resolve_window"):
        windows = tuple(int(selector._resolve_window(int(L), int(L))) for L in eval_lengths)
        return ("causal_window",) + windows
    return None


def _dedupe_equivalent_selectors(selectors, eval_lengths):
    if not eval_lengths:
        return selectors
    deduped = []
    seen = set()
    for selector in selectors:
        key = _effective_window_signature(selector, eval_lengths)
        if key is None:
            deduped.append(selector)
            continue
        if key in seen:
            continue
        seen.add(key)
        deduped.append(selector)
    return deduped


def build_selectors_from_space(space: dict, eval_lengths=None):
    selectors = []
    selectors.append(_attach_variant(DenseSelector(), "dense_no_select", {}))

    fw = space.get("fixed_window", {})
    for window_size in fw.get("window_size", [512]):
        selectors.append(_attach_variant(FixedWindowSelector(window_size=int(window_size)), f"fixed_window__window_{int(window_size)}", {"window_size": int(window_size)}))

    topk_score = space.get("topk_score", {})
    for keep_ratio in topk_score.get("keep_ratio", [0.2]):
        selectors.append(_attach_variant(TopKScoreSelector(keep_ratio=float(keep_ratio)), f"topk_score__keep_{float(keep_ratio):.3f}", {"keep_ratio": float(keep_ratio)}))

    fixed_topk = space.get("fixed_topk", {})
    for topk in fixed_topk.get("topk", [256]):
        selectors.append(_attach_variant(FixedTopKSelector(topk=int(topk)), f"fixed_topk__topk_{int(topk)}", {"topk": int(topk)}))

    vertical_only = space.get("vertical_only", {})
    for keep_columns in vertical_only.get("keep_columns", [256]):
        selectors.append(_attach_variant(VerticalOnlySelector(keep_columns=int(keep_columns)), f"vertical_only__cols_{int(keep_columns)}", {"keep_columns": int(keep_columns)}))

    slash_only = space.get("slash_only", {})
    for offsets in slash_only.get("offsets_sets", [[0, 128, 256, 512]]):
        normalized = tuple(int(x) for x in offsets)
        selectors.append(_attach_variant(SlashOnlySelector(offsets=normalized), f"slash_only__offsets_{'-'.join(str(x) for x in normalized)}", {"offsets": list(normalized)}))

    vertical_slash = space.get("vertical_slash", {})
    for keep_columns in vertical_slash.get("keep_columns", [128]):
        for offsets in vertical_slash.get("offsets_sets", [[0, 64, 128, 256]]):
            normalized = tuple(int(x) for x in offsets)
            selectors.append(_attach_variant(VerticalSlashSelector(keep_columns=int(keep_columns), offsets=normalized), f"vertical_slash__cols_{int(keep_columns)}__offsets_{'-'.join(str(x) for x in normalized)}", {"keep_columns": int(keep_columns), "offsets": list(normalized)}))

    qafb = space.get("query_aware_full_block", {})
    for block_size in qafb.get("block_size", [128]):
        for blocks_per_query_block in qafb.get("blocks_per_query_block", [4]):
            selectors.append(_attach_variant(QueryAwareFullBlockSelector(block_size=int(block_size), blocks_per_query_block=int(blocks_per_query_block)), f"query_aware_full_block__bs_{int(block_size)}__k_{int(blocks_per_query_block)}", {"block_size": int(block_size), "blocks_per_query_block": int(blocks_per_query_block)}))

    for v in space.get("adaptive_fraction_window", {}).get("variants", []):
        frac = float(v["fraction"])
        lo = int(v["min_w"])
        hi = int(v["max_w"])
        tag_frac = str(frac).replace(".", "p")
        selectors.append(_attach_variant(AdaptiveFractionWindowSelector(frac, lo, hi), f"adaptive_frac_w__f{tag_frac}__lo{lo}__hi{hi}", {"fraction": frac, "min_w": lo, "max_w": hi}))

    for v in space.get("sqrt_window", {}).get("variants", []):
        coeff = float(v["coeff"])
        lo = int(v["min_w"])
        hi = int(v["max_w"])
        tag_c = str(coeff).replace(".", "p")
        selectors.append(_attach_variant(SqrtWindowSelector(coeff, lo, hi), f"sqrt_window__c{tag_c}__lo{lo}__hi{hi}", {"coeff": coeff, "min_w": lo, "max_w": hi}))

    for v in space.get("target_density_window", {}).get("variants", []):
        target = float(v["target"])
        lo = int(v["min_w"])
        hi = int(v["max_w"])
        tag_t = str(target).replace(".", "p")
        selectors.append(_attach_variant(TargetDensityWindowSelector(target, lo, hi), f"target_density_window__t{tag_t}__lo{lo}__hi{hi}", {"target": target, "min_w": lo, "max_w": hi}))

    for v in space.get("progressive_sqrt_topk", {}).get("variants", []):
        coeff = float(v["coeff"])
        lo_r = float(v["lo"])
        hi_r = float(v["hi"])
        tag_c = str(coeff).replace(".", "p")
        selectors.append(_attach_variant(ProgressiveSqrtTopKSelector(coeff, lo_r, hi_r), f"prog_sqrt_topk__c{tag_c}__lo{lo_r}__hi{hi_r}".replace(".", "p"), {"coeff": coeff, "lo": lo_r, "hi": hi_r}))

    for key in space.get("length_tier_routers", []):
        if key == "alpha":
            selectors.append(_attach_variant(TierRouterAlphaSelector(), "length_tier__alpha", {"tier": "alpha"}))
        elif key == "beta":
            selectors.append(_attach_variant(TierRouterBetaSelector(), "length_tier__beta", {"tier": "beta"}))
        elif key == "gamma":
            selectors.append(_attach_variant(TierRouterGammaSelector(), "length_tier__gamma", {"tier": "gamma"}))

    if "length_based_hybrid" in space:
        selectors.append(_attach_variant(LengthBasedHybridSelector(), "length_based_hybrid", {}))

    return _dedupe_equivalent_selectors(selectors, eval_lengths)


def build_selectors(config: dict):
    selectors = [DenseSelector()]
    sel_cfg = config.get("selectors", {})
    selectors.append(FixedWindowSelector(window_size=sel_cfg.get("fixed_window", 512)))
    selectors.append(TopKScoreSelector(keep_ratio=sel_cfg.get("topk_keep_ratio", 0.2)))
    selectors.append(FixedTopKSelector(topk=sel_cfg.get("fixed_topk", 256)))
    selectors.append(VerticalOnlySelector(keep_columns=sel_cfg.get("vertical_keep_columns", 256)))
    selectors.append(SlashOnlySelector(offsets=tuple(sel_cfg.get("slash_offsets", [0, 128, 256, 512]))))
    selectors.append(VerticalSlashSelector(keep_columns=sel_cfg.get("vs_keep_columns", 128), offsets=tuple(sel_cfg.get("vs_offsets", [0, 64, 128, 256]))))
    selectors.append(QueryAwareFullBlockSelector(block_size=sel_cfg.get("query_aware_block_size", 128), blocks_per_query_block=sel_cfg.get("query_aware_blocks_per_query_block", 4)))
    selectors.append(LengthBasedHybridSelector())
    return selectors
