#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from length_routing_study.real_text_tasks import (
    CATEGORY_SPECS,
    DEFAULT_LONGBENCH_ARROW,
    pick_representative_rows,
    prepare_example,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate model-derived QKV caches from public real-text prompts."
    )
    parser.add_argument(
        "--model",
        default="/root/autodl-tmp/qwen/Qwen2.5-3B-Instruct",
        help="Local HuggingFace model path.",
    )
    parser.add_argument(
        "--arrow-path",
        default=str(DEFAULT_LONGBENCH_ARROW),
        help="Cached LongBench-v2 Arrow file.",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(CATEGORY_SPECS.keys()),
        choices=sorted(CATEGORY_SPECS.keys()),
    )
    parser.add_argument(
        "--target-lengths",
        nargs="+",
        type=int,
        default=[4096, 8192, 16384],
        help="Prompt lengths to materialize from each category.",
    )
    parser.add_argument("--samples-per-category", type=int, default=1)
    parser.add_argument("--layer-index", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("data/cache_real"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        choices=["auto", "eager", "sdpa", "flash_attention_2"],
    )
    return parser.parse_args()


def _normalize_layer_index(layer_index: int, num_layers: int) -> int:
    if layer_index < 0:
        layer_index += num_layers
    if layer_index < 0 or layer_index >= num_layers:
        raise IndexError(f"layer_index={layer_index} out of range for {num_layers} layers")
    return layer_index


def _make_hook(name: str, storage: dict[str, torch.Tensor]):
    def _hook(_module, _inputs, output):
        storage[name] = output.detach()
    return _hook


def _load_model(args: argparse.Namespace, device: torch.device) -> AutoModelForCausalLM:
    kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": {"": 0} if device.type == "cuda" else None,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if args.attn_implementation != "auto":
        kwargs["attn_implementation"] = args.attn_implementation
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    except Exception:
        kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    if device.type != "cuda":
        model.to(device)
    model.eval()
    return model


def _capture_qkv(
    model: AutoModelForCausalLM,
    *,
    input_ids: list[int],
    layer_index: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    layer = model.model.layers[layer_index].self_attn
    captured: dict[str, torch.Tensor] = {}
    hooks = [
        layer.q_proj.register_forward_hook(_make_hook("q", captured)),
        layer.k_proj.register_forward_hook(_make_hook("k", captured)),
        layer.v_proj.register_forward_hook(_make_hook("v", captured)),
    ]
    try:
        input_tensor = torch.tensor([input_ids], device=device, dtype=torch.long)
        with torch.inference_mode():
            model(input_ids=input_tensor, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
    q = captured["q"].view(1, -1, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
    k = captured["k"].view(1, -1, num_kv_heads, head_dim).permute(0, 2, 1, 3).contiguous()
    v = captured["v"].view(1, -1, num_kv_heads, head_dim).permute(0, 2, 1, 3).contiguous()
    if num_kv_heads != num_heads:
        repeat_factor = num_heads // num_kv_heads
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)
    return q, k, v


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = _load_model(args, device)

    num_heads = int(model.config.num_attention_heads)
    num_kv_heads = int(getattr(model.config, "num_key_value_heads", num_heads))
    head_dim = int(model.config.hidden_size // num_heads)
    layer_index = _normalize_layer_index(args.layer_index, int(model.config.num_hidden_layers))

    chosen_rows = pick_representative_rows(
        tokenizer,
        categories=args.categories,
        max_target_len=max(args.target_lengths),
        arrow_path=args.arrow_path,
        samples_per_category=args.samples_per_category,
    )

    manifest: list[dict[str, object]] = []
    for category in args.categories:
        rows = chosen_rows[category]
        for sample_index, row in enumerate(rows):
            print(
                f"[select] {category:<16} sample={sample_index:<2} id={row['_id']}  "
                f"domain={row['domain']} / {row['sub_domain']} difficulty={row.get('difficulty', '')}"
            )
            for target_len in args.target_lengths:
                prepared = prepare_example(
                    tokenizer,
                    row,
                    category=category,
                    sample_index=sample_index,
                    target_len=target_len,
                    prompt_family=f"real_{category}",
                )
                out_name = f"qkv_len_{prepared.actual_len}_{prepared.cache_key}_seed_{args.seed}.pt"
                out_path = args.output_dir / out_name
                if out_path.exists() and not args.overwrite:
                    print(f"  skip {out_name}")
                    manifest.append(prepared.as_dict() | {"cache_file": out_name, "layer_index": layer_index})
                    continue

                print(
                    f"  build L={prepared.actual_len:<7} category={category:<16} "
                    f"sample={prepared.sample_tag} file={out_name}"
                )
                q, k, v = _capture_qkv(
                    model,
                    input_ids=prepared.input_ids,
                    layer_index=layer_index,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    device=device,
                )
                payload = {
                    "q": q.to(dtype=torch.bfloat16).cpu(),
                    "k": k.to(dtype=torch.bfloat16).cpu(),
                    "v": v.to(dtype=torch.bfloat16).cpu(),
                    "meta": {
                        "category": category,
                        "sample_index": prepared.sample_index,
                        "sample_tag": prepared.sample_tag,
                        "source_id": prepared.source_id,
                        "domain": prepared.domain,
                        "sub_domain": prepared.sub_domain,
                        "target_len": prepared.target_len,
                        "actual_len": prepared.actual_len,
                        "answer_letter": prepared.answer_letter,
                        "layer_index": layer_index,
                    },
                }
                torch.save(payload, out_path)
                manifest.append(prepared.as_dict() | {"cache_file": out_name, "layer_index": layer_index})
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    manifest_path = args.output_dir / "real_text_manifest.json"
    manifest_obj = {
        "model": args.model,
        "seed": args.seed,
        "layer_index": layer_index,
        "categories": args.categories,
        "target_lengths": args.target_lengths,
        "samples_per_category": args.samples_per_category,
        "entries": manifest,
    }
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(manifest_obj, fp, indent=2, ensure_ascii=False)
    print(f"saved manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
