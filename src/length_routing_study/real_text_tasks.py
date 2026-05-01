from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import math
import re
from typing import Any

import pyarrow.ipc as ipc


DEFAULT_LONGBENCH_ARROW = Path(
    "/root/.cache/huggingface/datasets/THUDM___long_bench-v2/default/0.0.0/"
    "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9/long_bench-v2-train.arrow"
)
DEFAULT_SYSTEM_PROMPT = (
    "You are a careful long-context reasoning assistant. Read the provided context, "
    "identify the relevant evidence, and answer the multiple-choice question. "
    "Return exactly one capital letter."
)


@dataclass(frozen=True)
class CategorySpec:
    name: str
    domain: str
    description: str


CATEGORY_SPECS: dict[str, CategorySpec] = {
    "single_doc_qa": CategorySpec(
        name="single_doc_qa",
        domain="Single-Document QA",
        description="Public long-document QA prompts from LongBench-v2.",
    ),
    "multi_doc_qa": CategorySpec(
        name="multi_doc_qa",
        domain="Multi-Document QA",
        description="Public multi-document / retrieval-style QA prompts from LongBench-v2.",
    ),
    "code_repo_qa": CategorySpec(
        name="code_repo_qa",
        domain="Code Repository Understanding",
        description="Public code repository understanding prompts from LongBench-v2.",
    ),
}


_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")


@dataclass(frozen=True)
class PreparedExample:
    category: str
    sample_index: int
    sample_tag: str
    cache_key: str
    prompt_family: str
    source_id: str
    domain: str
    sub_domain: str
    difficulty: str
    target_len: int
    actual_len: int
    answer_letter: str
    answer_text: str
    question: str
    input_ids: list[int]
    prompt_text: str
    evidence_block_index: int
    evidence_block_indices: list[int]
    evidence_preview: str
    context_start_token: int
    context_end_token: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "sample_index": self.sample_index,
            "sample_tag": self.sample_tag,
            "cache_key": self.cache_key,
            "prompt_family": self.prompt_family,
            "source_id": self.source_id,
            "domain": self.domain,
            "sub_domain": self.sub_domain,
            "difficulty": self.difficulty,
            "target_len": self.target_len,
            "actual_len": self.actual_len,
            "answer_letter": self.answer_letter,
            "answer_text": self.answer_text,
            "question": self.question,
            "input_ids": self.input_ids,
            "prompt_text": self.prompt_text,
            "evidence_block_index": self.evidence_block_index,
            "evidence_block_indices": self.evidence_block_indices,
            "evidence_preview": self.evidence_preview,
            "context_start_token": self.context_start_token,
            "context_end_token": self.context_end_token,
        }


def _normalize_words(text: str) -> list[str]:
    return [tok.lower() for tok in _WORD_RE.findall(text or "")]


def _lexical_overlap_score(block_text: str, query_terms: set[str]) -> int:
    if not query_terms:
        return 0
    block_terms = set(_normalize_words(block_text))
    return len(block_terms & query_terms)


def gold_choice_text(row: dict[str, Any]) -> tuple[str, str]:
    letter = str(row["answer"]).strip().upper()[:1]
    text = str(row.get(f"choice_{letter}", "")).strip()
    if letter not in {"A", "B", "C", "D"} or not text:
        raise ValueError(f"invalid multiple-choice row: answer={row.get('answer')!r}")
    return letter, text


def build_prompt_fragments(row: dict[str, Any]) -> tuple[str, str, str]:
    choices = []
    for letter in ("A", "B", "C", "D"):
        value = str(row.get(f"choice_{letter}", "")).strip()
        if value:
            choices.append(f"{letter}. {value}")
    prefix = (
        "Carefully read the long context and answer the multiple-choice question.\n"
        "Use the context rather than world knowledge.\n\n"
        f"[Task Domain]\n{row['domain']} / {row['sub_domain']}\n\n"
        "[Long Context]\n"
    )
    suffix = (
        "\n\n[Question]\n"
        f"{str(row['question']).strip()}\n\n"
        "[Choices]\n"
        + "\n".join(choices)
        + "\n\nRespond with exactly one capital letter: A, B, C, or D.\nAnswer: "
    )
    answer_letter, answer_text = gold_choice_text(row)
    query_text = f"{row['question']} {answer_text}"
    return prefix, suffix, query_text


def _render_chat_ids(
    tokenizer: Any,
    *,
    user_content: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    add_generation_prompt: bool = True,
) -> list[int]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    return list(rendered)


@lru_cache(maxsize=4)
def load_longbench_rows(arrow_path: str = str(DEFAULT_LONGBENCH_ARROW)) -> list[dict[str, Any]]:
    with ipc.open_stream(arrow_path) as reader:
        return reader.read_all().to_pylist()


def pick_representative_rows(
    tokenizer: Any,
    categories: list[str],
    max_target_len: int,
    arrow_path: str = str(DEFAULT_LONGBENCH_ARROW),
    *,
    samples_per_category: int = 2,
    min_available_context_tokens: int = 4096,
    shortlist_size: int = 80,
) -> dict[str, list[dict[str, Any]]]:
    rows = load_longbench_rows(arrow_path)
    selected: dict[str, list[dict[str, Any]]] = {}
    for category in categories:
        if category not in CATEGORY_SPECS:
            raise KeyError(f"unknown category {category!r}")
        spec = CATEGORY_SPECS[category]
        domain_rows = [row for row in rows if row.get("domain") == spec.domain]
        domain_rows.sort(key=lambda item: len(str(item.get("context", ""))), reverse=True)
        first_pass: list[dict[str, Any]] = []
        fallback: list[dict[str, Any]] = []
        seen_subdomains: set[str] = set()
        for row in domain_rows[:shortlist_size]:
            prefix, suffix, _ = build_prompt_fragments(row)
            base_ids = _render_chat_ids(tokenizer, user_content=prefix + suffix)
            budget = max_target_len - len(base_ids)
            if budget < min_available_context_tokens:
                continue
            context_len = len(tokenizer.encode(str(row["context"]), add_special_tokens=False))
            if context_len < min_available_context_tokens:
                continue
            if context_len >= budget and str(row.get("sub_domain", "")) not in seen_subdomains:
                seen_subdomains.add(str(row.get("sub_domain", "")))
                first_pass.append(row)
            else:
                fallback.append(row)
        chosen = list(first_pass[:samples_per_category])
        if len(chosen) < samples_per_category:
            used_ids = {str(item["_id"]) for item in chosen}
            for row in fallback:
                if str(row["_id"]) in used_ids:
                    continue
                chosen.append(row)
                used_ids.add(str(row["_id"]))
                if len(chosen) >= samples_per_category:
                    break
        if len(chosen) < samples_per_category:
            raise RuntimeError(
                f"could not find {samples_per_category} usable rows for category={category}"
            )
        selected[category] = chosen
    return selected


def _select_context_slice(
    tokenizer: Any,
    context_ids: list[int],
    query_text: str,
    available_tokens: int,
    *,
    score_window_tokens: int = 256,
    stride_tokens: int = 128,
) -> tuple[list[int], int, str]:
    if len(context_ids) <= available_tokens:
        preview = tokenizer.decode(
            context_ids[: min(len(context_ids), score_window_tokens)],
            skip_special_tokens=True,
        )
        return context_ids, 0, preview

    query_terms = set(_normalize_words(query_text))
    best_score = -1
    best_start = 0
    last_start = max(0, len(context_ids) - score_window_tokens)
    start = 0
    while start <= last_start:
        block_ids = context_ids[start : start + score_window_tokens]
        block_text = tokenizer.decode(block_ids, skip_special_tokens=True)
        score = _lexical_overlap_score(block_text, query_terms)
        if score > best_score:
            best_score = score
            best_start = start
        start += stride_tokens
    center = best_start + min(score_window_tokens, len(context_ids) - best_start) // 2
    slice_start = max(0, min(center - available_tokens // 2, len(context_ids) - available_tokens))
    slice_ids = context_ids[slice_start : slice_start + available_tokens]
    evidence_preview = tokenizer.decode(
        context_ids[best_start : best_start + score_window_tokens],
        skip_special_tokens=True,
    )
    evidence_relative = max(0, best_start - slice_start)
    return slice_ids, evidence_relative, evidence_preview


def _locate_context_span(
    tokenizer: Any,
    *,
    user_prefix: str,
    user_suffix: str,
    context_text: str,
    system_prompt: str,
) -> tuple[int, int]:
    del user_suffix  # suffix does not affect the token span before or inside the context.
    context_start = len(
        _render_chat_ids(
            tokenizer,
            user_content=user_prefix,
            system_prompt=system_prompt,
            add_generation_prompt=False,
        )
    )
    context_end = len(
        _render_chat_ids(
            tokenizer,
            user_content=user_prefix + context_text,
            system_prompt=system_prompt,
            add_generation_prompt=False,
        )
    )
    if context_end <= context_start:
        raise RuntimeError("failed to locate context span inside rendered chat prompt")
    return context_start, context_end


def _top_evidence_block_indices(
    tokenizer: Any,
    *,
    context_ids: list[int],
    query_text: str,
    block_size: int,
    top_k: int = 3,
) -> list[int]:
    query_terms = set(_normalize_words(query_text))
    if not context_ids:
        return [0]
    scored: list[tuple[int, int]] = []
    num_blocks = math.ceil(len(context_ids) / block_size)
    for block_idx in range(num_blocks):
        block_ids = context_ids[block_idx * block_size : (block_idx + 1) * block_size]
        block_text = tokenizer.decode(block_ids, skip_special_tokens=True)
        score = _lexical_overlap_score(block_text, query_terms)
        scored.append((score, block_idx))
    scored.sort(key=lambda item: (-item[0], item[1]))
    winners = [idx for score, idx in scored if score > 0][:top_k]
    return winners or [scored[0][1]]


def prepare_example(
    tokenizer: Any,
    row: dict[str, Any],
    *,
    category: str,
    sample_index: int,
    target_len: int,
    prompt_family: str | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    block_size: int = 128,
) -> PreparedExample:
    if category not in CATEGORY_SPECS:
        raise KeyError(f"unknown category {category!r}")
    user_prefix, user_suffix, query_text = build_prompt_fragments(row)
    answer_letter, answer_text = gold_choice_text(row)

    base_ids = _render_chat_ids(
        tokenizer,
        user_content=user_prefix + user_suffix,
        system_prompt=system_prompt,
    )
    context_ids = tokenizer.encode(str(row["context"]), add_special_tokens=False)
    available_context_tokens = target_len - len(base_ids)
    if available_context_tokens <= block_size:
        raise ValueError(
            f"target_len={target_len} leaves only {available_context_tokens} context tokens"
        )
    chosen_context_ids, evidence_relative, evidence_preview = _select_context_slice(
        tokenizer,
        context_ids,
        query_text,
        available_context_tokens,
    )

    while True:
        context_text = tokenizer.decode(chosen_context_ids, skip_special_tokens=True)
        input_ids = _render_chat_ids(
            tokenizer,
            user_content=user_prefix + context_text + user_suffix,
            system_prompt=system_prompt,
        )
        if len(input_ids) <= target_len:
            break
        overflow = len(input_ids) - target_len
        if overflow >= len(chosen_context_ids):
            raise RuntimeError(f"could not fit prompt into target_len={target_len}")
        chosen_context_ids = chosen_context_ids[:-overflow]

    context_start_token, context_end_token = _locate_context_span(
        tokenizer,
        user_prefix=user_prefix,
        user_suffix=user_suffix,
        context_text=context_text,
        system_prompt=system_prompt,
    )
    context_block_indices = _top_evidence_block_indices(
        tokenizer,
        context_ids=chosen_context_ids,
        query_text=query_text,
        block_size=block_size,
        top_k=3,
    )
    evidence_block_indices = [
        (context_start_token + block_idx * block_size) // block_size
        for block_idx in context_block_indices
    ]
    evidence_block_index = evidence_block_indices[0]

    prompt_family = prompt_family or f"real_{category}"
    sample_tag = f"s{sample_index:02d}"
    cache_key = f"{prompt_family}_{sample_tag}"
    primary_block_offset = max(0, context_block_indices[0] * block_size - evidence_relative)
    preview_slice = chosen_context_ids[
        primary_block_offset : primary_block_offset + min(256, len(chosen_context_ids))
    ]
    if preview_slice:
        evidence_preview = tokenizer.decode(preview_slice, skip_special_tokens=True)

    return PreparedExample(
        category=category,
        sample_index=sample_index,
        sample_tag=sample_tag,
        cache_key=cache_key,
        prompt_family=prompt_family,
        source_id=str(row["_id"]),
        domain=str(row["domain"]),
        sub_domain=str(row["sub_domain"]),
        difficulty=str(row.get("difficulty", "")),
        target_len=target_len,
        actual_len=len(input_ids),
        answer_letter=answer_letter,
        answer_text=answer_text,
        question=str(row["question"]).strip(),
        input_ids=list(input_ids),
        prompt_text=tokenizer.decode(input_ids, skip_special_tokens=False),
        evidence_block_index=evidence_block_index,
        evidence_block_indices=evidence_block_indices,
        evidence_preview=evidence_preview[:600],
        context_start_token=context_start_token,
        context_end_token=context_end_token,
    )


def choice_token_map(tokenizer: Any) -> dict[str, int]:
    token_map: dict[str, int] = {}
    for letter in ("A", "B", "C", "D"):
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"choice {letter!r} is not a single token for this tokenizer")
        token_map[letter] = ids[0]
    return token_map
