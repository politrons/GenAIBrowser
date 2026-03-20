from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, pipeline

from .mcp_web import search_web_via_mcp
from .rag_retriever import TfidfRagRetriever

try:
    import torch
except Exception:  # pragma: no cover - optional at static analysis time
    torch = None

DEFAULT_PROMPT_ARTIFACT: dict[str, Any] = {
    "instructions": (
        "You are a web research assistant. "
        "Answer only from retrieved web snippets and knowledge snippets. "
        "If evidence is missing, say it clearly and suggest a better search query."
    ),
    "default_context": (
        "Focus on practical web research tasks: find facts, compare sources, summarize with citations, and stay concise."
    ),
    "few_shot_demos": [
        {
            "question": "What are the most recent announcements about OpenAI API models?",
            "answer": "Summarize key points and cite the source URLs used.",
        },
        {
            "question": "Compare two frameworks for browser automation.",
            "answer": "Give a short comparison and cite source links for each claim.",
        },
    ],
}

DEFAULT_SYSTEM_POLICY = (
    "You are a Web Search Assistant with strict grounding rules.\n"
    "Rules:\n"
    "- Use only retrieved evidence from web pages and KB snippets.\n"
    "- Never invent facts, dates, prices, versions, or URLs.\n"
    "- Prioritize recent evidence when user asks latest/recent/current.\n"
    "- If evidence is insufficient, say it explicitly and propose one improved query.\n"
    "Output style:\n"
    "- First line: direct answer.\n"
    "- Then: short evidence bullets with title and URL.\n"
    "- Keep responses concise."
)

QUERY_PLANNER_PROMPT = (
    "You are a JSON API for web retrieval planning.\n"
    "Return exactly one valid JSON object and nothing else.\n"
    "Required keys and types:\n"
    "{\n"
    '  "intent": "web_search",\n'
    '  "search_query": "",\n'
    '  "topic_terms": [],\n'
    '  "must_match_all_terms": true,\n'
    '  "request_latest": false,\n'
    '  "need_fresh_search": true,\n'
    '  "max_results": 6,\n'
    '  "language": "en"\n'
    "}\n"
    "Rules:\n"
    "- intent must be one of: web_search, summarize_existing, open_qa.\n"
    "- Use intent=web_search for most user questions that require internet evidence.\n"
    "- Use intent=summarize_existing only for explicit follow-up requests over already retrieved evidence.\n"
    "- search_query must be specific and ready for a search tool.\n"
    "- topic_terms must include meaningful keywords.\n"
    "- max_results must be an integer in [3, 12].\n"
    "- need_fresh_search should be true unless user clearly asks to summarize previous context only.\n"
    "- Output must start with '{' and end with '}'.\n"
    "Example valid output:\n"
    '{"intent":"web_search","search_query":"latest rust release notes","topic_terms":["rust","release","notes"],"must_match_all_terms":true,"request_latest":true,"need_fresh_search":true,"max_results":6,"language":"en"}\n'
    "- Do not output markdown, comments, or extra text."
)

TASK_ALIASES = {
    "text2text-generation": "text-generation",
}

DEFAULT_MCP_SERVER_COMMAND = "npx -y firecrawl-mcp"


def _load_env_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            os.environ.setdefault(key, value)


def _env_value(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value is None:
            continue
        cleaned = value.strip()
        if cleaned:
            return cleaned
    return default


def parse_args() -> argparse.Namespace:
    _load_env_file(".env")

    parser = argparse.ArgumentParser(description="Chat with a web browser assistant using MCP + local LLM + local RAG")
    parser.add_argument("--context", default="")
    parser.add_argument("--web-chunks", default=os.getenv("WEB_CHUNKS_PATH", "data/web_chunks.jsonl"))
    parser.add_argument(
        "--knowledge-chunks",
        default=os.getenv("RAG_CHUNKS_PATH", "data/knowledge_base_chunks.jsonl"),
        help="Optional KB chunks JSONL. Use empty value to disable.",
    )
    parser.add_argument(
        "--prompt-artifact",
        default=os.getenv("DSPY_OPTIMIZED_PROMPT_PATH", ""),
        help="Optional path to optimized_prompt.json. If omitted, auto-discovery is used.",
    )
    parser.add_argument(
        "--system-prompt-file",
        default=os.getenv("LOCAL_SYSTEM_PROMPT_FILE", "config/browser_llm_system_prompt.txt"),
        help="Path to system prompt policy.",
    )
    parser.add_argument("--hf-model-id", default=os.getenv("LOCAL_HF_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct"))
    parser.add_argument("--hf-task", default=os.getenv("LOCAL_HF_TASK", "text-generation"))
    parser.add_argument("--max-new-tokens", type=int, default=int(os.getenv("LOCAL_MAX_NEW_TOKENS", "260")))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("LOCAL_TEMPERATURE", "0.0")))
    parser.add_argument("--rag-top-k", type=int, default=int(os.getenv("LOCAL_RAG_TOP_K", "6")))
    parser.add_argument("--rag-min-score", type=float, default=float(os.getenv("LOCAL_RAG_MIN_SCORE", "0.01")))
    parser.add_argument("--max-snippet-chars", type=int, default=int(os.getenv("LOCAL_MAX_SNIPPET_CHARS", "420")))

    parser.add_argument(
        "--mcp-server-command",
        default=_env_value("MCP_SERVER_COMMAND", "mcp_server_command", default=DEFAULT_MCP_SERVER_COMMAND),
        help='Command used to launch MCP server, e.g. "npx -y <server>"',
    )
    parser.add_argument(
        "--mcp-search-tool",
        default=_env_value("MCP_WEB_SEARCH_TOOL", "mcp_web_search_tool", default=""),
        help="MCP tool name for web search. If empty, runtime will try auto-detection.",
    )
    parser.add_argument(
        "--mcp-fetch-tool",
        default=_env_value("MCP_WEB_FETCH_TOOL", "mcp_web_fetch_tool", default=""),
        help="MCP tool name for page fetch/extract. If empty, runtime will try auto-detection.",
    )
    parser.add_argument(
        "--mcp-account",
        default=_env_value("MCP_ACCOUNT", "mcp_account", default=""),
        help="Optional account id/name for MCP servers that require explicit account parameter",
    )
    parser.add_argument(
        "--mcp-startup-timeout",
        type=int,
        default=int(_env_value("MCP_STARTUP_TIMEOUT", "mcp_startup_timeout", default="20")),
        help="Seconds to wait for MCP server startup and initialize",
    )
    parser.add_argument(
        "--mcp-request-timeout",
        type=int,
        default=int(_env_value("MCP_REQUEST_TIMEOUT", "mcp_request_timeout", default="45")),
        help="Seconds to wait for each MCP request",
    )

    parser.add_argument(
        "--search-max-results",
        type=int,
        default=int(_env_value("MCP_WEB_MAX_RESULTS", "mcp_web_max_results", default="8")),
        help="Maximum search results requested per turn.",
    )
    parser.add_argument(
        "--fetch-top-pages",
        type=int,
        default=int(_env_value("MCP_WEB_FETCH_TOP_PAGES", "mcp_web_fetch_top_pages", default="4")),
        help="How many top search results should be fetched for page content.",
    )
    parser.add_argument(
        "--fetch-max-chars",
        type=int,
        default=int(_env_value("MCP_WEB_MAX_PAGE_CHARS", "mcp_web_max_page_chars", default="7000")),
        help="Maximum chars saved per fetched page.",
    )
    parser.add_argument(
        "--page-chunk-size",
        type=int,
        default=int(_env_value("WEB_PAGE_CHUNK_SIZE", "web_page_chunk_size", default="1200")),
        help="Chunk size for fetched page text stored in local index.",
    )
    parser.add_argument(
        "--page-chunk-overlap",
        type=int,
        default=int(_env_value("WEB_PAGE_CHUNK_OVERLAP", "web_page_chunk_overlap", default="160")),
        help="Chunk overlap for fetched page text stored in local index.",
    )
    parser.add_argument("--json-output", action="store_true")
    return parser.parse_args()


def _discover_latest_prompt_artifact() -> Path | None:
    candidates: list[Path] = []

    default_path = Path("artifacts/dspy_optimized/optimized_prompt.json")
    if default_path.exists():
        candidates.append(default_path)

    artifacts_dir = Path("artifacts")
    if artifacts_dir.exists():
        for item in artifacts_dir.rglob("optimized_prompt.json"):
            if item.is_file():
                candidates.append(item)

    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _resolve_prompt_artifact_path(raw_path: str | None) -> Path | None:
    candidate = (raw_path or "").strip()
    if candidate:
        path = Path(candidate)
        if path.exists():
            return path
        print(f"[warn] Prompt artifact not found at {candidate}. Falling back to auto-discovery/default prompt.")

    return _discover_latest_prompt_artifact()


def _load_prompt_artifact(path: str | Path | None) -> tuple[dict[str, Any], str]:
    if path is None:
        return dict(DEFAULT_PROMPT_ARTIFACT), "built-in-default"

    artifact_path = Path(path)
    if not artifact_path.exists():
        return dict(DEFAULT_PROMPT_ARTIFACT), "built-in-default"

    try:
        data = json.loads(artifact_path.read_text(encoding="utf-8"))
    except Exception:
        return dict(DEFAULT_PROMPT_ARTIFACT), "built-in-default"

    if not isinstance(data, dict):
        return dict(DEFAULT_PROMPT_ARTIFACT), "built-in-default"

    merged = dict(DEFAULT_PROMPT_ARTIFACT)
    merged.update(data)
    return merged, str(artifact_path)


def _load_system_policy(path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.exists():
        return DEFAULT_SYSTEM_POLICY

    text = candidate.read_text(encoding="utf-8").strip()
    return text if text else DEFAULT_SYSTEM_POLICY


def _load_rows(path: str | Path) -> list[dict[str, Any]]:
    path_obj = Path(path)
    if not path_obj.exists():
        return []

    rows: list[dict[str, Any]] = []
    for line in path_obj.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _write_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _parse_iso_date(value: str) -> datetime:
    text = (value or "").strip()
    if not text:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _chunk_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    source = str(text or "").strip()
    if not source:
        return []

    if len(source) <= max_chars:
        return [source]

    chunks: list[str] = []
    start = 0
    while start < len(source):
        end = min(start + max_chars, len(source))
        chunk = source[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(source):
            break
        start = max(0, end - overlap_chars)

    return chunks


def _hash_key(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:12]


def _to_url_domain(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc.strip().lower()


def _rows_from_search_summary(summary: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    query = str(summary.get("query", "")).strip()
    timestamp = datetime.now(timezone.utc).isoformat()

    search_results = summary.get("search_results")
    if not isinstance(search_results, list):
        search_results = []

    fetched_pages = summary.get("fetched_pages")
    if not isinstance(fetched_pages, list):
        fetched_pages = []

    fetched_by_url: dict[str, dict[str, Any]] = {}
    for item in fetched_pages:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url", "")).strip()
        if url:
            fetched_by_url[url] = item

    rows: list[dict[str, Any]] = []

    for result in search_results:
        if not isinstance(result, dict):
            continue
        url = str(result.get("url", "")).strip()
        if not url:
            continue

        rank = int(result.get("rank", 0) or 0)
        title = str(result.get("title", "")).strip()
        snippet = str(result.get("snippet", "")).strip()
        published_at = str(result.get("published_at", "")).strip()
        domain = _to_url_domain(url)
        key = f"search::{url}"

        rows.append(
            {
                "chunk_id": f"web-search-{_hash_key(key)}",
                "source": "mcp-web:search",
                "text": "\n".join(
                    [
                        f"Title: {title}",
                        f"URL: {url}",
                        f"Snippet: {snippet}",
                    ]
                ).strip(),
                "metadata": {
                    "kind": "search_result",
                    "url": url,
                    "title": title,
                    "domain": domain,
                    "rank": str(rank),
                    "published_at": published_at,
                    "query": query,
                    "retrieved_at": timestamp,
                },
            }
        )

        page = fetched_by_url.get(url)
        if not isinstance(page, dict):
            continue

        content = str(page.get("content", "")).strip()
        if not content:
            continue

        chunks = _chunk_text(content, max_chars=args.page_chunk_size, overlap_chars=args.page_chunk_overlap)
        total_chunks = len(chunks)
        for idx, chunk in enumerate(chunks, start=1):
            content_key = f"page::{url}::{idx}"
            rows.append(
                {
                    "chunk_id": f"web-page-{_hash_key(content_key)}",
                    "source": "mcp-web:page",
                    "text": "\n".join(
                        [
                            f"Title: {title}",
                            f"URL: {url}",
                            "Page excerpt:",
                            chunk,
                        ]
                    ).strip(),
                    "metadata": {
                        "kind": "web_page",
                        "url": url,
                        "title": title,
                        "domain": domain,
                        "rank": str(rank),
                        "published_at": published_at,
                        "query": query,
                        "retrieved_at": timestamp,
                        "chunk_index": str(idx),
                        "chunk_total": str(total_chunks),
                    },
                }
            )

    return rows


def _merge_rows(existing: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in existing + incoming:
        if not isinstance(row, dict):
            continue
        key = str(row.get("chunk_id", "")).strip()
        if not key:
            continue
        merged[key] = row

    out = list(merged.values())
    out.sort(
        key=lambda row: _parse_iso_date(
            str(((row.get("metadata") if isinstance(row.get("metadata"), dict) else {}) or {}).get("retrieved_at", ""))
        ),
        reverse=True,
    )
    return out


def _build_retriever_paths(args: argparse.Namespace) -> list[str]:
    paths: list[str] = []

    web_chunks_path = Path(args.web_chunks)
    if web_chunks_path.exists() and web_chunks_path.stat().st_size > 0:
        paths.append(str(web_chunks_path))

    knowledge_chunks = str(args.knowledge_chunks or "").strip()
    if knowledge_chunks:
        kb_path = Path(knowledge_chunks)
        if kb_path.exists() and kb_path.stat().st_size > 0:
            paths.append(str(kb_path))

    return paths


def _load_retriever_and_rows(args: argparse.Namespace) -> tuple[TfidfRagRetriever | None, list[dict[str, Any]]]:
    paths = _build_retriever_paths(args)
    if not paths:
        return None, []

    all_rows: list[dict[str, Any]] = []
    for path in paths:
        all_rows.extend(_load_rows(path))

    if not all_rows:
        return None, []

    retriever = TfidfRagRetriever.from_jsonl_paths(paths)
    return retriever, all_rows


def _corpus_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    domains: Counter[str] = Counter()
    unique_urls: set[str] = set()

    for row in rows:
        meta = row.get("metadata")
        if not isinstance(meta, dict):
            continue
        url = str(meta.get("url", "")).strip()
        if url:
            unique_urls.add(url)
            domain = _to_url_domain(url)
            if domain:
                domains[domain] += 1

    return {
        "indexed_chunks": len(rows),
        "unique_urls": len(unique_urls),
        "top_domains": domains.most_common(5),
    }


def _extract_json_object(text: str) -> str | None:
    raw = (text or "").strip()
    if not raw:
        return None
    start = raw.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False

    for idx in range(start, len(raw)):
        ch = raw[idx]

        if escaped:
            escaped = False
            continue

        if ch == "\\":
            escaped = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start : idx + 1]

    return None


def _try_parse_json_dict(raw: str) -> dict[str, Any] | None:
    blob = _extract_json_object(raw)
    if blob is None:
        return None
    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _repair_json_with_llm(generator: Any, schema_prompt: str, invalid_output: str, max_new_tokens: int) -> list[str]:
    invalid = str(invalid_output).strip()
    prompts = [
        "\n".join(
            [
                "You are a JSON repair assistant.",
                "Rewrite the invalid output as one valid JSON object that follows the schema and rules.",
                "Return only JSON.",
                "Output must start with '{' and end with '}'.",
                "",
                "Schema and rules:",
                schema_prompt,
                "",
                "Invalid output:",
                invalid,
                "",
                "Fixed JSON:",
            ]
        ),
        "\n".join(
            [
                "Return EXACTLY one minified JSON object.",
                "No prose. No markdown. No comments.",
                "If a field is unknown, use safe defaults from the schema example.",
                "",
                "Schema and rules:",
                schema_prompt,
                "",
                "Invalid output to fix:",
                invalid,
                "",
                "JSON:",
            ]
        ),
    ]

    out: list[str] = []
    for prompt in prompts:
        out.append(
            _generate_answer(
                generator=generator,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
            )
        )
    return out


def _normalize_llm_plan(plan: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(plan, dict):
        raise ValueError("Query plan must be a JSON object")

    intent = str(plan.get("intent", "")).strip()
    if intent not in {"web_search", "summarize_existing", "open_qa"}:
        raise ValueError("Invalid or missing plan.intent")

    search_query = str(plan.get("search_query", "")).strip()

    raw_terms = plan.get("topic_terms")
    if not isinstance(raw_terms, list):
        raise ValueError("Invalid or missing plan.topic_terms")
    terms = [str(item).strip().lower() for item in raw_terms if str(item).strip()]

    must_match_all_raw = plan.get("must_match_all_terms")
    if not isinstance(must_match_all_raw, bool):
        raise ValueError("Invalid or missing plan.must_match_all_terms")

    request_latest_raw = plan.get("request_latest")
    if not isinstance(request_latest_raw, bool):
        raise ValueError("Invalid or missing plan.request_latest")

    need_fresh_search_raw = plan.get("need_fresh_search")
    if not isinstance(need_fresh_search_raw, bool):
        raise ValueError("Invalid or missing plan.need_fresh_search")

    max_results_raw = plan.get("max_results")
    try:
        max_results = int(max_results_raw)
    except Exception:
        raise ValueError("Invalid or missing plan.max_results") from None
    if max_results < 3 or max_results > 12:
        raise ValueError("plan.max_results must be in [3,12]")

    language = str(plan.get("language", "")).strip().lower()
    if not language:
        raise ValueError("Invalid or missing plan.language")

    if intent == "web_search" and not search_query:
        raise ValueError("plan.search_query must not be empty for intent=web_search")

    return {
        "intent": intent,
        "search_query": search_query,
        "topic_terms": terms,
        "must_match_all_terms": must_match_all_raw,
        "request_latest": request_latest_raw,
        "need_fresh_search": need_fresh_search_raw,
        "max_results": max_results,
        "language": language,
    }


def _plan_query_with_llm(question: str, context: str, system_policy: str, generator: Any) -> dict[str, Any]:
    prompt = "\n".join(
        [
            QUERY_PLANNER_PROMPT,
            "",
            "System policy:",
            system_policy.strip() or DEFAULT_SYSTEM_POLICY,
            "",
            f"User question: {question}",
            f"Optional context: {context}",
            "",
            "JSON plan:",
        ]
    )

    raw_plan = _generate_answer(
        generator=generator,
        prompt=prompt,
        max_new_tokens=260,
        temperature=0.0,
    )

    parsed = _try_parse_json_dict(raw_plan)
    if parsed is None:
        repaired_candidates = _repair_json_with_llm(
            generator=generator,
            schema_prompt=QUERY_PLANNER_PROMPT,
            invalid_output=raw_plan,
            max_new_tokens=220,
        )
        for repaired in repaired_candidates:
            parsed = _try_parse_json_dict(repaired)
            if parsed is not None:
                break
        if parsed is None:
            preview = str(repaired_candidates[-1]).strip().replace("\n", " ")[:300]
            raise RuntimeError(f"LLM planner did not return valid JSON after repair. Raw output: {preview}")

    try:
        return _normalize_llm_plan(parsed)
    except ValueError as exc:
        repaired_candidates = _repair_json_with_llm(
            generator=generator,
            schema_prompt=QUERY_PLANNER_PROMPT,
            invalid_output=f"{json.dumps(parsed, ensure_ascii=True)}\nSchema error: {exc}",
            max_new_tokens=220,
        )
        for repaired in repaired_candidates:
            repaired_parsed = _try_parse_json_dict(repaired)
            if repaired_parsed is None:
                continue
            try:
                return _normalize_llm_plan(repaired_parsed)
            except ValueError:
                continue
        raise RuntimeError(f"LLM planner JSON schema validation failed: {exc}") from exc


def _metadata_from_chunk(chunk: dict[str, Any]) -> dict[str, str]:
    raw_meta = chunk.get("metadata")
    if not isinstance(raw_meta, dict):
        raw_meta = {}
    return {
        "kind": str(raw_meta.get("kind", "")).strip(),
        "title": str(raw_meta.get("title", "")).strip(),
        "url": str(raw_meta.get("url", "")).strip(),
        "domain": str(raw_meta.get("domain", "")).strip(),
        "published_at": str(raw_meta.get("published_at", "")).strip(),
    }


def _format_evidence(chunk: dict[str, Any], max_snippet_chars: int) -> str:
    metadata = _metadata_from_chunk(chunk)
    kind = metadata.get("kind") or "unknown"
    title = metadata.get("title") or "(untitled)"
    url = metadata.get("url") or "(no-url)"
    published_at = metadata.get("published_at") or "unknown-date"

    text = str(chunk.get("text", "")).strip()
    if len(text) > max_snippet_chars:
        text = text[:max_snippet_chars].rstrip() + "..."

    score = float(chunk.get("score", 0.0))
    return f"score={score:.4f} | {kind} | {published_at} | {title} | {url}\n{text}"


def _build_prompt(
    system_policy: str,
    prompt_artifact: dict[str, Any],
    llm_plan: dict[str, Any],
    question: str,
    context: str,
    retrieved_chunks: list[dict[str, Any]],
    corpus_stats: dict[str, Any],
    query_hit_count: int,
    max_snippet_chars: int,
) -> str:
    instruction = str(prompt_artifact.get("instructions", DEFAULT_PROMPT_ARTIFACT["instructions"]))
    default_context = str(prompt_artifact.get("default_context", DEFAULT_PROMPT_ARTIFACT["default_context"]))
    demos = prompt_artifact.get("few_shot_demos", [])

    lines: list[str] = []
    if system_policy.strip():
        lines.append("System policy:")
        lines.append(system_policy.strip())
    if llm_plan:
        lines.append("LLM query plan:")
        lines.append(json.dumps(llm_plan, ensure_ascii=True))
    lines.append(instruction)
    if default_context:
        lines.append(f"Domain context: {default_context}")
    if context:
        lines.append(f"Request context: {context}")

    lines.append("Indexed corpus stats:")
    lines.append(f"- indexed_chunks: {corpus_stats.get('indexed_chunks', 0)}")
    lines.append(f"- unique_urls: {corpus_stats.get('unique_urls', 0)}")
    lines.append(f"- query_url_hits: {query_hit_count}")

    top_domains = corpus_stats.get("top_domains", [])
    if isinstance(top_domains, list) and top_domains:
        domains_line = ", ".join([f"{domain} ({count})" for domain, count in top_domains[:5]])
        lines.append(f"- top_domains: {domains_line}")

    if isinstance(demos, list) and demos:
        lines.append("Examples:")
        for idx, demo in enumerate(demos[:3], start=1):
            q_demo = str((demo or {}).get("question", "")).strip()
            a_demo = str((demo or {}).get("answer", "")).strip()
            if q_demo and a_demo:
                lines.append(f"{idx}. Q: {q_demo}")
                lines.append(f"   A: {a_demo}")

    lines.append("Retrieved evidence:")
    if retrieved_chunks:
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            lines.append(f"{idx}. {_format_evidence(chunk, max_snippet_chars=max_snippet_chars)}")
    else:
        lines.append("No snippets retrieved for this query.")

    lines.append("Response requirements:")
    lines.append("1) Be concise and concrete.")
    lines.append("2) Cite source URLs in the evidence bullets.")
    lines.append("3) If evidence is missing, say it explicitly and propose one better query.")

    lines.append(f"User question: {question}")
    lines.append("Answer:")
    return "\n".join(lines)


def _normalize_task(task: str) -> str:
    requested = (task or "").strip().lower()
    if not requested:
        return "text-generation"
    return TASK_ALIASES.get(requested, requested)


def _safe_model_max_input_tokens(tokenizer: Any, default_max_tokens: int = 2048) -> int:
    model_max = getattr(tokenizer, "model_max_length", None)
    if not isinstance(model_max, int) or model_max <= 0 or model_max > 100_000:
        return default_max_tokens
    return model_max


def _build_generator(task: str, model_id: str):
    if torch is None:
        raise RuntimeError("PyTorch is required for local generation. Install torch in your environment.")

    pipeline_device = -1
    model_device = "cpu"
    if torch is not None and torch.cuda.is_available():
        pipeline_device = 0
        model_device = "cuda"

    normalized_task = _normalize_task(task)
    try:
        config = AutoConfig.from_pretrained(model_id)
    except Exception:
        config = None

    if config is not None and bool(getattr(config, "is_encoder_decoder", False)):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        model.to(model_device)
        model.eval()
        return {
            "engine": "seq2seq-manual",
            "model": model,
            "tokenizer": tokenizer,
            "device": model_device,
        }

    try:
        pipe = pipeline(
            normalized_task,
            model=model_id,
            tokenizer=model_id,
            device=pipeline_device,
        )
        model_obj = getattr(pipe, "model", None)
        generation_config = getattr(model_obj, "generation_config", None)
        if generation_config is not None:
            if hasattr(generation_config, "do_sample"):
                generation_config.do_sample = False
            if hasattr(generation_config, "temperature"):
                generation_config.temperature = 1.0
            if hasattr(generation_config, "top_p"):
                generation_config.top_p = 1.0
            if hasattr(generation_config, "top_k"):
                generation_config.top_k = 50
            if hasattr(generation_config, "max_length"):
                generation_config.max_length = None
        return pipe
    except Exception as exc:
        raise RuntimeError(
            f"Could not build a local generator for task '{task}' and model '{model_id}'. "
            "Use a compatible local model/task combination."
        ) from exc


def _generate_answer(generator: Any, prompt: str, max_new_tokens: int, temperature: float) -> str:
    if isinstance(generator, dict) and str(generator.get("engine", "")).endswith("-manual"):
        if torch is None:
            raise RuntimeError("PyTorch is required for manual generation.")
        model = generator.get("model")
        tokenizer = generator.get("tokenizer")
        device = str(generator.get("device", "cpu"))
        if model is None or tokenizer is None:
            raise RuntimeError("Manual generator is missing model/tokenizer.")

        max_input_tokens = _safe_model_max_input_tokens(tokenizer)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        )
        if device == "cuda":
            inputs = {key: value.to("cuda") for key, value in inputs.items()}

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature

        with torch.no_grad():
            output_ids = model.generate(**inputs, **generation_kwargs)

        text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        return text

    generation_kwargs: dict[str, Any] = {}
    model_obj = getattr(generator, "model", None)
    base_cfg = getattr(model_obj, "generation_config", None)
    if base_cfg is not None:
        cfg = GenerationConfig.from_dict(base_cfg.to_dict())
        cfg.max_new_tokens = max_new_tokens
        cfg.max_length = None
        cfg.do_sample = temperature > 0
        if temperature > 0:
            cfg.temperature = temperature
        else:
            cfg.temperature = 1.0
            cfg.top_p = 1.0
            cfg.top_k = 50
        generation_kwargs["generation_config"] = cfg
    else:
        generation_kwargs["max_new_tokens"] = max_new_tokens
        generation_kwargs["do_sample"] = temperature > 0
        if temperature > 0:
            generation_kwargs["temperature"] = temperature

    result = generator(prompt, **generation_kwargs)

    if isinstance(result, list) and result:
        candidate = result[0]
        if isinstance(candidate, dict):
            for key in ("generated_text", "summary_text", "text"):
                if key in candidate:
                    generated = str(candidate[key]).strip()
                    if generated.startswith(prompt):
                        generated = generated[len(prompt) :].strip()
                    return generated
        generated = str(candidate).strip()
        if generated.startswith(prompt):
            generated = generated[len(prompt) :].strip()
        return generated

    generated = str(result).strip()
    if generated.startswith(prompt):
        generated = generated[len(prompt) :].strip()
    return generated


def _run_single_question(
    question: str,
    context: str,
    system_policy: str,
    prompt_artifact: dict[str, Any],
    llm_plan: dict[str, Any],
    retriever: TfidfRagRetriever | None,
    generator: Any,
    indexed_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if generator is None:
        raise RuntimeError("Generator is not initialized")

    if retriever is None:
        return {
            "question": question,
            "answer": "Insufficient evidence in local index. Run a web search query first.",
            "query_hit_count": 0,
            "evidence": [],
            "llm_plan": llm_plan,
        }

    topic_terms = llm_plan.get("topic_terms", [])
    terms = [str(t).strip() for t in topic_terms if str(t).strip()] if isinstance(topic_terms, list) else []

    selected_k = max(1, min(12, max(args.rag_top_k, int(llm_plan.get("max_results", args.rag_top_k)))))
    query_seed = str(llm_plan.get("search_query", "")).strip() or question
    if terms:
        query_seed = f"{query_seed} {' '.join(terms)}"

    retrieval_query = query_seed if not context else f"{query_seed}\n{context}"
    retrieval_candidates = retriever.retrieve(
        query=retrieval_query,
        top_k=max(args.rag_top_k * 4, selected_k),
        min_score=args.rag_min_score,
    )
    retrieved_chunks = retrieval_candidates[:selected_k]

    unique_urls: set[str] = set()
    for chunk in retrieved_chunks:
        meta = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
        url = str(meta.get("url", "")).strip()
        if url:
            unique_urls.add(url)

    corpus_stats = _corpus_stats(indexed_rows)
    prompt = _build_prompt(
        system_policy=system_policy,
        prompt_artifact=prompt_artifact,
        llm_plan=llm_plan,
        question=question,
        context=context,
        retrieved_chunks=retrieved_chunks,
        corpus_stats=corpus_stats,
        query_hit_count=len(unique_urls),
        max_snippet_chars=min(args.max_snippet_chars, 220),
    )

    answer = _generate_answer(
        generator,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    evidence: list[dict[str, Any]] = []
    for chunk in retrieved_chunks:
        meta = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
        evidence.append(
            {
                "score": float(chunk.get("score", 0.0)),
                "source": str(chunk.get("source", "")),
                "kind": str(meta.get("kind", "")),
                "title": str(meta.get("title", "")),
                "url": str(meta.get("url", "")),
                "published_at": str(meta.get("published_at", "")),
            }
        )

    return {
        "question": question,
        "answer": answer,
        "query_hit_count": len(unique_urls),
        "evidence": evidence,
        "llm_plan": llm_plan,
    }


def _print_result(result: dict[str, Any], search_summary: dict[str, Any] | None) -> None:
    print("Answer:")
    print(str(result.get("answer", "")).strip())
    print("")

    evidence = result.get("evidence")
    if isinstance(evidence, list) and evidence:
        print("Evidence:")
        for idx, item in enumerate(evidence[:6], start=1):
            if not isinstance(item, dict):
                continue
            score = float(item.get("score", 0.0))
            title = str(item.get("title", "")).strip() or "(untitled)"
            url = str(item.get("url", "")).strip() or "(no-url)"
            kind = str(item.get("kind", "")).strip() or "unknown"
            print(f"- [{idx}] ({kind}) score={score:.4f} | {title} | {url}")
        print("")

    if search_summary and isinstance(search_summary, dict):
        results = search_summary.get("search_results")
        if isinstance(results, list) and results:
            print("Top searched URLs:")
            for idx, item in enumerate(results[:5], start=1):
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title", "")).strip() or "(untitled)"
                url = str(item.get("url", "")).strip()
                if url:
                    print(f"- [{idx}] {title} | {url}")
            print("")


def main() -> None:
    args = parse_args()

    mcp_server_command = str(args.mcp_server_command).strip()

    if args.search_max_results <= 0:
        raise ValueError("--search-max-results must be greater than 0")
    if args.fetch_top_pages < 0:
        raise ValueError("--fetch-top-pages must be >= 0")
    if args.page_chunk_size < 200:
        raise ValueError("--page-chunk-size must be at least 200")
    if args.page_chunk_overlap < 0:
        raise ValueError("--page-chunk-overlap must be non-negative")
    if args.page_chunk_overlap >= args.page_chunk_size:
        raise ValueError("--page-chunk-overlap must be smaller than --page-chunk-size")
    if not mcp_server_command:
        raise ValueError(
            "Missing MCP server command.\n"
            "Set MCP_SERVER_COMMAND in .env or pass --mcp-server-command.\n"
            "Example:\n"
            "python3 main.py run -- --mcp-server-command \"npx -y <your-web-mcp-server>\""
        )
    if mcp_server_command == DEFAULT_MCP_SERVER_COMMAND and not _env_value("FIRECRAWL_API_KEY", default=""):
        raise ValueError(
            "Default MCP server is Firecrawl (`npx -y firecrawl-mcp`) and requires FIRECRAWL_API_KEY.\n"
            "Set FIRECRAWL_API_KEY in your environment/.env, or override --mcp-server-command."
        )

    prompt_artifact_path = _resolve_prompt_artifact_path(args.prompt_artifact)
    prompt_artifact, prompt_source = _load_prompt_artifact(prompt_artifact_path)
    if prompt_source == "built-in-default":
        print("[info] Using built-in default prompt artifact (no DSPy optimized prompt found).")
    else:
        print(f"[info] Using DSPy optimized prompt artifact: {prompt_source}")

    system_policy = _load_system_policy(args.system_prompt_file)
    generator: Any = _build_generator(task=args.hf_task, model_id=args.hf_model_id)

    retriever, indexed_rows = _load_retriever_and_rows(args)

    print("Chat mode. Ask any topic and the assistant will search the web via MCP.")
    print("Type 'exit' to quit.")

    while True:
        try:
            user_message = input("browser-chat> ").strip()
        except EOFError:
            break

        if not user_message:
            continue
        if user_message.lower() in {"exit", "quit"}:
            break

        if not args.json_output:
            print(f"{args.hf_model_id} Thinking.....")

        try:
            llm_plan = _plan_query_with_llm(
                question=user_message,
                context=str(args.context),
                system_policy=system_policy,
                generator=generator,
            )
        except Exception as exc:
            fallback_warning = f"Planner failed ({exc}). Falling back to direct web search."
            llm_plan = {
                "intent": "web_search",
                "search_query": user_message,
                "topic_terms": [],
                "must_match_all_terms": True,
                "request_latest": False,
                "need_fresh_search": True,
                "max_results": max(3, min(12, args.search_max_results)),
                "language": "en",
            }
            if args.json_output:
                print(json.dumps({"warning": fallback_warning}, ensure_ascii=True))
            else:
                print(f"[warn] {fallback_warning}")

        search_summary: dict[str, Any] | None = None
        search_error: str = ""

        if bool(llm_plan.get("need_fresh_search", True)):
            planned_results = int(llm_plan.get("max_results", args.search_max_results))
            planned_results = max(1, min(args.search_max_results, planned_results))
            try:
                search_summary = search_web_via_mcp(
                    mcp_server_command=str(args.mcp_server_command),
                    search_tool_name=str(args.mcp_search_tool),
                    fetch_tool_name=str(args.mcp_fetch_tool),
                    mcp_account=str(args.mcp_account),
                    query=str(llm_plan.get("search_query", "")).strip() or user_message,
                    max_results=planned_results,
                    fetch_top_pages=max(0, min(args.fetch_top_pages, planned_results)),
                    fetch_max_chars=int(args.fetch_max_chars),
                    startup_timeout_s=int(args.mcp_startup_timeout),
                    request_timeout_s=int(args.mcp_request_timeout),
                )

                new_rows = _rows_from_search_summary(search_summary, args=args)
                if new_rows:
                    existing_rows = _load_rows(args.web_chunks)
                    merged_rows = _merge_rows(existing_rows, new_rows)
                    _write_rows(args.web_chunks, merged_rows)

                retriever, indexed_rows = _load_retriever_and_rows(args)
            except Exception as exc:
                search_error = str(exc)
                if not args.json_output:
                    print(f"[warn] MCP search failed: {search_error}")

        result = _run_single_question(
            question=user_message,
            context=str(args.context),
            system_policy=system_policy,
            prompt_artifact=prompt_artifact,
            llm_plan=llm_plan,
            retriever=retriever,
            generator=generator,
            indexed_rows=indexed_rows,
            args=args,
        )

        payload: dict[str, Any] = {
            "llm_plan": llm_plan,
            "result": result,
        }
        if search_summary is not None:
            payload["search_summary"] = search_summary
        if search_error:
            payload["warning"] = f"MCP search failed: {search_error}"

        if args.json_output:
            print(json.dumps(payload, indent=2, ensure_ascii=True))
        else:
            _print_result(result=result, search_summary=search_summary)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
