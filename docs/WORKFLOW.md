# Main Workflow

This document describes the runtime flow of the local Browser assistant.

## 1. Entry Point

1. Launcher command:
   - `python3 main.py run --`
2. `main.py` forwards to:
   - `src.browser_local_assistant.ask_browser`

## 2. Runtime Initialization

1. `ask_browser.main()` parses CLI + `.env` values.
   - Default MCP command is `npx -y firecrawl-mcp` (requires `FIRECRAWL_API_KEY`).
2. Loads prompt configuration:
   - System policy from `config/browser_llm_system_prompt.txt`
   - Prompt artifact from DSPy output if present (`artifacts/dspy_optimized/optimized_prompt.json`)
3. Builds local LLM generator (`_build_generator`) from the configured HF model/task.

## 3. LLM Query Planning

For each user message:

1. `_plan_query_with_llm(...)` asks the model to return strict JSON plan.
2. Plan includes search query, topic terms, max results, and whether fresh search is needed.
3. Runtime validates the schema and applies one LLM repair pass if JSON is invalid.

## 4. MCP Web Retrieval

When plan requires fresh search:

1. `search_web_via_mcp(...)` calls MCP web search tool.
2. Runtime optionally fetches top pages via MCP fetch/extract tool.
3. Search + page content are normalized to local chunk schema.
4. Chunks are merged into `data/web_chunks.jsonl`.

## 5. Retriever Build (RAG)

1. `_load_retriever_and_rows(...)` loads rows from:
   - `data/web_chunks.jsonl`
   - optional `data/knowledge_base_chunks.jsonl`
2. `TfidfRagRetriever.from_jsonl_paths(...)` builds in-memory index.

## 6. Grounded Answer Generation

1. Runtime retrieves top chunks (`retriever.retrieve(...)`).
2. `_build_prompt(...)` injects:
   - system policy
   - DSPy instructions/demos
   - LLM plan
   - retrieved evidence snippets
   - corpus stats
3. `_generate_answer(...)` produces final answer.
4. Runtime returns answer + evidence + search summary.

## 7. Design Principle

LLM-first orchestration:

1. LLM decides what to search and how broad.
2. MCP tools execute web retrieval.
3. Local RAG keeps evidence for follow-up questions.
4. Final output remains evidence-grounded and URL-cited.
