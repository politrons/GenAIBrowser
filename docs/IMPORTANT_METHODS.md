# Important Methods (Developer Guide)

## 1) Query/chat engine (`src/browser_local_assistant/ask_browser.py`)

### `parse_args()`
- Loads `.env` and builds CLI for chat runtime, model config, RAG config, and MCP web transport.
- Main groups: `--web-chunks` / `--knowledge-chunks`, prompt settings, HF model settings, MCP tool settings, and fetch/chunk limits.

### `_plan_query_with_llm(question, context, system_policy, generator)`
- First LLM planning stage.
- Produces strict JSON retrieval plan (`search_query`, `topic_terms`, `need_fresh_search`, `max_results`, etc.).
- If JSON is invalid, it runs one LLM-based repair pass.

### `_rows_from_search_summary(summary, args)`
- Converts MCP search/fetch output into local JSONL chunk rows.
- Stores both search snippets and chunked fetched-page content.

### `_load_retriever_and_rows(args)`
- Builds retriever from local indexed web chunks and optional KB chunks.
- Returns retriever + loaded rows used for corpus stats and evidence.

### `_run_single_question(...)`
- Retrieval + grounded answer pipeline:
1. Build retrieval query from LLM plan.
2. Retrieve top chunks from TF-IDF index.
3. Build final grounded prompt.
4. Generate final answer.
5. Return answer + evidence metadata.

### `_build_generator(...)` and `_generate_answer(...)`
- Local generation backend for both seq2seq and causal models.
- Handles max token settings and deterministic generation defaults.

### `main()`
- Interactive chat loop.
- Runs query planning, MCP web retrieval, local index updates, RAG retrieval, and final response rendering.

## 2) MCP web transport (`src/browser_local_assistant/mcp_web.py`)

### `_StdioMcpClient`
- Minimal MCP stdio JSON-RPC client (`initialize`, `tools/list`, `tools/call`).
- Supports both LSP framing and NDJSON framing.

### `search_web_via_mcp(...)`
- Calls configured (or auto-detected) MCP search tool.
- Optionally calls fetch/extract tool for top result URLs.
- Returns normalized `search_results` and `fetched_pages` payload.

### `_extract_search_items_from_tool_result(...)` / `_extract_page_text_from_tool_result(...)`
- Normalization helpers for heterogeneous MCP server payloads.
- Extracts URL/title/snippet for search and text fragments for page content.

## 3) Local retriever (`src/browser_local_assistant/rag_retriever.py`)

### `TfidfRagRetriever.from_jsonl_paths(paths)`
- Loads JSONL chunk files and builds in-memory TF-IDF index.

### `TfidfRagRetriever.retrieve(query, top_k, min_score)`
- Returns ranked chunks for grounded answer prompts.

## 4) Prompt optimization (`src/browser_local_assistant/optimize_prompts.py`)

### `_preflight_ollama(args)`
- Validates local Ollama availability when using `ollama/...` compiler model.

### `_extract_prompt_artifact(...)`
- Extracts optimized instructions + demos from DSPy optimized program.
- Writes runtime artifact `optimized_prompt.json`.

### `main()`
- Loads dataset, splits train/dev, runs DSPy optimization, evaluates baseline vs optimized, and writes artifacts.

## 5) Runtime strictness

- Planner JSON is schema-validated.
- One LLM repair pass is attempted on malformed JSON.
- If schema remains invalid after repair, runtime fails fast with explicit error.
