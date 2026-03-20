# Local Web Browser Assistant (MCP + LLM + DSPy + RAG)

A local chat assistant for internet research using:

- MCP tools to search and read web pages.
- Local LLM to understand user intent.
- Local RAG (TF-IDF) over indexed pages plus knowledge-base chunks.
- Optional DSPy prompt optimization.

## Setup (one-time)

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create local config:

```bash
cp .env.example .env
```

3. Get a Firecrawl API key:

1. Create/sign in to your Firecrawl account at [firecrawl.dev](https://www.firecrawl.dev).
2. Open the API keys page: [firecrawl.dev/app/api-keys](https://www.firecrawl.dev/app/api-keys).
3. Create a new key and copy it.

4. Configure `.env`:

- Default MCP server is `firecrawl-mcp`:
  - `MCP_SERVER_COMMAND="npx -y firecrawl-mcp"`
  - `FIRECRAWL_API_KEY=fc-...`
- You can override `MCP_SERVER_COMMAND` with any other MCP web server if needed.
- Optional: `MCP_WEB_SEARCH_TOOL`, `MCP_WEB_FETCH_TOOL`, `MCP_ACCOUNT`.

Note: if tool names are not provided, runtime attempts auto-detection using common aliases.

## Run Chat

```bash
python3 main.py run --
```

Prompt examples:

- `search the latest Rust updates in 2026`
- `compare Playwright vs Selenium with sources`
- `summarize this technology and include links`

## Runtime Flow

1. LLM creates a structured query plan (JSON).
2. MCP executes web search and optional page fetch.
3. Chunks are indexed in `data/web_chunks.jsonl`.
4. RAG retrieves relevant evidence (`data/web_chunks.jsonl` + `data/knowledge_base_chunks.jsonl`).
5. LLM answers with grounded evidence and URLs.

## Additional Commands

Build KB chunks:

```bash
python3 main.py build-rag -- --input-paths data/knowledge_base.md --output-jsonl data/knowledge_base_chunks.jsonl
```

Optimize prompts with DSPy (optional):

```bash
python3 main.py optimize-prompts -- \
  --dataset data/web_assistant_qa.jsonl \
  --domain-context-file config/browser_domain_context.txt \
  --output-dir artifacts/dspy_optimized \
  --compiler-model ollama_chat/llama3.1:8b \
  --auto light
```

## Security

- Do not commit `.env`.
- `data/web_chunks.jsonl` may contain sensitive content retrieved from websites.
