from __future__ import annotations

import json
import queue
import shlex
import subprocess
import threading
import time
from typing import Any

SEARCH_TOOL_ALIASES = (
    "searchweb",
    "websearch",
    "web_search",
    "search",
    "brave_web_search",
    "tavily_search",
    "firecrawl_search",
    "exa_search",
    "duckduckgo_search",
    "google_search",
)

FETCH_TOOL_ALIASES = (
    "fetchurl",
    "fetch_url",
    "fetch",
    "readurl",
    "read_url",
    "getwebpage",
    "get_web_page",
    "webfetch",
    "web_fetch",
    "extract",
    "tavily_extract",
    "firecrawl_scrape",
    "exa_get_contents",
    "browser_get_content",
)


def _read_json_message_lsp(stream) -> dict[str, Any] | None:
    """Read one LSP-framed JSON-RPC message from a byte stream."""
    content_length = -1

    while True:
        raw_line = stream.readline()
        if raw_line == b"":
            return None
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line:
            break
        lower = line.lower()
        if lower.startswith("content-length:"):
            _, raw_len = line.split(":", 1)
            try:
                content_length = int(raw_len.strip())
            except Exception:
                content_length = -1

    if content_length <= 0:
        return None

    payload = stream.read(content_length)
    if not payload:
        return None

    try:
        decoded = json.loads(payload.decode("utf-8", errors="replace"))
    except Exception:
        return None

    if isinstance(decoded, dict):
        return decoded
    return None


class _StdioMcpClient:
    """Minimal MCP stdio client supporting LSP and NDJSON transport."""

    def __init__(
        self,
        command: str,
        startup_timeout_s: int = 20,
        request_timeout_s: int = 45,
        stdio_protocol: str = "auto",
    ):
        """Configure process command, protocol mode, and timeout settings."""
        self.command = (command or "").strip()
        self.startup_timeout_s = max(5, int(startup_timeout_s))
        self.request_timeout_s = max(5, int(request_timeout_s))
        protocol = (stdio_protocol or "auto").strip().lower()
        if protocol not in {"auto", "lsp", "ndjson"}:
            protocol = "auto"
        self.stdio_protocol = protocol

        self._active_stdio_protocol = "lsp"
        self._proc: subprocess.Popen[bytes] | None = None
        self._messages: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._reader_error: Exception | None = None
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stderr_lines: list[str] = []
        self._next_id = 1
        self._write_lock = threading.Lock()

    def __enter__(self) -> "_StdioMcpClient":
        """Start MCP subprocess, negotiate protocol, and send initialize."""
        if not self.command:
            raise ValueError("Missing MCP server command. Set --mcp-server-command or MCP_SERVER_COMMAND.")

        cmd = shlex.split(self.command)
        if not cmd:
            raise ValueError("Invalid MCP server command.")

        protocol_attempts = ["lsp", "ndjson"] if self.stdio_protocol == "auto" else [self.stdio_protocol]
        last_error: Exception | None = None
        for protocol in protocol_attempts:
            self._start_process(cmd, protocol=protocol)
            try:
                self.initialize()
                return self
            except Exception as exc:
                last_error = exc
                self._stop_process()

        if last_error is not None:
            raise last_error
        raise RuntimeError("Could not initialize MCP process.")

    def __exit__(self, exc_type, exc, tb) -> None:
        """Ensure subprocess resources are released on context exit."""
        self._stop_process()

    def _start_process(self, cmd: list[str], protocol: str) -> None:
        """Spawn MCP process and start background reader threads."""
        self._active_stdio_protocol = protocol
        self._messages = queue.Queue()
        self._reader_error = None
        self._stderr_lines = []

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        if self._proc.stdin is None or self._proc.stdout is None or self._proc.stderr is None:
            self._stop_process()
            raise RuntimeError("Could not open MCP stdio pipes.")

        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

        self._stderr_thread = threading.Thread(target=self._stderr_loop, daemon=True)
        self._stderr_thread.start()

    def _stop_process(self) -> None:
        """Terminate MCP process gracefully, force-kill as fallback."""
        proc = self._proc
        self._proc = None
        if proc is None:
            return

        try:
            proc.terminate()
        except Exception:
            pass

        try:
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _stderr_loop(self) -> None:
        """Continuously collect stderr lines for diagnostics."""
        proc = self._proc
        if proc is None or proc.stderr is None:
            return

        while True:
            line = proc.stderr.readline()
            if line == b"":
                break
            text = line.decode("utf-8", errors="replace").rstrip()
            if text:
                self._stderr_lines.append(text)
                if len(self._stderr_lines) > 80:
                    self._stderr_lines = self._stderr_lines[-80:]

    def _reader_loop(self) -> None:
        """Continuously parse stdout JSON messages into queue entries."""
        try:
            proc = self._proc
            if proc is None or proc.stdout is None:
                return

            if self._active_stdio_protocol == "ndjson":
                while True:
                    raw_line = proc.stdout.readline()
                    if raw_line == b"":
                        break
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        decoded = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(decoded, dict):
                        self._messages.put(decoded)
            else:
                while True:
                    msg = _read_json_message_lsp(proc.stdout)
                    if msg is None:
                        break
                    self._messages.put(msg)
        except Exception as exc:
            self._reader_error = exc
        finally:
            self._messages.put(None)

    def _send_message(self, payload: dict[str, Any]) -> None:
        """Serialize and write one JSON-RPC payload to MCP stdin."""
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise RuntimeError("MCP process is not available.")

        raw = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        with self._write_lock:
            if self._active_stdio_protocol == "ndjson":
                proc.stdin.write(raw + b"\n")
            else:
                header = f"Content-Length: {len(raw)}\r\n\r\n".encode("ascii")
                proc.stdin.write(header)
                proc.stdin.write(raw)
            proc.stdin.flush()

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        self._send_message(payload)

    def request(self, method: str, params: dict[str, Any] | None = None, timeout_s: int | None = None) -> dict[str, Any]:
        """Send a JSON-RPC request and return the matching result payload."""
        request_id = self._next_id
        self._next_id += 1

        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            payload["params"] = params
        self._send_message(payload)

        timeout_value = self.request_timeout_s if timeout_s is None else max(1, int(timeout_s))
        deadline = time.monotonic() + timeout_value

        while time.monotonic() < deadline:
            if self._reader_error is not None:
                raise RuntimeError(f"MCP reader failed: {self._reader_error}") from self._reader_error

            remaining = max(0.1, deadline - time.monotonic())
            try:
                msg = self._messages.get(timeout=remaining)
            except queue.Empty:
                continue

            if msg is None:
                break

            msg_id = msg.get("id")
            if msg_id is None:
                continue
            if str(msg_id) != str(request_id):
                continue

            if "error" in msg and isinstance(msg["error"], dict):
                err = msg["error"]
                code = err.get("code")
                text = str(err.get("message", "Unknown MCP error")).strip()
                stderr_tail = "\n".join(self._stderr_lines[-10:]).strip()
                if stderr_tail:
                    text = f"{text}\nMCP stderr:\n{stderr_tail}"
                raise RuntimeError(f"MCP request '{method}' failed (code={code}): {text}")

            result = msg.get("result")
            if isinstance(result, dict):
                return result
            return {"value": result}

        stderr_tail = "\n".join(self._stderr_lines[-10:]).strip()
        if stderr_tail:
            raise TimeoutError(f"MCP request timeout for '{method}'.\nMCP stderr:\n{stderr_tail}")
        raise TimeoutError(f"MCP request timeout for '{method}'.")

    def initialize(self) -> None:
        """Perform MCP handshake and emit initialized notification."""
        self.request(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "genai-browser-local-assistant", "version": "0.1.0"},
            },
            timeout_s=self.startup_timeout_s,
        )
        self.notify("notifications/initialized", {})

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call MCP `tools/call` with tool name plus argument object."""
        return self.request("tools/call", {"name": name, "arguments": arguments})

    def list_tools(self) -> list[str]:
        """Return available MCP tool names from `tools/list`."""
        result = self.request("tools/list", {})
        tools = result.get("tools")
        if not isinstance(tools, list):
            return []

        out: list[str] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            name = str(tool.get("name", "")).strip()
            if name:
                out.append(name)
        return out


def _extract_json_from_text(text: str) -> Any | None:
    """Try to parse JSON from raw text, including embedded object/array slices."""
    raw = (text or "").strip()
    if not raw:
        return None

    try:
        return json.loads(raw)
    except Exception:
        pass

    first_obj = raw.find("{")
    last_obj = raw.rfind("}")
    if first_obj >= 0 and last_obj > first_obj:
        try:
            return json.loads(raw[first_obj : last_obj + 1])
        except Exception:
            pass

    first_arr = raw.find("[")
    last_arr = raw.rfind("]")
    if first_arr >= 0 and last_arr > first_arr:
        try:
            return json.loads(raw[first_arr : last_arr + 1])
        except Exception:
            pass

    return None


def _as_text(value: Any) -> str:
    """Normalize scalar/list values into a trimmed string representation."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value).strip()
    if isinstance(value, list):
        merged = ", ".join([_as_text(v) for v in value if _as_text(v)])
        return merged.strip()
    return ""


def _dedupe_keep_order(values: list[str]) -> list[str]:
    """Remove duplicates while preserving first-seen order."""
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _walk_dicts(payload: Any) -> list[dict[str, Any]]:
    """Collect every dictionary found recursively inside payload."""
    found: list[dict[str, Any]] = []

    def _walk(node: Any) -> None:
        """Recursively traverse nested containers collecting dict nodes."""
        if isinstance(node, dict):
            found.append(node)
            for child in node.values():
                _walk(child)
            return
        if isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(payload)
    return found


def _select_search_tool(available_tools: list[str], explicit_tool_name: str) -> str:
    """Resolve search tool name via explicit value, aliases, then heuristics."""
    explicit = (explicit_tool_name or "").strip()
    if explicit:
        return explicit

    if not available_tools:
        return "search"

    normalized_available = {name.lower(): name for name in available_tools}
    for alias in SEARCH_TOOL_ALIASES:
        if alias in normalized_available:
            return normalized_available[alias]

    for name in available_tools:
        lname = name.lower()
        if "search" in lname and "web" in lname:
            return name

    for name in available_tools:
        lname = name.lower()
        if "search" in lname:
            return name

    raise RuntimeError(
        "Could not auto-detect a web search MCP tool. "
        "Set --mcp-search-tool explicitly. Available tools: "
        + ", ".join(available_tools)
    )


def _select_fetch_tool(available_tools: list[str], explicit_tool_name: str) -> str:
    """Resolve fetch/extract tool name via explicit value, aliases, then heuristics."""
    explicit = (explicit_tool_name or "").strip()
    if explicit:
        return explicit

    if not available_tools:
        return ""

    normalized_available = {name.lower(): name for name in available_tools}
    for alias in FETCH_TOOL_ALIASES:
        if alias in normalized_available:
            return normalized_available[alias]

    for name in available_tools:
        lname = name.lower()
        if ("fetch" in lname or "extract" in lname or "scrape" in lname or "read" in lname) and "search" not in lname:
            return name

    return ""


def _with_account(arguments: dict[str, Any], account: str) -> dict[str, Any]:
    """Attach optional account field to tool arguments."""
    account_value = (account or "").strip()
    if not account_value:
        return dict(arguments)
    merged = dict(arguments)
    merged["account"] = account_value
    return merged


def _call_tool_with_candidates(
    client: _StdioMcpClient,
    tool_name: str,
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Try argument variants until one tool call succeeds."""
    last_error: Exception | None = None
    for arguments in candidates:
        try:
            result = client.call_tool(tool_name, arguments)
            return result, arguments
        except Exception as exc:
            last_error = exc

    if last_error is None:
        raise RuntimeError(f"Could not call MCP tool '{tool_name}'. No argument candidates configured.")
    raise RuntimeError(f"Could not call MCP tool '{tool_name}' with available arguments: {last_error}") from last_error


def _normalize_search_item(record: dict[str, Any], rank: int) -> dict[str, Any] | None:
    """Map heterogeneous search record fields into normalized result schema."""
    url = ""
    for key in ("url", "link", "href", "uri", "source_url", "sourceUrl"):
        url = _as_text(record.get(key))
        if url:
            break

    if not url:
        return None

    title = ""
    for key in ("title", "name", "headline"):
        title = _as_text(record.get(key))
        if title:
            break

    snippet = ""
    for key in ("snippet", "description", "summary", "text", "content", "excerpt"):
        snippet = _as_text(record.get(key))
        if snippet:
            break

    published_at = ""
    for key in ("published_at", "publishedAt", "date", "timestamp", "lastModified"):
        published_at = _as_text(record.get(key))
        if published_at:
            break

    return {
        "rank": rank,
        "url": url,
        "title": title,
        "snippet": snippet,
        "published_at": published_at,
    }


def _extract_search_items_from_tool_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract and deduplicate normalized search results from MCP payload."""
    candidates: list[dict[str, Any]] = []

    structured = result.get("structuredContent")
    candidates.extend(_walk_dicts(structured))

    content = result.get("content")
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue

            for key in ("json", "data"):
                candidates.extend(_walk_dicts(block.get(key)))

            raw_text = _as_text(block.get("text"))
            parsed = _extract_json_from_text(raw_text)
            if parsed is not None:
                candidates.extend(_walk_dicts(parsed))

    out: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for record in candidates:
        item = _normalize_search_item(record, rank=len(out) + 1)
        if item is None:
            continue
        key = item["url"].strip()
        if key in seen_urls:
            continue
        seen_urls.add(key)
        out.append(item)

    return out


def _collect_text_fragments(payload: Any, max_fragments: int = 80) -> list[str]:
    """Collect candidate text snippets recursively from nested payload objects."""
    fragments: list[str] = []

    def _add(text: str) -> None:
        """Normalize and append one candidate fragment if non-empty."""
        normalized = " ".join(text.split())
        if not normalized:
            return
        fragments.append(normalized)

    def _walk(node: Any) -> None:
        """Traverse nested payload and extract text-like fields recursively."""
        if len(fragments) >= max_fragments:
            return

        if isinstance(node, str):
            _add(node)
            return

        if isinstance(node, list):
            for item in node:
                _walk(item)
                if len(fragments) >= max_fragments:
                    return
            return

        if isinstance(node, dict):
            for key in (
                "markdown",
                "content",
                "text",
                "body",
                "article",
                "summary",
                "description",
                "excerpt",
                "html",
                "raw",
            ):
                if key in node:
                    _walk(node.get(key))
                    if len(fragments) >= max_fragments:
                        return

            for value in node.values():
                if isinstance(value, (dict, list, str)):
                    _walk(value)
                    if len(fragments) >= max_fragments:
                        return

    _walk(payload)

    deduped: list[str] = []
    seen: set[str] = set()
    for item in fragments:
        key = item[:220]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _extract_page_text_from_tool_result(result: dict[str, Any]) -> str:
    """Extract merged textual page content from MCP fetch tool response."""
    fragments: list[str] = []

    structured = result.get("structuredContent")
    if structured is not None:
        fragments.extend(_collect_text_fragments(structured))

    content = result.get("content")
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue

            for key in ("json", "data"):
                payload = block.get(key)
                if payload is not None:
                    fragments.extend(_collect_text_fragments(payload))

            text = _as_text(block.get("text"))
            if text:
                parsed = _extract_json_from_text(text)
                if parsed is not None:
                    fragments.extend(_collect_text_fragments(parsed))
                else:
                    fragments.append(" ".join(text.split()))

    if not fragments:
        return ""

    merged = "\n".join(_dedupe_keep_order([f for f in fragments if f]))
    return merged.strip()


def _trim_text(text: str, max_chars: int) -> str:
    """Clamp text length to max_chars, preserving whole prefix semantics."""
    if max_chars <= 0:
        return ""
    raw = (text or "").strip()
    if len(raw) <= max_chars:
        return raw
    return raw[:max_chars].rstrip() + "..."


def search_web_via_mcp(
    *,
    mcp_server_command: str,
    search_tool_name: str,
    fetch_tool_name: str,
    mcp_account: str,
    query: str,
    max_results: int,
    fetch_top_pages: int,
    fetch_max_chars: int,
    startup_timeout_s: int = 20,
    request_timeout_s: int = 45,
) -> dict[str, Any]:
    """Execute MCP web search plus optional page fetch and return normalized output."""
    query_value = (query or "").strip()
    if not query_value:
        raise ValueError("query must be a non-empty string")

    result_limit = max(1, int(max_results))
    top_pages = max(0, int(fetch_top_pages))
    max_chars = max(300, int(fetch_max_chars))

    account_value = (mcp_account or "").strip()

    with _StdioMcpClient(
        command=mcp_server_command,
        startup_timeout_s=startup_timeout_s,
        request_timeout_s=request_timeout_s,
    ) as client:
        available_tools = client.list_tools()
        selected_search_tool = _select_search_tool(available_tools, search_tool_name)
        selected_fetch_tool = _select_fetch_tool(available_tools, fetch_tool_name)

        search_candidates = [
            _with_account({"query": query_value, "max_results": result_limit}, account_value),
            _with_account({"query": query_value, "maxResults": result_limit}, account_value),
            _with_account({"q": query_value, "max_results": result_limit}, account_value),
            _with_account({"q": query_value, "limit": result_limit}, account_value),
            _with_account({"query": query_value}, account_value),
            _with_account({"q": query_value}, account_value),
        ]

        search_result, used_search_args = _call_tool_with_candidates(client, selected_search_tool, search_candidates)
        search_results = _extract_search_items_from_tool_result(search_result)
        search_results = search_results[:result_limit]

        fetched_pages: list[dict[str, Any]] = []
        if selected_fetch_tool and top_pages > 0:
            for item in search_results[:top_pages]:
                url = str(item.get("url", "")).strip()
                if not url:
                    continue

                fetch_candidates = [
                    _with_account({"url": url}, account_value),
                    _with_account({"link": url}, account_value),
                    _with_account({"href": url}, account_value),
                    _with_account({"uri": url}, account_value),
                    _with_account({"urls": [url]}, account_value),
                ]

                try:
                    page_result, used_fetch_args = _call_tool_with_candidates(client, selected_fetch_tool, fetch_candidates)
                    content = _trim_text(_extract_page_text_from_tool_result(page_result), max_chars=max_chars)
                    fetched_pages.append(
                        {
                            "url": url,
                            "title": str(item.get("title", "")).strip(),
                            "rank": int(item.get("rank", 0) or 0),
                            "content": content,
                            "used_fetch_arguments": used_fetch_args,
                        }
                    )
                except Exception as exc:
                    fetched_pages.append(
                        {
                            "url": url,
                            "title": str(item.get("title", "")).strip(),
                            "rank": int(item.get("rank", 0) or 0),
                            "content": "",
                            "error": str(exc),
                        }
                    )

    return {
        "transport": "mcp",
        "query": query_value,
        "mcp_search_tool": selected_search_tool,
        "mcp_fetch_tool": selected_fetch_tool,
        "used_search_arguments": used_search_args,
        "account": account_value,
        "results_count": len(search_results),
        "search_results": search_results,
        "fetched_pages": fetched_pages,
    }
