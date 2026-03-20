from __future__ import annotations

import argparse
import subprocess
import sys

MODULE_BY_COMMAND = {
    "run": "src.browser_local_assistant.ask_browser",
    "build-rag": "src.browser_local_assistant.build_rag_index",
    "optimize-prompts": "src.browser_local_assistant.optimize_prompts",
}


def run_module(module: str, module_args: list[str]) -> int:
    cmd = [sys.executable, "-m", module, *module_args]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launcher for local browser assistant (DSPy + MCP web + local HF)")
    parser.add_argument(
        "command",
        choices=sorted(MODULE_BY_COMMAND.keys()),
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "module_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the selected stage",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    module = MODULE_BY_COMMAND[args.command]

    forwarded = args.module_args
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    return run_module(module, forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
