from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import dspy

DEFAULT_DOMAIN_CONTEXT = (
    "You are a browser research assistant. "
    "Answer using retrieved web evidence, cite URLs, be concise, and do not invent facts not present in evidence."
)


@dataclass(frozen=True)
class QAExample:
    """Single supervised training example for DSPy prompt optimization."""

    question: str
    answer: str


def load_domain_examples(path: str | Path) -> list[QAExample]:
    """Load JSONL question/answer rows and return validated QA examples."""
    records: list[QAExample] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if question and answer:
            records.append(QAExample(question=question, answer=answer))
    if not records:
        raise ValueError(f"No records were found in {path}")
    return records


def split_examples(
    examples: list[QAExample],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[QAExample], list[QAExample]]:
    """Split examples into shuffled train/dev partitions."""
    if not 0.1 <= train_ratio <= 0.95:
        raise ValueError("train_ratio must be between 0.1 and 0.95")

    cloned = list(examples)
    random.Random(seed).shuffle(cloned)
    cut = max(1, int(len(cloned) * train_ratio))
    cut = min(cut, len(cloned) - 1)
    return cloned[:cut], cloned[cut:]


def to_dspy_examples(
    examples: Iterable[QAExample],
    context: str = DEFAULT_DOMAIN_CONTEXT,
) -> list[dspy.Example]:
    """Convert QA examples into DSPy examples with fixed context input."""
    out: list[dspy.Example] = []
    for row in examples:
        ex = dspy.Example(question=row.question, context=context, answer=row.answer).with_inputs("question", "context")
        out.append(ex)
    return out
