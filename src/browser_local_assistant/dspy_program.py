from __future__ import annotations

import dspy


class BrowserDomainAnswer(dspy.Signature):
    """Answer user questions from retrieved web evidence with clear citations."""

    context = dspy.InputField(desc="Web research policy and constraints")
    question = dspy.InputField(desc="User question about a topic to investigate on the web")
    answer = dspy.OutputField(desc="Accurate and concise answer grounded in retrieved web and knowledge evidence")


class BrowserAssistant(dspy.Module):
    """DSPy module wrapper that predicts grounded browser-research answers."""

    def __init__(self) -> None:
        """Initialize prediction head for the browser-domain signature."""
        super().__init__()
        self.respond = dspy.Predict(BrowserDomainAnswer)

    def forward(self, question: str, context: str):
        """Run one inference step using question plus domain context."""
        return self.respond(question=question, context=context)
