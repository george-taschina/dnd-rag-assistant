"""Per-session cost tracking for the D&D agent (course cost_tracker pattern).

Adapted from ai-engineering-buildcamp-code/documentation-agent/cost_tracker.py:
same price table and accumulate-then-price flow, grouped per app session.
Model keys are normalized so "openai-responses:gpt-4o-mini" maps to the
"gpt-4o-mini" prices.
"""

from dataclasses import dataclass, field

MODEL_PRICES = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-5.2": {"input": 1.75, "output": 14.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
}


def _base_model(model: str) -> str:
    """'openai-responses:gpt-4o-mini' -> 'gpt-4o-mini'."""
    return model.lower().split(":")[-1]


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    prices = MODEL_PRICES.get(_base_model(model), {"input": 0.0, "output": 0.0})
    return (input_tokens / 1_000_000) * prices["input"] + \
           (output_tokens / 1_000_000) * prices["output"]


@dataclass
class SessionCost:
    """Accumulates usage for one app session (one Logfire trace tree)."""

    model: str = "openai-responses:gpt-4o-mini"
    input_tokens: int = 0
    output_tokens: int = 0
    runs: int = 0
    tool_calls: list = field(default_factory=list)

    def add(self, usage) -> None:
        self.input_tokens += usage.input_tokens or 0
        self.output_tokens += usage.output_tokens or 0
        self.runs += 1

    @property
    def cost_usd(self) -> float:
        return cost_usd(self.model, self.input_tokens, self.output_tokens)

    def summary(self) -> str:
        return (
            f"{self.runs} runs | {self.input_tokens} in / "
            f"{self.output_tokens} out tokens | ${self.cost_usd:.6f}"
        )
