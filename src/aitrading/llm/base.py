"""LLM abstraction used by the NL screen translator and the explanation agent.

Only one capability is required by the deterministic pipeline: *structured* generation - a system
prompt plus user content in, a validated pydantic object out. Implementations record every call
(model, request id, token usage, stop reason) so runs are auditable.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from aitrading.core.models import LLMCallRecord

T = TypeVar("T", bound=BaseModel)

Effort = str  # "low" | "medium" | "high" | "xhigh" | "max"


class LLMError(RuntimeError):
    """The model call failed or returned unusable output."""


class LLMRefusalError(LLMError):
    """The model (and any configured fallback) declined the request."""

    def __init__(self, message: str, category: str | None = None):
        super().__init__(message)
        self.category = category


@runtime_checkable
class StructuredLLM(Protocol):
    name: str
    calls: list[LLMCallRecord]

    def structured(
        self,
        *,
        purpose: str,
        system: str,
        user: str | list[dict[str, Any]],
        output_model: type[T],
        effort: Effort | None = None,
        max_tokens: int = 16_000,
    ) -> T:
        """Return an instance of ``output_model``; raise LLMError / LLMRefusalError on failure."""
        ...


Responder = Callable[[str, str, Any, type], BaseModel]


class ScriptedLLM:
    """Deterministic test double: responses are produced by per-purpose callables.

    ``responders`` maps a purpose prefix (e.g. "nl_screen", "explain") to a function
    ``(purpose, system, user, output_model) -> output_model instance``. The longest matching
    prefix wins. Every call is recorded like a real one.
    """

    name = "scripted"

    def __init__(self, responders: dict[str, Responder]):
        self.responders = responders
        self.calls: list[LLMCallRecord] = []
        self.prompts: list[dict[str, Any]] = []

    def structured(self, *, purpose, system, user, output_model, effort=None, max_tokens=16_000):
        self.prompts.append({"purpose": purpose, "system": system, "user": user, "output_model": output_model.__name__})
        match = max((p for p in self.responders if purpose.startswith(p)), key=len, default=None)
        if match is None:
            self.calls.append(LLMCallRecord(purpose=purpose, model=self.name, error="no responder"))
            raise LLMError(f"ScriptedLLM has no responder for purpose '{purpose}'")
        out = self.responders[match](purpose, system, user, output_model)
        if not isinstance(out, output_model):
            out = output_model.model_validate(out)
        self.calls.append(LLMCallRecord(purpose=purpose, model=self.name, stop_reason="end_turn"))
        return out
