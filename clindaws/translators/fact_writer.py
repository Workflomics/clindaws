from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from io import StringIO


from clindaws.translators.utils import _fact



@dataclass
class _FactWriter:
    """Incrementally build fact text and counts."""

    buffer: StringIO = field(default_factory=StringIO)
    predicate_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    fact_count: int = 0

    def emit_fact(self, name: str, *args: str) -> None:
        self.buffer.write(_fact(name, *args))
        self.buffer.write("\n")
        self.predicate_counts[name] += 1
        self.fact_count += 1

    def emit_atom(self, name: str) -> None:
        self.buffer.write(f"{name}.\n")
        self.predicate_counts[name] += 1
        self.fact_count += 1

    def emit_rule(self, name: str, text: str) -> None:
        self.buffer.write(text)
        if not text.endswith("\n"):
            self.buffer.write("\n")
        self.predicate_counts[name] += 1
        self.fact_count += 1

    def emit_comment(self, text: str) -> None:
        self.buffer.write(f"% {text}\n")

    def text(self) -> str:
        return self.buffer.getvalue()

    def stats(self) -> dict[str, int]:
        return {
            "fact_count": self.fact_count,
            "text_chars": self.buffer.tell(),
        }
