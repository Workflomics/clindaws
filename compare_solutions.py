import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


ANSWER_SET_RE = re.compile(r"^Answer Set (\d+)$")
WORKFLOW_LINE_RE = re.compile(r"^Workflow.*:\s*(.*)$")
TOOLS_LINE_RE = re.compile(r"^Tools:\s*(.*)$")
SOLUTION_INLINE_RE = re.compile(r"^Solution\s+\d+:\s*(.*)$")
SAT_TOOL_RE = re.compile(r"^(?P<predicate>\S+)\(Tool(?P<index>\d+)\)$")
SAT_BIND_RE = re.compile(r"^memRef\((?P<src>[^,]+),(?P<dst>In\d+\.\d+)\)$")
SAT_OUT_RE = re.compile(r"Out(\d+)\.(\d+)")
SAT_IN_RE = re.compile(r"In(\d+)\.(\d+)")
NORMALIZED_BIND_RE = re.compile(r"^t(\d+):in\d+<-")
SNAKE_TOOL_RE = re.compile(r'tool_at_time\((\d+),"([^"]+)"\)')
SNAKE_BIND_WF_RE = re.compile(r'ape_bind\((\d+),(\d+),"wf_input_(\d+)"\)')
SNAKE_BIND_OUT_RE = re.compile(
    r'ape_bind\((\d+),(\d+),out\((\d+),"[^"]+",(?:(\d+)|"([^"]+)")\)\)'
)
SNAKE_HOLDS_OUT_RE = re.compile(
    r'ape_holds_dim\(out\((\d+),"[^"]+",(?:(\d+)|"([^"]+)")\),"[^"]+","[^"]+"\)'
)
SPLIT_BLANK_RE = re.compile(r"\n\s*\n+", re.MULTILINE)
LEGACY_PORT_SUFFIX_RE = re.compile(r"_p(?:ort_)?(\d+)$")
STRUCTURED_PORT_SUFFIX_RE = re.compile(r"_port_(\d+)$")

IGNORED_SAT_TOOL_PREFIXES = (
    "operation_",
    "data_",
    "format_",
)
IGNORED_SAT_TOOL_NAMES = {
    "tool",
    "format",
    "data",
    "operation",
}


@dataclass(frozen=True)
class NormalizedSolution:
    index: int
    parser_kind: str
    tool_sequence: tuple[str, ...]
    bindings: tuple[str, ...]
    outputs: tuple[str, ...]

    @property
    def tool_key(self) -> tuple[str, ...]:
        return self.tool_sequence

    @property
    def workflow_key(self) -> tuple[object, ...]:
        return (self.tool_sequence, self.bindings, self.outputs)


@dataclass(frozen=True)
class ParsedSolutions:
    parser_kind: str
    path: Path
    solutions: tuple[NormalizedSolution, ...]

    @property
    def tool_counter(self) -> Counter[tuple[str, ...]]:
        return Counter(solution.tool_key for solution in self.solutions)

    @property
    def workflow_counter(self) -> Counter[tuple[object, ...]]:
        return Counter(solution.workflow_key for solution in self.solutions)


def filter_by_length(parsed: ParsedSolutions, length: int | None) -> ParsedSolutions:
    if length is None:
        return parsed
    return ParsedSolutions(
        parser_kind=parsed.parser_kind,
        path=parsed.path,
        solutions=tuple(solution for solution in parsed.solutions if len(solution.tool_sequence) == length),
    )


def parse_tool_sequence_filter(raw: str) -> tuple[str, ...]:
    text = raw.strip()
    separator = "->" if "->" in text else ","
    return tuple(part.strip() for part in text.split(separator) if part.strip())


def filter_by_tool_sequence(
    parsed: ParsedSolutions,
    tool_sequence: tuple[str, ...] | None,
) -> ParsedSolutions:
    if tool_sequence is None:
        return parsed
    return ParsedSolutions(
        parser_kind=parsed.parser_kind,
        path=parsed.path,
        solutions=tuple(
            solution
            for solution in parsed.solutions
            if solution.tool_sequence == tool_sequence
        ),
    )


def extract_local_name(value: str) -> str:
    return value.rsplit("#", 1)[-1].rsplit("/", 1)[-1]


def strip_sat_suffix(value: str) -> str:
    return re.sub(r"\[[^\]]+\]$", "", value)


def strip_snake_suffix(value: str) -> str:
    return re.sub(r"__ann\d+$", "", value)


def normalize_name(value: str) -> str:
    return strip_sat_suffix(extract_local_name(value)).strip().casefold()


def resolve_from_config(config_path: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    for base in [config_path.parent, *config_path.parents]:
        resolved = (base / candidate).resolve()
        if resolved.exists():
            return resolved
    return (config_path.parent / candidate).resolve()


def infer_config_path(solution_path: Path) -> Path | None:
    for parent in [solution_path.parent, *solution_path.parents]:
        sat_candidate = parent / "config_sat.json"
        if sat_candidate.exists():
            return sat_candidate
        config_candidate = parent / "config.json"
        if config_candidate.exists():
            return config_candidate
    return None


def load_tool_catalog(config_path: Path | None) -> dict[str, str]:
    if config_path is None or not config_path.exists():
        return {}

    try:
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(
            f"WARNING: Failed to load comparison config {config_path}: {exc}. "
            "Continuing without tool catalog canonicalization.",
            file=sys.stderr,
        )
        return {}
    raw_tool_path = config_data.get("tool_annotations_path")
    if not raw_tool_path:
        return {}

    tool_path = resolve_from_config(config_path, raw_tool_path)
    if not tool_path.exists():
        return {}

    try:
        tool_data = json.loads(tool_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(
            f"WARNING: Failed to load tool annotations {tool_path}: {exc}. "
            "Continuing without tool catalog canonicalization.",
            file=sys.stderr,
        )
        return {}
    catalog: dict[str, str] = {}
    for function in tool_data.get("functions", []):
        label = str(function.get("label") or function.get("id") or "").strip()
        tool_id = str(function.get("id") or label).strip()
        canonical = label or tool_id
        for candidate in {label, tool_id, extract_local_name(label), extract_local_name(tool_id)}:
            candidate = candidate.strip()
            if candidate:
                catalog[normalize_name(candidate)] = canonical
    return catalog


def choose_sat_tool_name(
    candidates: list[str],
    tool_catalog: dict[str, str],
) -> str | None:
    if not candidates:
        return None

    resolved = []
    for candidate in candidates:
        normalized = normalize_name(candidate)
        if normalized in tool_catalog:
            resolved.append(tool_catalog[normalized])
    resolved = sorted(set(resolved))
    if len(resolved) == 1:
        return resolved[0]
    if len(resolved) > 1:
        return resolved[0]

    filtered = []
    for candidate in candidates:
        local_name = strip_sat_suffix(extract_local_name(candidate))
        normalized = normalize_name(local_name)
        if normalized in IGNORED_SAT_TOOL_NAMES:
            continue
        if any(normalized.startswith(prefix) for prefix in IGNORED_SAT_TOOL_PREFIXES):
            continue
        filtered.append(local_name)

    if not filtered:
        filtered = [strip_sat_suffix(extract_local_name(candidate)) for candidate in candidates]

    def heuristic_score(local_name: str) -> tuple[int, int, int, str]:
        score = 0
        if "_" in local_name or "-" in local_name:
            score += 4
        if local_name and local_name[0].islower():
            score += 3
        if any(char.islower() for char in local_name[1:]):
            score += 1
        if any(char.isdigit() for char in local_name):
            score -= 1
        if local_name and local_name[0].isupper() and "_" not in local_name and "-" not in local_name:
            score -= 1
        return (score, -len(local_name), -sum(1 for char in local_name if char.isupper()), local_name)

    return max(filtered, key=heuristic_score)


def normalize_sat_binding(source_ref: str, target_ref: str) -> str | None:
    if source_ref == "nullMem":
        return None

    source_match = SAT_OUT_RE.fullmatch(source_ref)
    target_match = SAT_IN_RE.fullmatch(target_ref)
    if not source_match or not target_match:
        return None

    source_step = int(source_match.group(1))
    source_port = int(source_match.group(2))
    consumer_index = int(target_match.group(1))
    input_port = int(target_match.group(2))
    consumer_step = consumer_index + 1

    if source_step == 0:
        source_key = f"wf_input_{source_port}"
    else:
        source_key = f"t{source_step}:out{source_port}"
    return f"t{consumer_step}:in{input_port}<-{source_key}"


def parse_snake_port_index(raw_numeric: str, raw_symbolic: str) -> int | None:
    if raw_numeric:
        return int(raw_numeric)
    if not raw_symbolic:
        return None
    match = LEGACY_PORT_SUFFIX_RE.search(raw_symbolic)
    if match is not None:
        return int(match.group(1))
    match = STRUCTURED_PORT_SUFFIX_RE.search(raw_symbolic)
    if match is not None:
        return int(match.group(1))
    return None


def normalize_snake_bindings(block: str) -> tuple[str, ...]:
    bindings = []
    for time_str, port_str, input_index in SNAKE_BIND_WF_RE.findall(block):
        bindings.append(f"t{int(time_str)}:in{int(port_str)}<-wf_input_{int(input_index)}")
    for time_str, port_str, source_time_str, source_port_numeric, source_port_symbolic in SNAKE_BIND_OUT_RE.findall(block):
        source_port = parse_snake_port_index(source_port_numeric, source_port_symbolic)
        if source_port is None:
            continue
        bindings.append(
            f"t{int(time_str)}:in{int(port_str)}<-t{int(source_time_str)}:out{source_port}"
        )
    return tuple(sorted(set(bindings)))


def normalize_snake_outputs(block: str) -> tuple[str, ...]:
    outputs = set()
    for step_str, port_numeric, port_symbolic in SNAKE_HOLDS_OUT_RE.findall(block):
        step = int(step_str)
        if step < 1:
            continue
        port_index = parse_snake_port_index(port_numeric, port_symbolic)
        if port_index is None:
            continue
        outputs.add(f"t{step}:out{port_index}")
    return tuple(sorted(outputs))


def normalize_sat_outputs(block: str) -> tuple[str, ...]:
    outputs = set()
    for line in block.splitlines():
        line = line.strip()
        if (
            not line
            or line.startswith("empty(")
            or line.startswith("memRef(")
            or line.startswith("r_rel(")
            or line.startswith("is_rel(")
            or line.startswith("varValue(")
            or line.startswith("emptyLabel(")
        ):
            continue
        for step_str, port_str in SAT_OUT_RE.findall(line):
            step = int(step_str)
            if step >= 1:
                outputs.add(f"t{step}:out{int(port_str)}")
    return tuple(sorted(outputs))


def split_sat_blocks(text: str) -> list[str]:
    blocks = [block.strip() for block in SPLIT_BLANK_RE.split(text) if block.strip()]
    if blocks:
        return blocks
    stripped = text.strip()
    return [stripped] if stripped else []


def parse_sat_complete(path: Path, tool_catalog: dict[str, str]) -> ParsedSolutions:
    text = path.read_text(encoding="utf-8")
    blocks = split_sat_blocks(text)
    solutions = []
    for index, block in enumerate(blocks, start=1):
        tool_candidates: dict[int, list[str]] = defaultdict(list)
        raw_bindings = set()
        outputs = ()

        for line in block.splitlines():
            line = line.strip()
            if not line:
                continue

            tool_match = SAT_TOOL_RE.match(line)
            if tool_match:
                tool_index = int(tool_match.group("index"))
                tool_candidates[tool_index].append(tool_match.group("predicate"))

            bind_match = SAT_BIND_RE.match(line)
            if bind_match:
                binding = normalize_sat_binding(bind_match.group("src"), bind_match.group("dst"))
                if binding is not None:
                    raw_bindings.add(binding)

        tool_sequence = []
        for tool_index in sorted(tool_candidates):
            chosen = choose_sat_tool_name(tool_candidates[tool_index], tool_catalog)
            if chosen is not None:
                tool_sequence.append(chosen)

        max_consumer_step = len(tool_sequence)
        bindings = []
        for binding in sorted(raw_bindings):
            match = NORMALIZED_BIND_RE.match(binding)
            if match and int(match.group(1)) <= max_consumer_step:
                bindings.append(binding)
        outputs = normalize_sat_outputs(block)

        solutions.append(
            NormalizedSolution(
                index=index,
                parser_kind="sat-complete",
                tool_sequence=tuple(tool_sequence),
                bindings=tuple(bindings),
                outputs=outputs,
            )
        )
    return ParsedSolutions(parser_kind="sat-complete", path=path, solutions=tuple(solutions))


def canonicalize_snake_tool_name(tool_name: str, tool_catalog: dict[str, str]) -> str:
    stripped = strip_snake_suffix(tool_name)
    normalized = normalize_name(stripped)
    return tool_catalog.get(normalized, stripped)


def parse_snake_answer_sets(path: Path, tool_catalog: dict[str, str]) -> ParsedSolutions:
    solutions = []
    current_index = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_index, current_lines
        if current_index is None:
            return
        block = " ".join(line.strip() for line in current_lines if line.strip())
        tools = {
            int(time_str): canonicalize_snake_tool_name(tool_name, tool_catalog)
            for time_str, tool_name in SNAKE_TOOL_RE.findall(block)
        }
        tool_sequence = tuple(tool_name for _, tool_name in sorted(tools.items()))
        solutions.append(
            NormalizedSolution(
                index=current_index,
                parser_kind="snake-answer-sets",
                tool_sequence=tool_sequence,
                bindings=normalize_snake_bindings(block),
                outputs=normalize_snake_outputs(block),
            )
        )
        current_index = None
        current_lines = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        header_match = ANSWER_SET_RE.match(line)
        if header_match:
            flush()
            current_index = int(header_match.group(1))
            current_lines = []
            continue
        if current_index is not None:
            current_lines.append(line)
    flush()
    return ParsedSolutions(parser_kind="snake-answer-sets", path=path, solutions=tuple(solutions))


def parse_workflow_summary(path: Path) -> ParsedSolutions:
    solutions = []
    pending_index = 0
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        inline_match = SOLUTION_INLINE_RE.match(line)
        if inline_match:
            pending_index += 1
            tools = tuple(tool.strip() for tool in inline_match.group(1).split("->") if tool.strip())
            solutions.append(
                NormalizedSolution(
                    index=pending_index,
                    parser_kind="summary",
                    tool_sequence=tools,
                    bindings=(),
                    outputs=(),
                )
            )
            continue

        workflow_match = WORKFLOW_LINE_RE.match(line)
        if workflow_match:
            pending_index += 1
            text = workflow_match.group(1).strip()
            separator = "->" if "->" in text else ","
            tools = tuple(tool.strip() for tool in text.split(separator) if tool.strip())
            solutions.append(
                NormalizedSolution(
                    index=pending_index,
                    parser_kind="summary",
                    tool_sequence=tools,
                    bindings=(),
                    outputs=(),
                )
            )
            continue

        tools_match = TOOLS_LINE_RE.match(line)
        if tools_match:
            pending_index += 1
            tools = tuple(tool.strip() for tool in tools_match.group(1).split("->") if tool.strip())
            solutions.append(
                NormalizedSolution(
                    index=pending_index,
                    parser_kind="summary",
                    tool_sequence=tools,
                    bindings=(),
                    outputs=(),
                )
            )
    return ParsedSolutions(parser_kind="summary", path=path, solutions=tuple(solutions))


def detect_parser_kind(path: Path) -> str:
    head = path.read_text(encoding="utf-8")[:20000]
    if "Answer Set 1" in head or "tool_at_time(" in head or "ape_bind(" in head:
        return "snake-answer-sets"
    if "No answer sets found." in head:
        return "snake-answer-sets"
    if "memRef(" in head:
        return "sat-complete"
    if "Workflow" in head or "Tools:" in head or "Solution 1:" in head or "Solution 1\n" in head:
        return "summary"
    if "No solutions found." in head:
        return "summary"
    raise ValueError(f"Unable to detect solution format for {path}")


def parse_any(path: Path, config_path: Path | None = None) -> ParsedSolutions:
    parser_kind = detect_parser_kind(path)
    if config_path is None:
        config_path = infer_config_path(path)
    tool_catalog = load_tool_catalog(config_path)
    if parser_kind == "snake-answer-sets":
        return parse_snake_answer_sets(path, tool_catalog)
    if parser_kind == "summary":
        return parse_workflow_summary(path)

    return parse_sat_complete(path, tool_catalog)


def format_sequence(sequence: tuple[str, ...]) -> str:
    return " -> ".join(sequence) if sequence else "<empty>"


def count_matching_candidates(
    left_tool_counter: Counter[tuple[str, ...]],
    right_tool_counter: Counter[tuple[str, ...]],
) -> int:
    return sum(
        min(left_tool_counter[key], right_tool_counter[key])
        for key in set(left_tool_counter) & set(right_tool_counter)
    )


def build_tool_sequence_differences(
    left_tool_counter: Counter[tuple[str, ...]],
    right_tool_counter: Counter[tuple[str, ...]],
    *,
    left_name: str,
    right_name: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    count_differences = []
    left_only_sequences = []
    right_only_sequences = []

    for tool_sequence in set(left_tool_counter) | set(right_tool_counter):
        left_count = left_tool_counter.get(tool_sequence, 0)
        right_count = right_tool_counter.get(tool_sequence, 0)
        entry = {
            "tool_sequence": list(tool_sequence),
            "tool_sequence_text": format_sequence(tool_sequence),
        }
        if left_count and right_count:
            if left_count != right_count:
                count_differences.append(
                    {
                        **entry,
                        "delta": abs(left_count - right_count),
                        left_name: left_count,
                        right_name: right_count,
                    }
                )
        elif left_count:
            left_only_sequences.append({**entry, "count": left_count})
        else:
            right_only_sequences.append({**entry, "count": right_count})

    count_differences.sort(key=lambda item: (-item["delta"], item["tool_sequence_text"]))
    left_only_sequences.sort(key=lambda item: (-item["count"], item["tool_sequence_text"]))
    right_only_sequences.sort(key=lambda item: (-item["count"], item["tool_sequence_text"]))
    return count_differences, left_only_sequences, right_only_sequences


def describe_length_filter(length: int | None) -> str:
    if length is None:
        return "all workflow lengths"
    return f"workflow length {length}"


def describe_scope(length: int | None, tool_sequence: tuple[str, ...] | None) -> str:
    scope = describe_length_filter(length)
    if tool_sequence is None:
        return scope
    return f"{scope}, tool sequence {format_sequence(tool_sequence)}"


def format_bindings(bindings: tuple[str, ...]) -> str:
    return ", ".join(bindings) if bindings else "<none>"


def format_outputs(outputs: tuple[str, ...]) -> str:
    return ", ".join(outputs) if outputs else "<none>"


def _split_binding(binding: str) -> tuple[str, str]:
    consumer, source = binding.split("<-", 1)
    return consumer, source


def describe_workflow_difference(
    left_bindings: tuple[str, ...],
    right_bindings: tuple[str, ...],
    left_outputs: tuple[str, ...],
    right_outputs: tuple[str, ...],
) -> str:
    if left_outputs != right_outputs:
        return "Different produced outputs; this is a real workflow-behavior difference."

    left_only = sorted(set(left_bindings) - set(right_bindings))
    right_only = sorted(set(right_bindings) - set(left_bindings))
    if not left_only and not right_only:
        return "No strict-signature difference remains after normalization."

    if len(left_only) == len(right_only):
        left_consumers = [consumer for consumer, _ in map(_split_binding, left_only)]
        right_consumers = [consumer for consumer, _ in map(_split_binding, right_only)]
        if left_consumers == right_consumers:
            return (
                "Same tool sequence and same outputs; only artifact provenance/binding choices differ."
            )

        left_steps = [consumer.split(":")[0] for consumer in left_consumers]
        right_steps = [consumer.split(":")[0] for consumer in right_consumers]
        if left_steps == right_steps:
            return (
                "Same tool sequence and same outputs; bindings differ within the same consumer steps."
            )

    return "Same tool sequence, but the normalized binding signature still differs."


def print_bucket_interpretation(
    left_name: str,
    right_name: str,
    exact_matches: int,
    left_only_same_tool: int,
    right_only_same_tool: int,
    left_only_new_tools: int,
    right_only_new_tools: int,
) -> None:
    print("\n--- Interpretation ---")
    print(
        f"Exact workflow parity: {exact_matches} workflows are identical after normalization."
    )
    print(
        f"Strict-signature mismatch on shared tool sequences: {left_only_same_tool}/{right_only_same_tool}."
    )
    print(
        "These have the same workflow skeleton but differ in bindings, provenance, or produced outputs."
    )
    print(
        f"Tool-sequence-only mismatch: {left_only_new_tools}/{right_only_new_tools}."
    )
    print(
        "These indicate one side contains workflows whose tool order does not appear at all on the other side."
    )

    if left_only_new_tools == 0 and right_only_new_tools == 0:
        print("Practical meaning: workflow coverage matches; any remaining gap is below the tool-sequence level.")
    elif left_only_same_tool == 0 and right_only_same_tool == 0:
        print("Practical meaning: the two files agree exactly on normalized workflows.")
    else:
        print("Practical meaning: investigate the strict-signature samples below to determine whether the gap is semantic or only provenance ordering.")


def print_single_summary(parsed: ParsedSolutions, sample_limit: int) -> None:
    print(f"File: {parsed.path}")
    print(f"Format: {parsed.parser_kind}")
    print(f"Solutions parsed: {len(parsed.solutions)}")
    print(f"Unique tool sequences: {len(parsed.tool_counter)}")

    if not parsed.solutions:
        return

    print("\n--- Interpretation ---")
    print("This script compares normalized tool sequences only.")
    if parsed.parser_kind != "summary":
        print("Bindings and outputs were parsed, but they are ignored for APE-vs-ASP matching.")

    print("\nTop tool sequences:")
    for tool_sequence, count in parsed.tool_counter.most_common(sample_limit):
        print(f"  {count:>6}  {format_sequence(tool_sequence)}")


def sample_workflow_key(key: tuple[object, ...]) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    tool_sequence, bindings, outputs = key
    return tool_sequence, bindings, outputs


def serialize_solution(solution: NormalizedSolution) -> dict[str, object]:
    return {
        "index": solution.index,
        "tool_sequence": list(solution.tool_sequence),
        "tool_sequence_text": format_sequence(solution.tool_sequence),
        "bindings": list(solution.bindings),
        "outputs": list(solution.outputs),
    }


def serialize_workflow_key(
    key: tuple[object, ...],
    *,
    count: int,
) -> dict[str, object]:
    tool_sequence, bindings, outputs = sample_workflow_key(key)
    return {
        "count": count,
        "tool_sequence": list(tool_sequence),
        "tool_sequence_text": format_sequence(tool_sequence),
        "bindings": list(bindings),
        "outputs": list(outputs),
    }


def build_single_summary_report(
    parsed: ParsedSolutions,
    *,
    sample_limit: int,
    length: int | None,
    tool_sequence: tuple[str, ...] | None,
) -> dict[str, object]:
    report = {
        "path": str(parsed.path),
        "format": parsed.parser_kind,
        "scope": describe_scope(length, tool_sequence),
        "solutions_parsed": len(parsed.solutions),
        "comparison_mode": "tool-sequence",
        "unique_tool_sequences": len(parsed.tool_counter),
        "top_tool_sequences": [
            {
                "count": count,
                "tool_sequence": list(sequence),
                "tool_sequence_text": format_sequence(sequence),
            }
            for sequence, count in parsed.tool_counter.most_common(sample_limit)
        ],
    }
    return report


def build_comparison_report(
    left: ParsedSolutions,
    right: ParsedSolutions,
    *,
    left_name: str,
    right_name: str,
    sample_limit: int,
    length: int | None,
    tool_sequence: tuple[str, ...] | None,
) -> dict[str, object]:
    left_tool_counter = left.tool_counter
    right_tool_counter = right.tool_counter
    matching_candidates = count_matching_candidates(left_tool_counter, right_tool_counter)
    count_differences, left_only_sequences, right_only_sequences = build_tool_sequence_differences(
        left_tool_counter,
        right_tool_counter,
        left_name=left_name,
        right_name=right_name,
    )
    left_unmatched_candidates = sum(
        max(left_tool_counter[key] - right_tool_counter.get(key, 0), 0)
        for key in left_tool_counter
    )
    right_unmatched_candidates = sum(
        max(right_tool_counter[key] - left_tool_counter.get(key, 0), 0)
        for key in right_tool_counter
    )

    return {
        "left": {
            "name": left_name,
            "path": str(left.path),
            "format": left.parser_kind,
        },
        "right": {
            "name": right_name,
            "path": str(right.path),
            "format": right.parser_kind,
        },
        "scope": describe_scope(length, tool_sequence),
        "comparison_mode": "tool-sequence",
        "matches": left_tool_counter == right_tool_counter,
        "counts": {
            f"{left_name}_total_solutions": len(left.solutions),
            f"{right_name}_total_solutions": len(right.solutions),
            f"{left_name}_unique_tool_sequences": len(left_tool_counter),
            f"{right_name}_unique_tool_sequences": len(right_tool_counter),
            "matching_candidates": matching_candidates,
            f"{left_name}_unmatched_candidates": left_unmatched_candidates,
            f"{right_name}_unmatched_candidates": right_unmatched_candidates,
        },
        "count_differences": count_differences[:sample_limit],
        "left_only_sequences": left_only_sequences[:sample_limit],
        "right_only_sequences": right_only_sequences[:sample_limit],
    }


def print_comparison(
    left: ParsedSolutions,
    right: ParsedSolutions,
    left_name: str,
    right_name: str,
    sample_limit: int,
    length: int | None,
    tool_sequence: tuple[str, ...] | None,
) -> None:
    left_tool_counter = left.tool_counter
    right_tool_counter = right.tool_counter
    matching_candidates = count_matching_candidates(left_tool_counter, right_tool_counter)
    count_differences, left_only_sequences, right_only_sequences = build_tool_sequence_differences(
        left_tool_counter,
        right_tool_counter,
        left_name=left_name,
        right_name=right_name,
    )
    left_unmatched_candidates = sum(
        max(left_tool_counter[key] - right_tool_counter.get(key, 0), 0)
        for key in left_tool_counter
    )
    right_unmatched_candidates = sum(
        max(right_tool_counter[key] - left_tool_counter.get(key, 0), 0)
        for key in right_tool_counter
    )
    matches = left_tool_counter == right_tool_counter

    print(f"{left_name}: {left.path} [{left.parser_kind}]")
    print(f"{right_name}: {right.path} [{right.parser_kind}]")
    print(f"Scope: {describe_scope(length, tool_sequence)}")
    print("Comparison mode: normalized tool sequences")
    print("\n--- Counts ---")
    print(f"{left_name} total solutions: {len(left.solutions)}")
    print(f"{right_name} total solutions: {len(right.solutions)}")
    print(f"{left_name} unique tool sequences: {len(left_tool_counter)}")
    print(f"{right_name} unique tool sequences: {len(right_tool_counter)}")
    print("\n--- Match Result ---")
    print(f"Exact normalized tool-sequence multiset match: {'yes' if matches else 'no'}")
    print(f"Matching candidates across both files: {matching_candidates}")
    print(f"{left_name} unmatched candidates: {left_unmatched_candidates}")
    print(f"{right_name} unmatched candidates: {right_unmatched_candidates}")

    print("\n--- Interpretation ---")
    if matches:
        print("The APE workflow candidates and ASP answer sets match exactly by ordered tool sequence.")
    else:
        print("The APE workflow candidates and ASP answer sets do not match by ordered tool sequence.")
        print("Bindings and produced outputs are ignored; only the normalized tool sequence counts matter here.")

    print("\n--- Shared Sequences With Different Counts ---")
    if count_differences:
        for entry in count_differences[:sample_limit]:
            print(
                f"  {entry[left_name]:>6} vs {entry[right_name]:>6}  "
                f"{entry['tool_sequence_text']}"
            )
    else:
        print("  None.")

    print(f"\n--- {left_name}-Only Sequences ---")
    if left_only_sequences:
        for entry in left_only_sequences[:sample_limit]:
            print(f"  {entry['count']:>6}  {entry['tool_sequence_text']}")
    else:
        print("  None.")

    print(f"\n--- {right_name}-Only Sequences ---")
    if right_only_sequences:
        for entry in right_only_sequences[:sample_limit]:
            print(f"  {entry['count']:>6}  {entry['tool_sequence_text']}")
    else:
        print("  None.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize and compare APE SAT complete solutions, snakeAPE answer sets, "
            "or older workflow summary files."
        )
    )
    parser.add_argument("left", help="Path to the first solutions file.")
    parser.add_argument("right", nargs="?", help="Optional second solutions file to compare against.")
    parser.add_argument(
        "--left-config",
        help="Optional config.json/config_sat.json used to resolve concrete tool names for the left file.",
    )
    parser.add_argument(
        "--right-config",
        help="Optional config.json/config_sat.json used to resolve concrete tool names for the right file.",
    )
    parser.add_argument(
        "--left-name",
        default="Left",
        help="Display name for the left file in comparison output.",
    )
    parser.add_argument(
        "--right-name",
        default="Right",
        help="Display name for the right file in comparison output.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=5,
        help="How many representative mismatches or sequence counts to print.",
    )
    parser.add_argument(
        "--length",
        type=int,
        help="Optional workflow length filter applied after parsing.",
    )
    parser.add_argument(
        "--tool-sequence",
        help='Optional exact tool-sequence filter after normalization, e.g. "Tool A -> Tool B".',
    )
    parser.add_argument(
        "--json-out",
        help="Optional path for a structured JSON report.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    tool_sequence = parse_tool_sequence_filter(args.tool_sequence) if args.tool_sequence else None

    left_path = Path(args.left).resolve()
    left_config = Path(args.left_config).resolve() if args.left_config else None
    left_parsed = filter_by_tool_sequence(
        filter_by_length(parse_any(left_path, left_config), args.length),
        tool_sequence,
    )

    if not args.right:
        print_single_summary(left_parsed, args.sample_limit)
        if args.json_out:
            Path(args.json_out).resolve().write_text(
                json.dumps(
                    build_single_summary_report(
                        left_parsed,
                        sample_limit=args.sample_limit,
                        length=args.length,
                        tool_sequence=tool_sequence,
                    ),
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        return 0

    right_path = Path(args.right).resolve()
    right_config = Path(args.right_config).resolve() if args.right_config else None
    right_parsed = filter_by_tool_sequence(
        filter_by_length(parse_any(right_path, right_config), args.length),
        tool_sequence,
    )
    print_comparison(
        left_parsed,
        right_parsed,
        args.left_name,
        args.right_name,
        args.sample_limit,
        args.length,
        tool_sequence,
    )
    if args.json_out:
        Path(args.json_out).resolve().write_text(
            json.dumps(
                build_comparison_report(
                    left_parsed,
                    right_parsed,
                    left_name=args.left_name,
                    right_name=args.right_name,
                    sample_limit=args.sample_limit,
                    length=args.length,
                    tool_sequence=tool_sequence,
                ),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
