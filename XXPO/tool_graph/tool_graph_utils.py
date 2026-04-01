#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

START_NODE = "__START__"
NO_TOOL_NODE = "__NO_TOOL__"


def _normalize_termination_reason(reason: str) -> str:
    value = (reason or "unknown").strip()
    if value.startswith("TerminationReason."):
        value = value.split(".", 1)[1]
    value = value.lower().replace(" ", "_")
    return value or "unknown"


def terminal_node_from_reason(reason: str) -> str:
    return f"__END_{_normalize_termination_reason(reason).upper()}__"


def extract_tool_sequence_from_tool_steps(
    tool_steps: list[dict[str, Any]],
) -> list[str]:
    seq: list[str] = []
    for step in tool_steps or []:
        for call in step.get("tool_calls", []):
            name = call.get("name")
            if isinstance(name, str) and name.strip():
                seq.append(name.strip())
    return seq


def extract_edges_from_sequence(
    sequence: list[str],
    terminal_reason: str,
) -> list[tuple[str, str]]:
    edges: list[tuple[str, str]] = []
    terminal = terminal_node_from_reason(terminal_reason)

    if not sequence:
        edges.append((START_NODE, NO_TOOL_NODE))
        edges.append((NO_TOOL_NODE, terminal))
        return edges

    prev = START_NODE
    for tool in sequence:
        edges.append((prev, tool))
        prev = tool
    edges.append((prev, terminal))
    return edges


def _normalize_status_filter(status_filter: str | None) -> str:
    if status_filter is None:
        return "all"
    value = status_filter.strip().lower()
    if value in {"all", "ok", "error"}:
        return value
    raise ValueError(
        f"Unsupported status_filter={status_filter!r}. Use one of: all, ok, error"
    )


def _status_matches(record_status: str, status_filter: str) -> bool:
    if status_filter == "all":
        return True
    if status_filter == "ok":
        return record_status == "ok"
    # error: everything not marked as ok
    return record_status != "ok"


def build_tool_graph_from_records(
    records: list[dict[str, Any]],
    min_edge_count: int = 1,
    status_filter: str | None = None,
) -> dict[str, Any]:
    """
    Build an aggregated directed tool graph from rollout records.

    Args:
        records: Rollout records containing at least `status`, `tool_steps`, and
            `termination_reason` fields.
        min_edge_count: Keep only edges with count >= min_edge_count.
            Set to 1 to keep all observed edges.
        status_filter: One of {"all", "ok", "error"}.
            - all: include all records
            - ok: include only successful records
            - error: include only failed records
    """
    normalized_filter = _normalize_status_filter(status_filter)

    success_records = [r for r in records if r.get("status") == "ok"]
    selected_records = [
        r
        for r in records
        if _status_matches(str(r.get("status", "error")), normalized_filter)
    ]

    node_counter: Counter[str] = Counter()
    edge_counter: Counter[tuple[str, str]] = Counter()

    for record in selected_records:
        seq = extract_tool_sequence_from_tool_steps(record.get("tool_steps", []))
        term = str(record.get("termination_reason", "unknown"))
        edges = extract_edges_from_sequence(seq, term)
        for src, dst in edges:
            node_counter[src] += 1
            node_counter[dst] += 1
            edge_counter[(src, dst)] += 1

    out_degree: dict[str, int] = defaultdict(int)
    for (src, _dst), count in edge_counter.items():
        out_degree[src] += count

    threshold = max(1, int(min_edge_count))
    edges = []
    for (src, dst), count in sorted(
        edge_counter.items(), key=lambda x: (-x[1], x[0][0], x[0][1])
    ):
        if count < threshold:
            continue
        p_src = count / out_degree[src] if out_degree[src] > 0 else 0.0
        edges.append(
            {
                "src": src,
                "dst": dst,
                "count": int(count),
                "p_src": float(p_src),
            }
        )

    nodes = []
    for node, count in sorted(node_counter.items(), key=lambda x: (-x[1], x[0])):
        if node == START_NODE:
            kind = "start"
        elif node == NO_TOOL_NODE:
            kind = "no_tool"
        elif node.startswith("__END_"):
            kind = "end"
        else:
            kind = "tool"
        nodes.append({"id": node, "count": int(count), "kind": kind})

    top_tools = []
    for node, count in sorted(node_counter.items(), key=lambda x: (-x[1], x[0])):
        if node.startswith("__"):
            continue
        top_tools.append({"tool": node, "count": int(count)})

    return {
        "status_filter": normalized_filter,
        "min_edge_count": threshold,
        "num_records": len(records),
        "num_success_records": len(success_records),
        "num_selected_records": len(selected_records),
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "nodes": nodes,
        "edges": edges,
        "top_tools": top_tools[:50],
    }
