#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from tool_graph_utils import extract_edges_from_sequence, extract_tool_sequence_from_tool_steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score one rollout trace at step-level using tool graphs. "
            "Supports success/failure graph scoring for training-style rewards."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Input rollout data path. Supported: "
            "(1) flattened records JSONL, "
            "(2) tau2 results.json, "
            "(3) tau2 run directory containing results.json."
        ),
    )
    parser.add_argument(
        "--input-format",
        choices=["auto", "records_jsonl", "tau2_results"],
        default="auto",
        help="Input format. Default auto-detect.",
    )

    parser.add_argument(
        "--task-id",
        default=None,
        help="Optional task_id filter.",
    )
    parser.add_argument(
        "--simulation-id",
        default=None,
        help="Optional simulation_id filter.",
    )
    parser.add_argument(
        "--trial",
        type=int,
        default=None,
        help="Optional trial filter.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed filter.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index within filtered candidates. Default 0.",
    )

    parser.add_argument(
        "--graph-success",
        default=None,
        help="Path to success graph JSON.",
    )
    parser.add_argument(
        "--graph-failure",
        default=None,
        help="Path to failure graph JSON.",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "log_odds", "success_only", "failure_only"],
        default="auto",
        help=(
            "Scoring mode. auto: log_odds if both graphs are given, otherwise "
            "success_only / failure_only."
        ),
    )

    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.5,
        help=(
            "Additive smoothing for edge probability lookup from graph counts. "
            "Set 0 to disable smoothing."
        ),
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="Numerical floor for probabilities before log().",
    )
    parser.add_argument(
        "--assign-terminal-to-last-step",
        action="store_true",
        help="If set, add terminal edge score to the last tool step score.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSON path. If omitted, print JSON to stdout.",
    )
    return parser.parse_args()


def _parse_json_obj(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_tool_steps(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")

        if role == "assistant" and message.get("tool_calls"):
            steps.append(
                {
                    "assistant_turn_idx": message.get("turn_idx"),
                    "tool_calls": message.get("tool_calls", []),
                    "tool_results": [],
                }
            )
            continue

        if role == "tool":
            result = {
                "tool_call_id": message.get("id") or message.get("tool_call_id"),
                "requestor": message.get("requestor"),
                "error": bool(message.get("error", False)),
                "content": message.get("content"),
                "turn_idx": message.get("turn_idx"),
            }
            attached = False
            for step in reversed(steps):
                call_ids = {call.get("id", "") for call in step["tool_calls"]}
                if result["tool_call_id"] in call_ids:
                    step["tool_results"].append(result)
                    attached = True
                    break
            if not attached:
                steps.append(
                    {
                        "assistant_turn_idx": None,
                        "tool_calls": [],
                        "tool_results": [result],
                    }
                )
    return steps


def _extract_reward(sim: dict[str, Any]) -> float | None:
    reward_info = sim.get("reward_info")
    if isinstance(reward_info, dict):
        reward = reward_info.get("reward")
        if isinstance(reward, (int, float)):
            return float(reward)
    return None


def _status_from_sim(sim: dict[str, Any], success_threshold: float = 1.0) -> str:
    reward = _extract_reward(sim)
    if reward is not None:
        return "ok" if reward >= success_threshold else "error"
    term = str(sim.get("termination_reason", "")).lower()
    return "ok" if term == "agent_stop" else "error"


def _resolve_tau2_results_path(path: Path) -> Path:
    if path.is_dir():
        p = path / "results.json"
        if not p.exists():
            raise FileNotFoundError(f"results.json not found in directory: {path}")
        return p
    return path


def _load_records_from_tau2_results(path: Path) -> list[dict[str, Any]]:
    payload = _parse_json_obj(path)
    simulations = payload.get("simulations", []) if isinstance(payload, dict) else []
    if not isinstance(simulations, list):
        raise ValueError(f"Invalid tau2 results format: {path}")

    records: list[dict[str, Any]] = []
    for sim in simulations:
        if not isinstance(sim, dict):
            continue
        messages = sim.get("messages", [])
        if not isinstance(messages, list):
            messages = []

        records.append(
            {
                "status": _status_from_sim(sim),
                "task_id": sim.get("task_id"),
                "simulation_id": sim.get("id"),
                "trial": sim.get("trial"),
                "seed": sim.get("seed"),
                "termination_reason": str(sim.get("termination_reason", "unknown")),
                "reward": _extract_reward(sim),
                "tool_steps": _build_tool_steps(messages),
                "messages": messages,
                "source_results": str(path),
            }
        )
    return records


def _load_records_from_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if isinstance(rec, dict):
                records.append(rec)
    return records


def _load_records(input_path: Path, input_format: str) -> list[dict[str, Any]]:
    if input_format == "records_jsonl":
        return _load_records_from_jsonl(input_path)

    if input_format == "tau2_results":
        tau2_path = _resolve_tau2_results_path(input_path)
        return _load_records_from_tau2_results(tau2_path)

    # auto detection
    if input_path.suffix.lower() == ".jsonl":
        return _load_records_from_jsonl(input_path)

    json_path = _resolve_tau2_results_path(input_path)
    payload = _parse_json_obj(json_path)
    if isinstance(payload, dict) and "simulations" in payload:
        return _load_records_from_tau2_results(json_path)

    if isinstance(payload, dict) and "tool_steps" in payload:
        return [payload]

    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return payload

    raise ValueError(
        f"Cannot auto-detect input format for: {input_path}. "
        "Use --input-format explicitly."
    )


def _select_record(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    selected = records

    if args.simulation_id is not None:
        selected = [
            r for r in selected if str(r.get("simulation_id", "")) == str(args.simulation_id)
        ]
    if args.task_id is not None:
        selected = [r for r in selected if str(r.get("task_id", "")) == str(args.task_id)]
    if args.trial is not None:
        selected = [r for r in selected if int(r.get("trial", -1)) == int(args.trial)]
    if args.seed is not None:
        selected = [r for r in selected if int(r.get("seed", -1)) == int(args.seed)]

    if not selected:
        raise ValueError("No rollout record matched the provided filters.")

    if args.index < 0 or args.index >= len(selected):
        raise IndexError(
            f"index={args.index} out of range for {len(selected)} matched records."
        )

    return selected[args.index]


def _load_graph(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path).expanduser().resolve()
    data = _parse_json_obj(p)
    if not isinstance(data, dict):
        raise ValueError(f"Graph file must be a JSON object: {p}")
    return data


def _graph_index(graph: dict[str, Any] | None) -> dict[str, Any] | None:
    if graph is None:
        return None

    edge_counts: dict[tuple[str, str], int] = {}
    out_degree: dict[str, int] = defaultdict(int)
    out_neighbors: dict[str, set[str]] = defaultdict(set)

    for edge in graph.get("edges", []):
        if not isinstance(edge, dict):
            continue
        src = str(edge.get("src", ""))
        dst = str(edge.get("dst", ""))
        if not src or not dst:
            continue
        count = int(edge.get("count", 0) or 0)
        if count < 0:
            count = 0
        edge_counts[(src, dst)] = count
        out_degree[src] += count
        out_neighbors[src].add(dst)

    return {
        "edge_counts": edge_counts,
        "out_degree": dict(out_degree),
        "out_neighbors": {k: v for k, v in out_neighbors.items()},
    }


def _edge_prob(
    gidx: dict[str, Any] | None,
    src: str,
    dst: str,
    smoothing: float,
    eps: float,
) -> float | None:
    if gidx is None:
        return None

    edge_counts = gidx["edge_counts"]
    out_degree = gidx["out_degree"]
    out_neighbors = gidx["out_neighbors"]

    c = edge_counts.get((src, dst), 0)
    out = int(out_degree.get(src, 0))
    vocab = len(out_neighbors.get(src, set()))

    if out <= 0:
        # Unknown source in graph: return tiny prior.
        return float(eps)

    if smoothing <= 0:
        p = c / out if c > 0 else 0.0
        return float(max(p, eps))

    # Additive smoothing with an extra unseen bucket.
    denom = out + smoothing * (vocab + 1)
    p = (c + smoothing) / denom
    return float(max(p, eps))


def _resolve_mode(mode: str, has_success: bool, has_failure: bool) -> str:
    if mode != "auto":
        return mode
    if has_success and has_failure:
        return "log_odds"
    if has_success:
        return "success_only"
    if has_failure:
        return "failure_only"
    raise ValueError("At least one graph must be provided: --graph-success/--graph-failure")


def _edge_score(
    mode: str,
    p_success: float | None,
    p_failure: float | None,
    eps: float,
) -> float:
    if mode == "log_odds":
        if p_success is None or p_failure is None:
            raise ValueError("log_odds mode requires both success and failure graphs")
        return math.log(max(p_success, eps)) - math.log(max(p_failure, eps))
    if mode == "success_only":
        if p_success is None:
            raise ValueError("success_only mode requires --graph-success")
        return math.log(max(p_success, eps))
    if mode == "failure_only":
        if p_failure is None:
            raise ValueError("failure_only mode requires --graph-failure")
        return -math.log(max(p_failure, eps))
    raise ValueError(f"Unsupported mode: {mode}")


def _edge_kind(i: int, edges: list[tuple[str, str]], sequence_len: int) -> str:
    if sequence_len == 0:
        return "no_tool"
    if i == len(edges) - 1:
        return "terminal"
    if i == 0:
        return "start_to_tool"
    return "tool_to_tool"


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    records = _load_records(input_path, args.input_format)
    if not records:
        raise ValueError(f"No records loaded from: {input_path}")

    record = _select_record(records, args)
    tool_steps = record.get("tool_steps", []) or []
    sequence = extract_tool_sequence_from_tool_steps(tool_steps)
    term_reason = str(record.get("termination_reason", "unknown"))
    edges = extract_edges_from_sequence(sequence, term_reason)

    success_graph = _load_graph(args.graph_success)
    failure_graph = _load_graph(args.graph_failure)
    success_idx = _graph_index(success_graph)
    failure_idx = _graph_index(failure_graph)

    mode = _resolve_mode(
        args.mode,
        has_success=success_idx is not None,
        has_failure=failure_idx is not None,
    )

    edge_rows: list[dict[str, Any]] = []
    for i, (src, dst) in enumerate(edges):
        p_success = _edge_prob(success_idx, src, dst, args.smoothing, args.eps)
        p_failure = _edge_prob(failure_idx, src, dst, args.smoothing, args.eps)
        score = _edge_score(mode, p_success, p_failure, args.eps)
        edge_rows.append(
            {
                "edge_index": i,
                "kind": _edge_kind(i, edges, len(sequence)),
                "src": src,
                "dst": dst,
                "p_success": p_success,
                "p_failure": p_failure,
                "score": score,
            }
        )

    tool_step_rows: list[dict[str, Any]] = []
    for i, tool_name in enumerate(sequence):
        incoming_edge = edge_rows[i]
        tool_step_rows.append(
            {
                "tool_step_index": i,
                "tool": tool_name,
                "incoming_edge": {
                    "src": incoming_edge["src"],
                    "dst": incoming_edge["dst"],
                    "kind": incoming_edge["kind"],
                },
                "score": incoming_edge["score"],
            }
        )

    terminal_edge = edge_rows[-1]
    if args.assign_terminal_to_last_step and tool_step_rows:
        tool_step_rows[-1]["score"] = float(tool_step_rows[-1]["score"] + terminal_edge["score"])
        tool_step_rows[-1]["terminal_added"] = True

    result = {
        "trace_selector": {
            "input": str(input_path),
            "input_format": args.input_format,
            "task_id": record.get("task_id"),
            "simulation_id": record.get("simulation_id"),
            "trial": record.get("trial"),
            "seed": record.get("seed"),
            "status": record.get("status"),
            "reward": record.get("reward"),
            "termination_reason": term_reason,
        },
        "scoring_config": {
            "mode": mode,
            "smoothing": args.smoothing,
            "eps": args.eps,
            "assign_terminal_to_last_step": bool(args.assign_terminal_to_last_step),
            "graph_success": str(Path(args.graph_success).expanduser().resolve())
            if args.graph_success
            else None,
            "graph_failure": str(Path(args.graph_failure).expanduser().resolve())
            if args.graph_failure
            else None,
        },
        "tool_sequence": sequence,
        "edge_scores": edge_rows,
        "tool_step_scores": tool_step_rows,
        "terminal_edge": terminal_edge,
        "aggregate": {
            "num_tools": len(sequence),
            "num_edges": len(edge_rows),
            "sum_edge_score": float(sum(e["score"] for e in edge_rows)),
            "sum_tool_step_score": float(sum(s["score"] for s in tool_step_rows)),
            "mean_tool_step_score": float(
                sum(s["score"] for s in tool_step_rows) / len(tool_step_rows)
            )
            if tool_step_rows
            else None,
        },
    }

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Scored trace saved to: {out_path}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
