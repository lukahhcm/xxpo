#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from score_rollout_with_graph import (
    _edge_kind,
    _edge_prob,
    _edge_score,
    _graph_index,
    _load_graph,
    _load_records,
    _resolve_mode,
)
from tool_graph_utils import extract_edges_from_sequence, extract_tool_sequence_from_tool_steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch score rollout traces at step-level using success/failure tool graphs. "
            "Outputs JSONL (one scored trace per line)."
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
        "--status-filter",
        choices=["all", "ok", "error"],
        default="all",
        help="Record status filter. Default all.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of matched records to score.",
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
        help="Additive smoothing for graph edge probabilities. Set 0 to disable smoothing.",
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
        "--output-jsonl",
        required=True,
        help="Output JSONL path (one scored trace per line).",
    )
    parser.add_argument(
        "--output-summary",
        default=None,
        help="Optional output path for summary JSON.",
    )
    parser.add_argument(
        "--include-edge-scores",
        action="store_true",
        help="If set, include detailed edge_scores in each output line.",
    )
    return parser.parse_args()


def _record_matches(record: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.simulation_id is not None and str(record.get("simulation_id", "")) != str(
        args.simulation_id
    ):
        return False
    if args.task_id is not None and str(record.get("task_id", "")) != str(args.task_id):
        return False
    if args.trial is not None and int(record.get("trial", -1)) != int(args.trial):
        return False
    if args.seed is not None and int(record.get("seed", -1)) != int(args.seed):
        return False
    status = str(record.get("status", "error"))
    if args.status_filter == "ok" and status != "ok":
        return False
    if args.status_filter == "error" and status == "ok":
        return False
    return True


def _score_one_record(
    record: dict[str, Any],
    *,
    mode: str,
    success_idx: dict[str, Any] | None,
    failure_idx: dict[str, Any] | None,
    smoothing: float,
    eps: float,
    assign_terminal_to_last_step: bool,
    include_edge_scores: bool,
    graph_success: str | None,
    graph_failure: str | None,
    source_input: str,
) -> dict[str, Any]:
    tool_steps = record.get("tool_steps", []) or []
    sequence = extract_tool_sequence_from_tool_steps(tool_steps)
    term_reason = str(record.get("termination_reason", "unknown"))
    edges = extract_edges_from_sequence(sequence, term_reason)

    edge_rows: list[dict[str, Any]] = []
    for i, (src, dst) in enumerate(edges):
        p_success = _edge_prob(success_idx, src, dst, smoothing, eps)
        p_failure = _edge_prob(failure_idx, src, dst, smoothing, eps)
        score = _edge_score(mode, p_success, p_failure, eps)
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
    if assign_terminal_to_last_step and tool_step_rows:
        tool_step_rows[-1]["score"] = float(tool_step_rows[-1]["score"] + terminal_edge["score"])
        tool_step_rows[-1]["terminal_added"] = True

    out: dict[str, Any] = {
        "trace_selector": {
            "input": source_input,
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
            "smoothing": smoothing,
            "eps": eps,
            "assign_terminal_to_last_step": bool(assign_terminal_to_last_step),
            "graph_success": graph_success,
            "graph_failure": graph_failure,
        },
        "tool_sequence": sequence,
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
    if include_edge_scores:
        out["edge_scores"] = edge_rows
    return out


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_jsonl = Path(args.output_jsonl).expanduser().resolve()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    records = _load_records(input_path, args.input_format)
    if not records:
        raise ValueError(f"No records loaded from: {input_path}")

    selected = [r for r in records if _record_matches(r, args)]
    if args.limit is not None:
        selected = selected[: max(0, args.limit)]
    if not selected:
        raise ValueError("No rollout record matched the provided filters.")

    success_graph = _load_graph(args.graph_success)
    failure_graph = _load_graph(args.graph_failure)
    success_idx = _graph_index(success_graph)
    failure_idx = _graph_index(failure_graph)

    mode = _resolve_mode(
        args.mode,
        has_success=success_idx is not None,
        has_failure=failure_idx is not None,
    )

    graph_success_path = (
        str(Path(args.graph_success).expanduser().resolve()) if args.graph_success else None
    )
    graph_failure_path = (
        str(Path(args.graph_failure).expanduser().resolve()) if args.graph_failure else None
    )

    status_counter: dict[str, int] = {"ok": 0, "error": 0}
    num_with_tools = 0
    sum_edge_score = 0.0
    sum_tool_step_score = 0.0
    sum_tool_steps = 0

    with output_jsonl.open("w", encoding="utf-8") as f:
        for record in selected:
            scored = _score_one_record(
                record,
                mode=mode,
                success_idx=success_idx,
                failure_idx=failure_idx,
                smoothing=args.smoothing,
                eps=args.eps,
                assign_terminal_to_last_step=bool(args.assign_terminal_to_last_step),
                include_edge_scores=bool(args.include_edge_scores),
                graph_success=graph_success_path,
                graph_failure=graph_failure_path,
                source_input=str(input_path),
            )
            f.write(json.dumps(scored, ensure_ascii=False) + "\n")

            status = str(record.get("status", "error"))
            if status == "ok":
                status_counter["ok"] += 1
            else:
                status_counter["error"] += 1

            agg = scored["aggregate"]
            num_tools = int(agg["num_tools"])
            if num_tools > 0:
                num_with_tools += 1
            sum_edge_score += float(agg["sum_edge_score"])
            sum_tool_step_score += float(agg["sum_tool_step_score"])
            sum_tool_steps += num_tools

    summary = {
        "input": str(input_path),
        "input_format": args.input_format,
        "output_jsonl": str(output_jsonl),
        "num_total_records_loaded": len(records),
        "num_scored_records": len(selected),
        "status_counter": status_counter,
        "num_records_with_tools": num_with_tools,
        "scoring_config": {
            "mode": mode,
            "smoothing": args.smoothing,
            "eps": args.eps,
            "assign_terminal_to_last_step": bool(args.assign_terminal_to_last_step),
            "graph_success": graph_success_path,
            "graph_failure": graph_failure_path,
            "include_edge_scores": bool(args.include_edge_scores),
        },
        "aggregate": {
            "sum_edge_score": float(sum_edge_score),
            "mean_edge_score_per_record": float(sum_edge_score / len(selected)),
            "sum_tool_step_score": float(sum_tool_step_score),
            "mean_tool_step_score_global": float(sum_tool_step_score / sum_tool_steps)
            if sum_tool_steps > 0
            else None,
        },
    }

    if args.output_summary:
        summary_path = Path(args.output_summary).expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Summary saved to: {summary_path}")

    print(f"Scored {len(selected)} traces -> {output_jsonl}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
