#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tool_graph_utils import build_tool_graph_from_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build tool graph(s) from tau2 native run output (results.json). "
            "Supports all rollouts, success/failure split, and incremental updates."
        )
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help=(
            "One or more tau2 result paths. Each can be a results.json file or "
            "a run directory containing results.json."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path for graph built from all records.",
    )
    parser.add_argument(
        "--output-success",
        default=None,
        help="Optional JSON path for success-only graph.",
    )
    parser.add_argument(
        "--output-failure",
        default=None,
        help="Optional JSON path for failure-only graph.",
    )
    parser.add_argument(
        "--records-jsonl",
        default=None,
        help="Optional JSONL path to write merged flattened rollout records.",
    )
    parser.add_argument(
        "--existing-records-jsonl",
        default=None,
        help=(
            "Optional existing flattened records JSONL. If provided, new rollout "
            "records are merged into it (deduplicated) before graph building."
        ),
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=1.0,
        help=(
            "Reward threshold for success classification. "
            "Default: 1.0 (reward >= 1.0 => success)."
        ),
    )
    parser.add_argument(
        "--min-edge-count",
        type=int,
        default=1,
        help=(
            "Minimum edge count retained in graph output. "
            "Default 1 keeps all observed edges (no low-frequency filtering)."
        ),
    )
    return parser.parse_args()


def _resolve_results_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser().resolve()
    if path.is_dir():
        candidate = path / "results.json"
        if not candidate.exists():
            raise FileNotFoundError(
                f"Directory does not contain results.json: {path}"
            )
        return candidate
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    return path


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
                "tool_call_id": message.get("id"),
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


def _is_success(
    sim: dict[str, Any],
    reward: float | None,
    success_threshold: float,
) -> bool:
    if reward is not None:
        return reward >= success_threshold
    return str(sim.get("termination_reason", "")).lower() == "agent_stop"


def _load_records(
    results_path: Path,
    success_threshold: float,
) -> list[dict[str, Any]]:
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    simulations = payload.get("simulations", [])
    if not isinstance(simulations, list):
        raise ValueError(f"Invalid tau2 results format: {results_path}")

    records: list[dict[str, Any]] = []
    for sim in simulations:
        if not isinstance(sim, dict):
            continue
        messages = sim.get("messages", [])
        if not isinstance(messages, list):
            messages = []

        reward = _extract_reward(sim)
        success = _is_success(sim, reward, success_threshold)
        tool_steps = _build_tool_steps(messages)

        records.append(
            {
                "status": "ok" if success else "error",
                "task_id": sim.get("task_id"),
                "simulation_id": sim.get("id"),
                "trial": sim.get("trial"),
                "seed": sim.get("seed"),
                "termination_reason": str(sim.get("termination_reason", "unknown")),
                "reward": reward,
                "tool_steps": tool_steps,
                "messages": messages,
                "source_results": str(results_path),
            }
        )
    return records


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _record_key(record: dict[str, Any]) -> tuple[str, ...]:
    source = str(record.get("source_results", ""))
    simulation_id = record.get("simulation_id")
    if simulation_id:
        return ("sim", source, str(simulation_id))
    return (
        "task",
        source,
        str(record.get("task_id", "")),
        str(record.get("trial", "")),
        str(record.get("seed", "")),
    )


def _merge_records(
    existing_records: list[dict[str, Any]],
    new_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[tuple[str, ...], dict[str, Any]] = {}
    for record in existing_records:
        merged[_record_key(record)] = record
    for record in new_records:
        merged[_record_key(record)] = record
    return list(merged.values())


def _write_graph(
    records: list[dict[str, Any]],
    path: Path,
    min_edge_count: int,
    status_filter: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    graph = build_tool_graph_from_records(
        records,
        min_edge_count=min_edge_count,
        status_filter=status_filter,
    )
    with path.open("w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    result_paths = [_resolve_results_path(p) for p in args.inputs]

    new_records: list[dict[str, Any]] = []
    for path in result_paths:
        new_records.extend(_load_records(path, success_threshold=args.success_threshold))

    existing_records: list[dict[str, Any]] = []
    if args.existing_records_jsonl:
        existing_path = Path(args.existing_records_jsonl).expanduser().resolve()
        existing_records = _load_jsonl_records(existing_path)

    all_records = _merge_records(existing_records, new_records)

    all_output = Path(args.output).expanduser().resolve()
    _write_graph(
        all_records,
        all_output,
        min_edge_count=args.min_edge_count,
        status_filter="all",
    )

    if args.output_success:
        success_output = Path(args.output_success).expanduser().resolve()
        _write_graph(
            all_records,
            success_output,
            min_edge_count=args.min_edge_count,
            status_filter="ok",
        )
        print(f"Success graph saved to: {success_output}")

    if args.output_failure:
        failure_output = Path(args.output_failure).expanduser().resolve()
        _write_graph(
            all_records,
            failure_output,
            min_edge_count=args.min_edge_count,
            status_filter="error",
        )
        print(f"Failure graph saved to: {failure_output}")

    if args.records_jsonl:
        records_output = Path(args.records_jsonl).expanduser().resolve()
        records_output.parent.mkdir(parents=True, exist_ok=True)
        with records_output.open("w", encoding="utf-8") as f:
            for record in all_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Merged records saved to: {records_output}")

    ok = sum(1 for r in all_records if r.get("status") == "ok")
    failed = len(all_records) - ok
    print(
        f"Loaded new={len(new_records)} records from {len(result_paths)} file(s). "
        f"existing={len(existing_records)}, merged={len(all_records)}. "
        f"Success={ok}, Failure={failed}. Graph saved to: {all_output}"
    )


if __name__ == "__main__":
    main()
