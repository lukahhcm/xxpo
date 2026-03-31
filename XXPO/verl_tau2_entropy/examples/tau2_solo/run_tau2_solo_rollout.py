#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Run tau2-bench tasks in solo-agent mode and export per-task rollout traces.

This bridges tau2's environment/task setting into the current project as a
first step: for each selected task, run one complete rollout with
`llm_agent_solo` (no user multi-turn interaction) and save trajectories to
JSONL.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from tool_graph_utils import (
    build_tool_graph_from_records,
    extract_edges_from_sequence,
    extract_tool_sequence_from_tool_steps,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tau2-bench solo rollouts and export trajectories."
    )
    parser.add_argument(
        "--tau2-repo",
        type=str,
        default=None,
        help=(
            "Path to a local tau2-bench repo clone. If set, `<tau2-repo>/src` "
            "is added to PYTHONPATH."
        ),
    )
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        help="tau2 domain name, e.g. airline / retail / telecom / banking_knowledge.",
    )
    parser.add_argument(
        "--task-set-name",
        type=str,
        default=None,
        help="tau2 task set name. Defaults to --domain.",
    )
    parser.add_argument(
        "--task-split-name",
        type=str,
        default="base",
        help="tau2 split name. Defaults to `base`.",
    )
    parser.add_argument(
        "--task-ids",
        nargs="*",
        default=None,
        help="Optional explicit task ids. If set, only these tasks are run.",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Optional cap on number of tasks.",
    )
    parser.add_argument(
        "--llm-agent",
        type=str,
        required=True,
        help="LLM name for tau2 solo agent (LiteLLM format).",
    )
    parser.add_argument(
        "--llm-args-agent",
        type=str,
        default="{}",
        help='JSON string for agent llm args, e.g. \'{"temperature": 0.0}\'.',
    )
    parser.add_argument(
        "--llm-user",
        type=str,
        default="gpt-4.1",
        help="Dummy user LLM field (not used in solo mode, kept for config completeness).",
    )
    parser.add_argument(
        "--llm-args-user",
        type=str,
        default="{}",
        help='JSON string for user llm args (not used in solo mode).',
    )
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-errors", type=int, default=10)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="If set, run evaluator and include reward.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="If set, stop immediately on first failed task rollout.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--tool-graph-output",
        type=str,
        default=None,
        help="Optional output JSON path for aggregated tool graph.",
    )
    parser.add_argument(
        "--tool-graph-min-edge-count",
        type=int,
        default=1,
        help="Minimum edge count retained in graph output.",
    )
    return parser.parse_args()


def _add_tau2_src_if_needed(tau2_repo: str | None) -> None:
    if tau2_repo is None:
        return
    tau2_src = Path(tau2_repo).expanduser().resolve() / "src"
    if not tau2_src.exists():
        raise FileNotFoundError(f"tau2 src path not found: {tau2_src}")
    sys.path.insert(0, str(tau2_src))


def _parse_json_dict(raw: str, field_name: str) -> dict[str, Any]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be a valid JSON object.") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{field_name} must be a JSON object.")
    return data


def _message_to_dict(message: Any) -> dict[str, Any]:
    role = getattr(message, "role", None)
    data: dict[str, Any] = {
        "role": role,
        "turn_idx": getattr(message, "turn_idx", None),
        "timestamp": getattr(message, "timestamp", None),
    }

    if hasattr(message, "content"):
        data["content"] = getattr(message, "content")

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        data["tool_calls"] = [
            {
                "id": getattr(tool_call, "id", ""),
                "name": getattr(tool_call, "name", ""),
                "arguments": getattr(tool_call, "arguments", {}),
                "requestor": getattr(tool_call, "requestor", None),
            }
            for tool_call in tool_calls
        ]

    if role == "tool":
        data["tool_call_id"] = getattr(message, "id", None)
        data["requestor"] = getattr(message, "requestor", None)
        data["error"] = getattr(message, "error", None)

    return {k: v for k, v in data.items() if v is not None}


def _build_tool_steps(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build step-level tool trace from flattened message list."""
    steps: list[dict[str, Any]] = []

    for message in messages:
        role = message.get("role")

        if role == "assistant" and message.get("tool_calls"):
            steps.append(
                {
                    "assistant_turn_idx": message.get("turn_idx"),
                    "tool_calls": message["tool_calls"],
                    "tool_results": [],
                }
            )
            continue

        if role == "tool":
            result = {
                "tool_call_id": message.get("tool_call_id"),
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


def main() -> None:
    args = parse_args()
    _add_tau2_src_if_needed(args.tau2_repo)

    try:
        from tau2.evaluator.evaluator import EvaluationType
        from tau2.orchestrator.orchestrator import Orchestrator
        from tau2.runner.build import build_agent, build_environment
        from tau2.runner.helpers import get_tasks
        from tau2.runner.simulation import run_simulation
        from tau2.user.user_simulator import DummyUser
    except ImportError as exc:
        raise ImportError(
            "Failed to import tau2 runtime. Install tau2 and its dependencies "
            "(or run `uv sync` in the tau2-bench repo), then retry. "
            f"Original import error: {exc}"
        ) from exc

    llm_args_agent = _parse_json_dict(args.llm_args_agent, "--llm-args-agent")
    llm_args_user = _parse_json_dict(args.llm_args_user, "--llm-args-user")

    task_set_name = args.task_set_name or args.domain
    tasks = get_tasks(
        task_set_name=task_set_name,
        task_split_name=args.task_split_name,
        task_ids=args.task_ids,
        num_tasks=args.num_tasks,
    )

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    success_count = 0
    error_count = 0
    all_records: list[dict[str, Any]] = []
    with out_path.open("w", encoding="utf-8") as writer:
        for index, task in enumerate(tasks):
            run_seed = args.seed + index
            task_id = str(task.id)
            print(
                f"[{index + 1}/{len(tasks)}] Running task_id={task_id} seed={run_seed}",
                flush=True,
            )

            try:
                environment = build_environment(
                    domain=args.domain,
                    solo_mode=True,
                    env_kwargs={},
                )
                agent = build_agent(
                    "llm_agent_solo",
                    environment,
                    llm=args.llm_agent,
                    llm_args=llm_args_agent,
                    task=task,
                    solo_mode=True,
                )
                user = DummyUser()
                orchestrator = Orchestrator(
                    domain=args.domain,
                    agent=agent,
                    user=user,
                    environment=environment,
                    task=task,
                    max_steps=args.max_steps,
                    max_errors=args.max_errors,
                    seed=run_seed,
                    solo_mode=True,
                    validate_communication=False,
                )
                if args.evaluate:
                    simulation = run_simulation(
                        orchestrator,
                        evaluation_type=EvaluationType.ALL,
                    )
                else:
                    simulation = orchestrator.run()

                raw_messages = simulation.get_messages()
                messages = [_message_to_dict(message) for message in raw_messages]
                tool_steps = _build_tool_steps(messages)
                reward = (
                    simulation.reward_info.reward
                    if getattr(simulation, "reward_info", None) is not None
                    else None
                )

                record = {
                    "status": "ok",
                    "domain": args.domain,
                    "task_set_name": task_set_name,
                    "task_split_name": args.task_split_name,
                    "task_id": task_id,
                    "seed": run_seed,
                    "ticket": str(getattr(task, "ticket", "")),
                    "termination_reason": str(simulation.termination_reason),
                    "duration": float(simulation.duration),
                    "agent_cost": simulation.agent_cost,
                    "user_cost": simulation.user_cost,
                    "reward": reward,
                    "num_messages": len(messages),
                    "num_tool_calls": sum(
                        len(message.get("tool_calls", []))
                        for message in messages
                        if message.get("role") == "assistant"
                    ),
                    "num_tool_errors": sum(
                        1
                        for message in messages
                        if message.get("role") == "tool"
                        and bool(message.get("error", False))
                    ),
                    "messages": messages,
                    "tool_steps": tool_steps,
                }
                tool_sequence = extract_tool_sequence_from_tool_steps(tool_steps)
                record["tool_sequence"] = tool_sequence
                record["tool_graph_edges"] = [
                    {"src": src, "dst": dst}
                    for src, dst in extract_edges_from_sequence(
                        tool_sequence, record["termination_reason"]
                    )
                ]

                writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                writer.flush()
                all_records.append(record)
                success_count += 1

            except Exception as exc:  # noqa: PERF203
                error_count += 1
                error_record = {
                    "status": "error",
                    "domain": args.domain,
                    "task_set_name": task_set_name,
                    "task_split_name": args.task_split_name,
                    "task_id": task_id,
                    "seed": run_seed,
                    "error": repr(exc),
                }
                writer.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                writer.flush()
                all_records.append(error_record)
                print(f"Task failed: {task_id} -> {exc}", flush=True)
                if args.stop_on_error:
                    raise

    if args.tool_graph_output:
        graph_output_path = Path(args.tool_graph_output).expanduser().resolve()
        graph_output_path.parent.mkdir(parents=True, exist_ok=True)
        graph = build_tool_graph_from_records(
            all_records,
            min_edge_count=args.tool_graph_min_edge_count,
        )
        with graph_output_path.open("w", encoding="utf-8") as f:
            json.dump(graph, f, ensure_ascii=False, indent=2)
        print(f"Tool graph saved to: {graph_output_path}", flush=True)

    print(
        f"Finished. Success={success_count}, Failed={error_count}, Output={out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
