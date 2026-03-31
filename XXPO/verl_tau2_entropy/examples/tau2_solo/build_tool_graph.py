#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tool_graph_utils import build_tool_graph_from_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build aggregated tool graph from rollout JSONL files."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more rollout JSONL files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Tool graph JSON output path.",
    )
    parser.add_argument(
        "--min-edge-count",
        type=int,
        default=1,
        help="Minimum edge count retained in output graph.",
    )
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def main() -> None:
    args = parse_args()
    input_paths = [Path(p).expanduser().resolve() for p in args.inputs]
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for p in input_paths:
        records.extend(_load_jsonl(p))

    graph = build_tool_graph_from_records(records, min_edge_count=args.min_edge_count)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)

    print(
        f"Loaded {len(records)} records from {len(input_paths)} files. "
        f"Graph saved to: {output_path}"
    )


if __name__ == "__main__":
    main()
