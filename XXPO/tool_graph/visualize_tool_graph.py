#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize a tool graph JSON by exporting Graphviz DOT and optionally "
            "rendering SVG/PNG images."
        )
    )
    parser.add_argument(
        "--graph",
        required=True,
        help="Input graph JSON path (from build_tool_graph_*.py).",
    )
    parser.add_argument(
        "--output-dot",
        default=None,
        help="Output .dot path. Default: <graph_basename>.dot",
    )
    parser.add_argument(
        "--output-svg",
        default=None,
        help="Optional output .svg path (requires Graphviz).",
    )
    parser.add_argument(
        "--output-png",
        default=None,
        help="Optional output .png path (requires Graphviz).",
    )
    parser.add_argument(
        "--engine",
        default="dot",
        help="Graphviz layout engine binary name. Default: dot",
    )
    parser.add_argument(
        "--rankdir",
        choices=["LR", "TB", "RL", "BT"],
        default="LR",
        help="Layout direction for DOT graph. Default: LR",
    )
    parser.add_argument(
        "--min-edge-count",
        type=int,
        default=1,
        help="Keep only edges with count >= this value. Default: 1",
    )
    parser.add_argument(
        "--top-n-edges",
        type=int,
        default=None,
        help="Keep only top-N edges by count after filtering.",
    )
    parser.add_argument(
        "--min-node-count",
        type=int,
        default=1,
        help="Hide nodes with count < this value unless used by kept edges.",
    )
    parser.add_argument(
        "--keep-isolated-nodes",
        action="store_true",
        help="Keep nodes even if disconnected after edge filtering.",
    )
    parser.add_argument(
        "--label-with-count",
        action="store_true",
        help="Append node count to each node label.",
    )
    parser.add_argument(
        "--show-edge-prob",
        action="store_true",
        help="Show edge p_src (if present) on edge labels.",
    )
    return parser.parse_args()


def _quote(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _safe_node_id(value: Any) -> str:
    return str(value) if value is not None else ""


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _kind_from_node_id(node_id: str) -> str:
    if node_id == "__START__":
        return "start"
    if node_id == "__NO_TOOL__":
        return "no_tool"
    if node_id.startswith("__END_"):
        return "end"
    return "tool"


def _display_node_label(node_id: str) -> str:
    if node_id == "__START__":
        return "START"
    if node_id == "__NO_TOOL__":
        return "NO_TOOL"
    if node_id.startswith("__END_") and node_id.endswith("__"):
        core = node_id[len("__END_") : -2]
        return f"END:{core}"
    return node_id


def _node_style(kind: str, count: int, max_count: int) -> dict[str, str]:
    ratio = 0.0 if max_count <= 0 else min(1.0, max(0.0, count / max_count))
    fontsize = 11.0 + 8.0 * math.sqrt(ratio)
    penwidth = 1.0 + 2.0 * math.sqrt(ratio)

    if kind == "start":
        return {
            "shape": "box",
            "fillcolor": "#D1FAE5",
            "color": "#059669",
            "fontsize": f"{fontsize:.2f}",
            "penwidth": f"{penwidth:.2f}",
        }
    if kind == "end":
        return {
            "shape": "box",
            "fillcolor": "#FEE2E2",
            "color": "#DC2626",
            "fontsize": f"{fontsize:.2f}",
            "penwidth": f"{penwidth:.2f}",
        }
    if kind == "no_tool":
        return {
            "shape": "box",
            "fillcolor": "#E5E7EB",
            "color": "#6B7280",
            "fontsize": f"{fontsize:.2f}",
            "penwidth": f"{penwidth:.2f}",
        }
    return {
        "shape": "ellipse",
        "fillcolor": "#DBEAFE",
        "color": "#2563EB",
        "fontsize": f"{fontsize:.2f}",
        "penwidth": f"{penwidth:.2f}",
    }


def _edge_style(src: str, dst: str, count: int, max_count: int) -> dict[str, str]:
    ratio = 0.0 if max_count <= 0 else min(1.0, max(0.0, count / max_count))
    penwidth = 1.0 + 5.0 * math.sqrt(ratio)

    if src == dst:
        color = "#F59E0B"
    elif src == "__START__":
        color = "#10B981"
    elif dst.startswith("__END_"):
        color = "#EF4444"
    else:
        color = "#64748B"

    return {
        "color": color,
        "penwidth": f"{penwidth:.2f}",
        "arrowsize": "0.8",
    }


def _load_graph(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Graph JSON must be an object: {path}")
    return payload


def _collect_nodes_and_edges(
    graph: dict[str, Any],
    min_edge_count: int,
    top_n_edges: int | None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    node_index: dict[str, dict[str, Any]] = {}
    for node in graph.get("nodes", []):
        if not isinstance(node, dict):
            continue
        node_id = _safe_node_id(node.get("id"))
        if not node_id:
            continue
        kind = str(node.get("kind") or _kind_from_node_id(node_id))
        count = max(0, _safe_int(node.get("count"), 0))
        node_index[node_id] = {"id": node_id, "kind": kind, "count": count}

    edges: list[dict[str, Any]] = []
    threshold = max(1, int(min_edge_count))
    for edge in graph.get("edges", []):
        if not isinstance(edge, dict):
            continue
        src = _safe_node_id(edge.get("src"))
        dst = _safe_node_id(edge.get("dst"))
        if not src or not dst:
            continue
        count = max(0, _safe_int(edge.get("count"), 0))
        if count < threshold:
            continue
        p_src = _safe_float(edge.get("p_src"), 0.0)
        edges.append({"src": src, "dst": dst, "count": count, "p_src": p_src})

        if src not in node_index:
            node_index[src] = {
                "id": src,
                "kind": _kind_from_node_id(src),
                "count": 0,
            }
        if dst not in node_index:
            node_index[dst] = {
                "id": dst,
                "kind": _kind_from_node_id(dst),
                "count": 0,
            }

    edges = sorted(edges, key=lambda e: (-e["count"], e["src"], e["dst"]))
    if top_n_edges is not None:
        edges = edges[: max(0, top_n_edges)]

    return node_index, edges


def _make_dot(
    *,
    node_index: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    rankdir: str,
    min_node_count: int,
    keep_isolated_nodes: bool,
    label_with_count: bool,
    show_edge_prob: bool,
) -> str:
    used_nodes = {e["src"] for e in edges} | {e["dst"] for e in edges}

    selected_nodes: list[dict[str, Any]] = []
    for node in node_index.values():
        node_id = node["id"]
        node_count = max(0, _safe_int(node.get("count"), 0))
        if keep_isolated_nodes:
            if node_count < min_node_count and node_id not in used_nodes:
                continue
            selected_nodes.append(node)
            continue

        if node_id in used_nodes:
            selected_nodes.append(node)
        elif node_count >= min_node_count:
            selected_nodes.append(node)

    selected_nodes = sorted(selected_nodes, key=lambda n: (-_safe_int(n.get("count"), 0), n["id"]))

    max_node_count = max([_safe_int(n.get("count"), 0) for n in selected_nodes], default=1)
    max_edge_count = max([_safe_int(e.get("count"), 0) for e in edges], default=1)

    lines: list[str] = []
    lines.append("digraph ToolGraph {")
    lines.append(f"  graph [rankdir={rankdir}, bgcolor=\"white\", splines=true, overlap=false, pad=0.2];")
    lines.append("  node [fontname=\"Helvetica\", style=\"filled,rounded\", margin=\"0.08,0.06\"]; ")
    lines.append("  edge [fontname=\"Helvetica\", fontsize=10];")

    for node in selected_nodes:
        node_id = str(node["id"])
        kind = str(node.get("kind") or _kind_from_node_id(node_id))
        count = max(0, _safe_int(node.get("count"), 0))
        style = _node_style(kind, count, max_node_count)

        label = _display_node_label(node_id)
        if label_with_count:
            label = f"{label}\\n(count={count})"

        attrs = {
            "label": label,
            **style,
        }
        attr_text = ", ".join(f"{k}={_quote(v)}" for k, v in attrs.items())
        lines.append(f"  {_quote(node_id)} [{attr_text}];")

    selected_node_ids = {n["id"] for n in selected_nodes}
    for edge in edges:
        src = edge["src"]
        dst = edge["dst"]
        if src not in selected_node_ids or dst not in selected_node_ids:
            continue

        count = max(0, _safe_int(edge.get("count"), 0))
        p_src = _safe_float(edge.get("p_src"), 0.0)
        style = _edge_style(src, dst, count, max_edge_count)

        label = str(count)
        if show_edge_prob:
            label = f"{count} | p={p_src:.3f}"

        attrs = {
            "label": label,
            **style,
        }
        attr_text = ", ".join(f"{k}={_quote(v)}" for k, v in attrs.items())
        lines.append(f"  {_quote(src)} -> {_quote(dst)} [{attr_text}];")

    lines.append("}")
    return "\n".join(lines) + "\n"


def _render_graphviz(engine: str, dot_path: Path, out_path: Path, fmt: str) -> None:
    binary = shutil.which(engine)
    if binary is None:
        raise FileNotFoundError(
            f"Graphviz engine '{engine}' not found in PATH. "
            "Install graphviz first (e.g., brew install graphviz / apt-get install graphviz)."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [binary, f"-T{fmt}", str(dot_path), "-o", str(out_path)]
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()

    graph_path = Path(args.graph).expanduser().resolve()
    graph = _load_graph(graph_path)

    node_index, edges = _collect_nodes_and_edges(
        graph,
        min_edge_count=args.min_edge_count,
        top_n_edges=args.top_n_edges,
    )

    dot_text = _make_dot(
        node_index=node_index,
        edges=edges,
        rankdir=args.rankdir,
        min_node_count=max(1, int(args.min_node_count)),
        keep_isolated_nodes=bool(args.keep_isolated_nodes),
        label_with_count=bool(args.label_with_count),
        show_edge_prob=bool(args.show_edge_prob),
    )

    if args.output_dot:
        dot_path = Path(args.output_dot).expanduser().resolve()
    else:
        dot_path = graph_path.with_suffix(".dot")
    dot_path.parent.mkdir(parents=True, exist_ok=True)
    dot_path.write_text(dot_text, encoding="utf-8")
    print(f"DOT saved to: {dot_path}")

    if args.output_svg:
        svg_path = Path(args.output_svg).expanduser().resolve()
        _render_graphviz(args.engine, dot_path, svg_path, "svg")
        print(f"SVG saved to: {svg_path}")

    if args.output_png:
        png_path = Path(args.output_png).expanduser().resolve()
        _render_graphviz(args.engine, dot_path, png_path, "png")
        print(f"PNG saved to: {png_path}")


if __name__ == "__main__":
    main()
