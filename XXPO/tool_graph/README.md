# Tool Graph Utilities (tau2 Native Solo Rollout)

This folder contains tool-graph utilities.

Recommended workflow:
1. Run tau2 native solo rollout (`llm_agent_solo` + `dummy_user`) in `envs/tau2-bench`.
2. Save rollout traces under `XXPO/output/traces`.
3. Build `all/success/failure` tool graphs from tau2 `results.json`.
4. Score a single rollout path at step-level using the graphs.

## 0) Path Setup

Run from your project root (the directory that contains `envs/` and `XXPO/`):

```bash
cd /path/to/XXPO
export XXPO_ROOT="$(pwd)"
export TAU2_ROOT="$XXPO_ROOT/envs/tau2-bench"
export TOOL_GRAPH_ROOT="$XXPO_ROOT/XXPO/tool_graph"
export TRACE_ROOT="$XXPO_ROOT/XXPO/output/traces"
mkdir -p "$TRACE_ROOT"
```

## A) Run tau2 Native Solo Rollout (vLLM endpoint via LiteLLM args)

```bash
cd "$TAU2_ROOT"

# Optional: check served model id(s)
curl -s http://127.0.0.1:8067/v1/models

uv run tau2 run \
  --domain telecom \
  --agent llm_agent_solo \
  --user dummy_user \
  --task-split-name train \
  --num-trials 8 \
  --agent-llm openai/Qwen3-8B \
  --agent-llm-args '{"temperature":0.0,"api_base":"http://127.0.0.1:8067/v1","api_key":"EMPTY","max_tokens":8192}' \
  --save-to "$TRACE_ROOT/tau2_telecom_train_8trials_qwen3_8b"
```

## B) Build All / Success / Failure Graphs

`--min-edge-count` 默认是 `1`，即保留所有观测到的边（不做低频过滤）。

```bash
cd "$TOOL_GRAPH_ROOT"

python build_tool_graph_from_tau2_results.py \
  --inputs "$TRACE_ROOT/tau2_telecom_train_8trials_qwen3_8b" \
  --output "$TRACE_ROOT/tau2_telecom_train_8trials_all_graph.json" \
  --output-success "$TRACE_ROOT/tau2_telecom_train_8trials_success_graph.json" \
  --output-failure "$TRACE_ROOT/tau2_telecom_train_8trials_failure_graph.json" \
  --records-jsonl "$TRACE_ROOT/tau2_telecom_train_8trials_flattened.jsonl"
```

## C) Incremental Update (New Rollout Arrives)

当有新的 rollout 结果时，使用历史 `flattened.jsonl` 进行增量合并并重建图：

```bash
cd "$TOOL_GRAPH_ROOT"

python build_tool_graph_from_tau2_results.py \
  --inputs "$TRACE_ROOT/tau2_telecom_train_8trials_qwen3_8b_new" \
  --existing-records-jsonl "$TRACE_ROOT/tau2_telecom_train_8trials_flattened.jsonl" \
  --records-jsonl "$TRACE_ROOT/tau2_telecom_train_8trials_flattened.jsonl" \
  --output "$TRACE_ROOT/tau2_telecom_train_8trials_all_graph.json" \
  --output-success "$TRACE_ROOT/tau2_telecom_train_8trials_success_graph.json" \
  --output-failure "$TRACE_ROOT/tau2_telecom_train_8trials_failure_graph.json"
```

说明：
- 脚本会按 `(source_results, simulation_id)` 去重；缺少 `simulation_id` 时回退到 `(source_results, task_id, trial, seed)`。
- 你可以重复执行同一个更新命令，已存在记录不会重复累加。

## D) Build Graph from Existing Rollout JSONL

```bash
cd "$TOOL_GRAPH_ROOT"
python build_tool_graph_from_jsonl.py \
  --inputs "$TRACE_ROOT/some_rollout.jsonl" \
  --output "$TRACE_ROOT/some_rollout_graph.json" \
  --min-edge-count 1
```

## E) Score One Rollout Path with Graphs (Step-Level)

脚本：`score_rollout_with_graph.py`

用途：给一条 rollout 路径逐 step 打分，模拟训练时 step-level reward。

默认模式：
- 同时提供 success/failure 图时，使用 `log_odds`：
  `score = log(p_success(edge)) - log(p_failure(edge))`
- 只提供 success 图：`success_only`
- 只提供 failure 图：`failure_only`

示例（从 flattened records 里选一条 task）：

```bash
cd "$TOOL_GRAPH_ROOT"

python score_rollout_with_graph.py \
  --input "$TRACE_ROOT/tau2_telecom_train_8trials_flattened.jsonl" \
  --task-id "<task_id>" \
  --index 0 \
  --graph-success "$TRACE_ROOT/tau2_telecom_train_8trials_success_graph.json" \
  --graph-failure "$TRACE_ROOT/tau2_telecom_train_8trials_failure_graph.json" \
  --output "$TRACE_ROOT/tau2_telecom_train_8trials_task_score.json"
```

也可以直接输入 tau2 的 `results.json` 或 run 目录：

```bash
python score_rollout_with_graph.py \
  --input "$TRACE_ROOT/tau2_telecom_train_8trials_qwen3_8b" \
  --input-format tau2_results \
  --task-id "<task_id>" \
  --graph-success "$TRACE_ROOT/tau2_telecom_train_8trials_success_graph.json" \
  --graph-failure "$TRACE_ROOT/tau2_telecom_train_8trials_failure_graph.json"
```

可选项：
- `--simulation-id / --trial / --seed`：精确选中一条 trace。
- `--assign-terminal-to-last-step`：把 terminal 边分数加到最后一个工具 step。
- `--smoothing`、`--eps`：控制边概率平滑和数值稳定性。

## F) Batch Score All Rollouts with Graphs (Step-Level)

脚本：`score_rollouts_with_graph.py`

用途：把一个输入里的多条 rollout 全部按 step-level 打分，输出 JSONL（每行一条 trace 的打分结果）。

示例（对训练集 rollout 全量打分）：

```bash
cd "$TOOL_GRAPH_ROOT"

python score_rollouts_with_graph.py \
  --input "$TRACE_ROOT/tau2_telecom_train_8trials_flattened.jsonl" \
  --graph-success "$TRACE_ROOT/tau2_telecom_train_8trials_success_graph.json" \
  --graph-failure "$TRACE_ROOT/tau2_telecom_train_8trials_failure_graph.json" \
  --mode log_odds \
  --output-jsonl "$TRACE_ROOT/tau2_telecom_train_8trials_step_scores.jsonl" \
  --output-summary "$TRACE_ROOT/tau2_telecom_train_8trials_step_scores_summary.json"
```

常用筛选：
- `--status-filter ok|error`：只打成功或失败轨迹。
- `--task-id / --trial / --seed / --simulation-id`：精确筛选子集。
- `--limit N`：只取前 N 条进行快速验证。
- `--assign-terminal-to-last-step`：把 terminal 边分数并入最后一个工具 step。
- `--include-edge-scores`：在每条输出里附带完整边级分数。

## G) Visualize Tool Graph (DOT / SVG / PNG)

脚本：`visualize_tool_graph.py`

用途：把 `*_graph.json` 转成 Graphviz DOT，并可直接渲染成 `svg/png`。

示例（可视化 success 图）：

```bash
cd "$TOOL_GRAPH_ROOT"

python visualize_tool_graph.py \
  --graph "$TRACE_ROOT/tau2_telecom_train_8trials_success_graph.json" \
  --output-dot "$TRACE_ROOT/tau2_telecom_train_8trials_success_graph.dot" \
  --output-svg "$TRACE_ROOT/tau2_telecom_train_8trials_success_graph.svg" \
  --output-png "$TRACE_ROOT/tau2_telecom_train_8trials_success_graph.png" \
  --rankdir LR \
  --top-n-edges 300 \
  --label-with-count
```

也可以对 failure 图跑一遍：

```bash
python visualize_tool_graph.py \
  --graph "$TRACE_ROOT/tau2_telecom_train_8trials_failure_graph.json" \
  --output-svg "$TRACE_ROOT/tau2_telecom_train_8trials_failure_graph.svg" \
  --rankdir LR \
  --top-n-edges 300
```

常用参数：
- `--min-edge-count`：过滤低频边。
- `--top-n-edges`：只保留最高频的前 N 条边，避免图太密。
- `--min-node-count`：隐藏低频节点。
- `--keep-isolated-nodes`：保留断开节点。
- `--show-edge-prob`：在边标签显示 `p_src`。

如果提示找不到 `dot`：
- macOS: `brew install graphviz`
- Ubuntu/Debian: `sudo apt-get install graphviz`
