# Tau2 Solo Rollout (First-Step Integration)

This example integrates `tau2-bench` task + environment setting into the current project for **solo-agent rollout**:

- Agent: `llm_agent_solo`
- User: `dummy_user` (no multi-turn user interaction)
- Behavior: agent can call **multiple tools sequentially** until stop

It is intended as the first milestone: **run rollout for each task and export trajectory**.

## 1) Prepare tau2

Option A: install `tau2` into your current environment.

Option B: keep a local clone and pass `--tau2-repo /path/to/tau2-bench` (script will add `/path/to/tau2-bench/src` to `PYTHONPATH`).

Note: if you only provide a local clone, you still need tau2 runtime dependencies installed. The simplest way is:

```bash
cd /path/to/tau2-bench
uv sync
```

## 2) Run

```bash
cd /Users/tarak30/Downloads/ARPO-main/XXPO/verl_tau2_entropy

python examples/tau2_solo/run_tau2_solo_rollout.py \
  --tau2-repo /path/to/tau2-bench \
  --domain telecom \
  --task-split-name base \
  --num-tasks 5 \
  --llm-agent gpt-4.1 \
  --llm-args-agent '{"temperature": 0.0}' \
  --output /tmp/tau2_telecom_solo_rollout.jsonl \
  --tool-graph-output /tmp/tau2_telecom_tool_graph.json
```

## 3) Output format

Each line in output JSONL is one task rollout:

- `status`: `ok` or `error`
- `task_id`, `termination_reason`, `duration`, `reward` (if `--evaluate`)
- `messages`: flattened message trajectory
- `tool_steps`: step-level tool call/result trace
- `tool_sequence`: ordered tool names used by the task
- `tool_graph_edges`: per-task directed edges including start/end nodes

For rollout-only speed, do not set `--evaluate`.  
If you also need reward fields, add `--evaluate`.

## 4) Rebuild graph from existing rollout files

```bash
python examples/tau2_solo/build_tool_graph.py \
  --inputs /tmp/tau2_telecom_solo_rollout.jsonl \
  --output /tmp/tau2_telecom_tool_graph_rebuild.json \
  --min-edge-count 1
```
