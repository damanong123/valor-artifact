# VALOR 实验部署路线

## Phase 0 — 工程环境
- [x] 工程目录
- [x] requirements.txt
- [x] Linux/macOS 一键安装脚本
- [x] Windows PowerShell 一键安装脚本
- [x] Windows CMD 一键安装脚本
- [x] 实验参数配置

## Phase 1 — Synthetic Tool Universe
后续实现：
- `workload/tool_generator.py`
- 30 tools
- 8 effect domains
- normal range
- exceptional range
- 4 effect coefficients
- 1 tool outside Dmax

输出：
- `results/raw/tools_seed_<seed>.csv`

## Phase 2 — Effect Adapter
后续实现：
- `valor/adapter.py`

输入：
- tool
- parameter
- synthetic state

输出：
- effect contract χ(a,x)
- domains
- bound vector
- hard guard
- receipt class

## Phase 3 — Trace Generator
后续实现：
- `workload/trace_generator.py`

生成：
- 2–5 actions / trace
- 至少一个动作超出 normal authorization
- feasible trace
- negative-control trace

## Phase 4 — Baselines
后续实现：
- Fail-safe
- Exact Catalog
- Union Break-glass
- Catalog + Reserve
- Envelope-only

## Phase 5 — VALOR Kernel
后续实现：
- Propose
- Execute
- Commit
- reserve ledger
- nonce
- one-shot capability
- execution-time revalidation

## Phase 6 — Experiments
后续实现：
- RQ1 coverage/exposure
- RQ2 ablation
- RQ3 latency
- RQ4 bad adapter
- RQ5 negative control
- concurrency/replay

## Phase 7 — Results
输出：
- CSV
- PNG
- tables
- logs
