# 目录说明

## config/
统一实验参数。
当前已有 `experiment.yaml`。

## valor/
未来放 VALOR 核心机制：
- models
- adapter
- capability
- ledger
- kernel
- vector utilities

## workload/
未来放合成工作负载：
- tool generator
- trace generator
- catalog generator
- adversarial request generator

## policies/
未来放实验对照组：
- fail-safe
- exact catalog
- union break-glass
- catalog + reserve
- envelope-only
- full VALOR wrapper

## experiments/
未来放：
- RQ1
- RQ2
- RQ3
- RQ4
- RQ5
- concurrency/replay

## results/raw/
原始 CSV / JSON 实验数据。

## results/figures/
实验图。

## results/tables/
论文表格。

## scripts/
未来放一键运行实验、汇总结果等脚本。

## tests/
单元测试、状态机测试、回归测试。

## data/
额外生成的数据集或中间数据。

## logs/
实验运行日志。
