# VALOR Research Artifact

This repository contains the VALOR reference implementation
and experimental artifacts accompanying the paper.

## 1. Experimental Evaluation

The artifact contains two complementary evaluation tracks.

### Synthetic Mechanism-Level Experiments

- RQ1: Unseen-action coverage and false-trigger exposure
- RQ2: Authorization mechanism ablation
- RQ3: Local enforcement overhead
- RQ4: Adapter-underestimation failure boundary
- RQ5: Construction-guaranteed negative controls
- E6: Cumulative reserve exhaustion
- E7: Execution-time state drift
- E8: False-trigger reserve exhaustion
- Concurrency and replay stress tests

### Qwen3-8B Agent Experiments

- Main experiment: 900 runs
- Prompt-injection experiment: 900 runs
- Prompt-injection clean controls: 180 runs
- Attacked runs: 720 runs

Raw execution logs, analysis scripts, result tables,
figures and final manifests are included.

## 2. Repository Structure

- `valor/`: VALOR authorization kernel and core mechanisms
- `policies/`: Baseline authorization policies
- `workload/`: Synthetic tools and trace generators
- `experiments/`: RQ1-RQ5 and supplementary experiments
- `tests/`: Mechanism and regression tests
- `scripts/`: Experiment utilities
- `agent_integration/`: Qwen3-8B agent integration
  and authorization gateway
- `results/`: Synthetic experiment outputs
- `agent_integration/results/`: Agent experiment outputs
- `config/`: Experiment configurations
- `docs/`: Experimental documentation

## 3. Installation

Requirements:

- Python 3.11 or later
- pip

Create an environment:

    python -m venv .venv

Windows PowerShell:

    .\.venv\Scripts\Activate.ps1

Linux/macOS:

    source .venv/bin/activate

Install synthetic experiment dependencies:

    python -m pip install -r requirements.txt

Run tests:

    python -m pytest -q tests

## 4. Synthetic Experiments

Experiment entry points:

    python -m experiments.rq1_coverage_exposure
    python -m experiments.rq2_ablation
    python -m experiments.rq3_latency
    python -m experiments.rq4_bad_adapter
    python -m experiments.rq5_negative_control
    python -m experiments.e6_cumulative_exhaustion
    python -m experiments.e7_state_drift
    python -m experiments.e8_attack_then_emergency
    python -m experiments.concurrency_replay

These commands regenerate experimental outputs.

## 5. Qwen3-8B Agent Integration

The agent imports the OpenAI Python SDK and communicates
with an OpenAI-compatible inference endpoint.

Install the additional agent dependency:

    python -m pip install -r requirements-agent.txt

The Qwen3-8B experiments were conducted on a remote
inference server using vLLM.

The agent endpoint is configurable. The default
127.0.0.1:8000/v1 address refers to the loopback
interface of the machine running the agent process.

The model weights and remote inference server are
not included in this repository.

The repository includes the agent integration source
code, experiment configurations, raw execution logs,
analysis scripts and generated results.

Re-running the agent experiments requires a separately
provisioned Qwen3-8B inference endpoint.
## 6. Experimental Scope and Limitations

The synthetic evaluation uses registered synthetic tools,
typed effect models and construction-controlled traces.

The Qwen3-8B evaluation uses a real language-model planner
connected to a simulated execution environment.

This artifact does not claim physical robot deployment
or real hardware-in-the-loop validation.

Physical-effect containment remains conditional on
the stated adapter-conservativeness and trusted-mediation
assumptions.

## 7. Recorded Results

Synthetic experiment results:

- `docs/EXPERIMENT_RESULTS.md`
- `results/figures/`
- `results/tables/`
- `results/raw/`

Agent experiment results:

- `agent_integration/results/final/AGENT_EXPERIMENT_RESULTS.md`
- `agent_integration/results/raw/`
- `agent_integration/results/tables/`
- `agent_integration/results/final/`


