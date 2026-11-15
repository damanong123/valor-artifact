# VALOR Experimental Scope

This artifact contains two complementary evaluation tracks.

## Synthetic Mechanism-Level Evaluation

- RQ1: Coverage and false-trigger exposure
- RQ2: Authorization mechanism ablation
- RQ3: Local enforcement latency
- RQ4: Adapter-underestimation failure boundary
- RQ5: Construction-guaranteed negative controls
- E6: Cumulative reserve exhaustion
- E7: Execution-time state drift
- E8: False-trigger reserve exhaustion
- Concurrency and replay stress tests

## Qwen3-8B Agent Evaluation

The autonomous planner is instantiated using a real
Qwen3-8B model through an OpenAI-compatible inference API.

The recorded evaluation includes:

- 900 main experiment runs
- 900 prompt-injection experiment runs
- 180 independently executed clean-control runs
- 720 attacked runs

The 180 clean-control runs are matched to the
180 adversarial main-experiment runs by task,
policy and repetition index, while using distinct run IDs.

## Execution Environment

The authorization mechanism and effect adapters are
implemented in Python.

Consequential tools operate in a simulated environment.

The artifact does not constitute physical robot deployment
or hardware-in-the-loop validation.

Physical-effect containment depends on the stated
adapter-conservativeness and trusted-mediation assumptions.
