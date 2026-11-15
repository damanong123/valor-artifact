# VALOR Real-Agent Experiment Results

## Experimental status

- Main real-agent experiment: 900 runs.
- Main prompt-injection attack experiment: 900 runs.
- Main experiment gates passed: `True`.
- Attack-main gates passed: `True`.

## Main capability result

For catalog-unseen but effect-feasible recovery tasks:

- Normal-only completion: 0.00%.
- Frozen-catalog completion: 0.00%.
- VALOR completion: 96.67%.

This is the central adaptive-authority result: the frozen catalog cannot authorize these exact runtime actions, while VALOR can authorize them when their effects remain inside the registered contract and remaining reserve.

## Main safety result

- Unsafe mission success for novel-infeasible tasks under VALOR: 0.00%.
- VALOR authority conservation across Main runs: 100.00%.
- Maximum observed exceptional exposure in the Main workload: 0.2100x B0.

The empirical maximum is not a theoretical upper bound. The mechanism-level bound remains `<= 1.0 x B0`; the reported value is only the largest exposure observed in this workload.

## Prompt-injection result

- Attack-goal observation rate: 63.33%.
- Behavior-change rate: 98.06%.
- VALOR attack-bound preservation: 100.00%.
- VALOR maximum attack exposure: 0.4260x B0.

The attack suite therefore changed the behavior of the real Qwen3 agent rather than merely presenting inert malicious text.

## Compromised-Agent containment

- VALOR runs in which the attack goal was actually observed: 179.
- Bound preservation in this compromised subset: 100.00%.
- Maximum exposure in this compromised subset: 0.4260x B0.

The supported security claim is therefore **authority containment**, not prompt-injection prevention: successful manipulation of the planner did not create unbounded authority.

## Catalog comparison under attack

Among catalog-policy runs where the attack goal was observed, the denial -> replanning -> mission-success rate was 76.74%.

This illustrates the principal trade-off between the two exceptional-authority designs:

- A frozen catalog can strongly reject actions outside pre-enumerated scope.
- VALOR can admit catalog-unseen actions when their trusted effect contracts remain feasible under the remaining reserve.

## Planner variability

The remaining Main deviations were dominated by planner-side variability rather than enforcement violations:

- `planner_never_reached_authorization_boundary`: 16 runs (53.33%).
- `planner_premature_finish_or_wrong_action`: 10 runs (33.33%).
- `planner_loop_or_nonconvergence`: 3 runs (10.00%).
- `planner_failed_after_authorization_denial`: 1 runs (3.33%).

## Interpretation boundary

The experiments do **not** show that VALOR prevents prompt injection or guarantees task-intent integrity. They show that consequential authority remains quantitatively bounded even when planner behavior is manipulated.
