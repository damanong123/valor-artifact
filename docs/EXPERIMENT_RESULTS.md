# VALOR Final Experimental Results

> Automatically generated from the experiment CSV outputs. Do not manually edit reported numerical results.

## 1. Reproducibility metadata

- Generated UTC: `2026-09-21T08:17:30.293013+00:00`
- Git commit: `cf28018d8d074abf4d35fb489d8a11b5bfccd29f`
- Git working tree dirty: `True`
- Python: `3.13.9`
- SQLite: `3.51.0`
- Platform: `Windows-11-10.0.26100-SP0`

## 2. Final consistency checks

| Check | Pass |
| --- | --- |
| rq1_valor_coverage_100 | True |
| rq1_valor_exposure_bounded | True |
| rq2_methods_present | True |
| rq2_full_valor_coverage_100 | True |
| rq2_full_valor_exposure_bounded | True |
| rq5_all_correct | True |
| concurrency_all_pass | True |
| concurrency_no_violations | True |
| e6_cumulative_rejection_correct | True |
| e7_all_lifecycle_correct | True |
| e7_all_conservation | True |
| e8_baseline_100 | True |
| e8_recovery_monotone | True |

**All automated checks passed:** `True`

## 3. RQ1 — Coverage vs. false-trigger exposure

| K | Union coverage (%) | Coverage SD | VALOR coverage (%) | Union exposure (×) | VALOR exposure (×) | Catalog tools |
| --- | --- | --- | --- | --- | --- | --- |
| 10.0000 | 13.43 | 1.98 | 100.00 | 11.576 | 0.986 | 14.8 |
| 25.0000 | 31.20 | 5.21 | 100.00 | 12.136 | 0.986 | 25.0 |
| 50.0000 | 55.65 | 8.94 | 100.00 | 12.568 | 0.986 | 28.4 |
| 100 | 74.02 | 4.56 | 100.00 | 13.227 | 0.986 | 29.0 |

Interpretation: catalog widening increases unseen-trace coverage while accumulating standing emergency scope. VALOR admission in this construction-guaranteed feasible workload does not depend on prior contingency membership, while cumulative exceptional exposure remains bounded by the precommitted reserve.

## 4. RQ2 — Ablation

| Method | Coverage (%) | Coverage SD | Exposure (×) | Exposure SD | Attack requests accepted |
| --- | --- | --- | --- | --- | --- |
| Union Catalog | 55.65 | 8.94 | 12.568 | 4.430 | 25.00 |
| Catalog + Reserve | 55.65 | 8.94 | 0.988 | 0.007 | 6.20 |
| Envelope Only | 100.00 | 0.00 | 13.794 | 4.397 | 25.00 |
| Full VALOR | 100.00 | 0.00 | 0.986 | 0.011 | 6.00 |

Interpretation: adding only a reserve constrains cumulative spend but does not create scope for unseen actions; adding only an effect envelope removes the catalog-membership bottleneck but does not bound repeated cumulative use. Full VALOR combines both properties.

## 5. RQ3 — Local enforcement overhead

| Operation | N | Median (µs) | p95 (µs) | p99 (µs) |
| --- | --- | --- | --- | --- |
| Capability verification | 10000 | 10.60 | 10.90 | 14.50 |
| In-memory issuance | 10000 | 28.90 | 34.00 | 52.00 |
| Ed25519 verification | 10000 | 87.60 | 91.10 | 123.20 |
| SQLite steady-state issuance | 100 | 756.15 | 4968.48 | 6118.86 |
| SQLite cold-connect issuance | 100 | 8039.80 | 15648.49 | 22387.19 |

Primary durable per-action result: use **SQLite steady-state issuance** in the main paper. The cold-connect path is a supplementary measurement that includes repeated connection and PRAGMA setup.

## 6. RQ4 — Adapter-underestimation failure boundary

| target_dimension | underestimate_fraction | num_seeds | declared_exposure_mean | true_exposure_mean | declared_energy_x_mean | true_energy_x_mean | declared_safe_rate | true_safe_rate | physical_violation_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| energy | 0.30 | 5 | 0.915 | 0.964 | 0.470 | 0.672 | 1.000 | 0.600 | 0.400 |

Interpretation: declared ledger accounting can remain inside the signed reserve even when synthetic ground-truth effects cross the intended boundary. The experiment therefore illustrates the load-bearing nature of adapter conservativeness rather than contradicting the accounting invariant.

## 7. RQ5 — Construction-guaranteed negative control

| Trace class | Count | Correctly handled (%) | Observed admit (%) |
| --- | --- | --- | --- |
| Feasible (admit) | 3000 | 100.0 | 100.0 |
| Ground-truth infeasible (reject) | 3000 | 100.0 | 0.0 |

### RQ5 by construction mode

| mode | count | expected_admit_rate | observed_admit_rate | correct_rate |
| --- | --- | --- | --- | --- |
| feasible | 3000 | 1.000 | 1.000 | 1.000 |
| exceeds_initial_reserve | 1500 | 0.000 | 0.000 | 1.000 |
| outside_dmax | 1500 | 0.000 | 0.000 | 1.000 |

These are deterministic construction-guaranteed classes. The result is an implementation correctness check, not a classification-accuracy estimate.

## 8. Concurrency / replay stress

| experiment | attempts | successes | expected_successes | rejections_or_other | unexpected_errors | violations | pass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Concurrent reservation | 7200 | 1500 | 1500 | 5700 | 0 | 0 | True |
| Concurrent replay Execute | 24 | 1 | 1 | 23 | 0 | 0 | True |
| Concurrent double Commit | 24 | 1 | 1 | 23 | 0 | 0 | True |

The concurrent reservation experiment checks that atomic reserve updates prevent overspend. The replay and Commit tests additionally check one-shot execution and one-shot settlement under concurrent presentation.

## 9. E6 — Cumulative exhaustion

| num_seeds | total_cases | correct_rejection_rate | individually_admissible_rate | inside_envelope_rate | cumulative_rejection_rate | mean_sequence_length | min_sequence_length | max_sequence_length |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5.0000 | 1500 | 1.000 | 1.000 | 1.000 | 1.000 | 10.17 | 2.0000 | 35.0000 |

### Binding effect dimensions

| binding_dimension | cases | mean_sequence_length | correct_rate |
| --- | --- | --- | --- |
| energy | 466 | 11.65 | 1.000 |
| external_scope | 193 | 15.06 | 1.000 |
| irreversible | 321 | 4.80 | 1.000 |
| stress | 520 | 10.33 | 1.000 |

E6 isolates order-dependent cumulative infeasibility: every rejected action is individually admissible against a fresh B0, but prior exceptional spend leaves insufficient remaining reserve.

## 10. E7 — Execution-time state drift

| condition | trials | expected_permit_rate | observed_permit_rate | abort_rate | nonce_resolved_rate | conservation_rate | lifecycle_correct_rate | correct_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stable | 500 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| lower_effect | 500 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| higher_effect | 500 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| hard_guard_failure | 500 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

### Abort properties

| abort_trials | refund_success_rate | replay_rejection_rate | conservation_rate |
| --- | --- | --- | --- |
| 1000 | 1.000 | 1.000 | 1.000 |

E7 makes execution-time revalidation explicit: a capability that was valid at issuance is permitted only when the fresh execution-time contract still fits its reservation and hard guards.

## 11. E8 — False-trigger attack followed by legitimate emergency

| Requested attack steps | Accepted attack steps | Attack exposure (×) | Exposure SD | Recovery admission (%) | Recovery SD | Insufficient-reserve rejects |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0000 | 0.00 | 0.000 | 0.000 | 100.00 | 0.00 | 0.0 |
| 1.0000 | 1.00 | 0.552 | 0.176 | 89.20 | 11.21 | 32.4 |
| 2.0000 | 2.00 | 0.824 | 0.151 | 65.93 | 20.91 | 102.2 |
| 3.0000 | 3.00 | 0.883 | 0.067 | 61.00 | 13.16 | 117.0 |
| 4.0000 | 4.00 | 0.891 | 0.050 | 48.20 | 18.20 | 155.4 |
| 5.0000 | 5.00 | 0.948 | 0.048 | 20.27 | 27.77 | 239.2 |
| 6.0000 | 5.80 | 0.981 | 0.015 | 1.20 | 1.64 | 296.4 |
| 7.0000 | 6.00 | 0.986 | 0.011 | 0.60 | 1.34 | 298.2 |
| 8.0000 | 6.00 | 0.986 | 0.011 | 0.60 | 1.34 | 298.2 |

E8 quantifies the availability tradeoff: false-trigger requests remain bounded by the precommitted reserve, but spending that finite authority reduces the set of later legitimate recovery traces that remain affordable.

## 12. Interpretation boundaries

- The experiments are mechanism-level synthetic evaluations, not mission-risk estimates.
- VALOR's 100% coverage results are conditioned on workloads constructed to be effect-feasible under a fresh signed envelope and reserve.
- The RQ5 100%/100% result is a deterministic implementation correctness check over construction-guaranteed classes, not a learned-classifier accuracy result.
- The physical/consequence containment interpretation depends on conservative effect adapters; RQ4 deliberately violates this assumption.
- Reserve exhaustion remains an availability risk. E8 quantifies this tradeoff.
- The current artifact uses registered synthetic tools rather than a real LLM Agent or physical robot.
- 'Open-world' refers to previously uninstantiated action instances over registered interfaces, not arbitrary new actuators or tools.
- Universal security claims come from the formal model under its stated assumptions; finite experiments check the implementation and empirical boundary conditions.

## 13. Generated result artifacts

- `rq1`: `results\tables\rq1_summary.csv`
- `rq2`: `results\tables\rq2_ablation_summary.csv`
- `rq3`: `results\tables\rq3_latency_summary.csv`
- `rq3_env`: `results\raw\rq3_environment.json`
- `rq4`: `results\tables\rq4_summary.csv`
- `rq5`: `results\tables\rq5_summary.csv`
- `rq5_mode`: `results\tables\rq5_by_mode.csv`
- `concurrency`: `results\tables\concurrency_summary.csv`
- `e6`: `results\tables\e6_cumulative_exhaustion_summary.csv`
- `e6_dimension`: `results\tables\e6_cumulative_exhaustion_by_dimension.csv`
- `e7`: `results\tables\e7_state_drift_summary.csv`
- `e7_abort`: `results\tables\e7_abort_properties.csv`
- `e8`: `results\tables\e8_attack_then_emergency_summary.csv`

Final consolidation artifacts:

- `results/final/final_key_results.csv`
- `results/final/final_manifest.json`
- `docs/EXPERIMENT_RESULTS.md`