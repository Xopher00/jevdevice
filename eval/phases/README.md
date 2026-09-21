# eval/phases — analysis and batch tooling

One-off and per-phase evaluation scripts live here, isolated from the
package. Functional names, not phase numbers; historical references in
LOGBOOK.md and agent-plans/ use the old names, which map as:

| script | was | purpose |
|---|---|---|
| laya_factcheck.py | phase0_factcheck | live fact battery for the laya runtime |
| smoke_laya.py | phase2_smoke | MCP-surface smoke on the in-process engine |
| smoke_budget.py | phase3_budget_smoke | budget-profile path smoke (shortlist, sweep fallback) |
| audit_budget.py | phase3_budget_audit | journal-only state-size audit vs profiles |
| recalibrate_thresholds.py | phase4_recalibrate | per-question-type quality + threshold refits (journal-only) |
| perturbations.py | phase4_perturbations | question drift under shuffle/paraphrase/evidence-drop (live) |
| ab.py | phase5_ab | dev-half A/B worker + scorecard |
| shadow_agreement.py | phase5_report | shadow vs primary agreement report (journal-only) |
| compile_questions.py | phase6_compile_questions | validates question_sets/v1.yaml against the journal |
| export_finetune.py | phase6_export_finetune | verified rows -> fine-tune examples |
| kaggle_finetune.py | phase6_kaggle_finetune | operator-run fine-tune on Kaggle |
| eval_finetune.py | phase6_eval_finetune | base vs fine-tuned scorecard (JEV_DEVICE=cuda) |
| perturbation_harness.py | phase7_harness | 34-run perturbation batch over dev phone goals |
| mine_recoveries.py | phase7_mine_failures | failure -> recovery pairs from the journal |
| export_flywheel.py | phase7_export_flywheel | training-format exporters (recovery + span-weighted) |
| build_recipes.py | phase75_build_recipes | journal -> recipe store (held-out guard is hard) |
| tier_report.py | phase75_report | per-tier planner telemetry from the journal |
| planner_batch.py | phase75_run_goals | unattended planner batch over dev phone goals |
| metr_family.py | — | METR Task Standard-shaped adapter over the phone family (dev-half tasks; verify = journaled verified outcome) |
| splitguard.py | — | the ONE shared dev/held-out split loader |

Data dirs: `recalibration/`, `ab/`, `finetune/`, `flywheel/`, `recipes/`.
The held-out half of `../goals.yaml` is never run, tuned on, or exported.
