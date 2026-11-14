# Repair study (D9) run records

Each case folder (`<pilot>/<fault>/<condition>/`) holds what `scripts/run_repair_case.py`
wrote, minus the images and plan arrays (re-renderable from the specs):

- `record.json`: the run record (`manip_sim/repair_study.py`): rounds, calls with stop
  reasons, edits, plan results, final grades, spec diff
- `fault.json`, `spec_start/`, `spec_faulty/`, `spec_final/`, `round_N_spec/`: the specs
- `round_N/report.json`, `round_N/chain.json`: what the critic / checker saw
- `responses/<kind>_NN.txt`: every raw model response (replay with `--mode replay`)
- `run.log`

Pilots (insert marker, claude-opus-5, temperature omitted):

- `pilot1/` (2026-11-13): Opus gt_opus1 starting spec. The render_check condition got empty
  answers (API refusal, category reasoning_extraction, triggered by the field name
  "Think Step By Step"); the critic spent rounds on the starting spec's own flaws. 33 requests.
- `pilot2/` (2026-11-14): the corrected starting spec (`tasks/repair_study/start/insert_marker`),
  `control*/` = the critic on the unfaulted spec (control 1 rejected, fix 5 applied; control 2
  accepted), `cases/` = 3 faults x 3 conditions. 29 requests.
