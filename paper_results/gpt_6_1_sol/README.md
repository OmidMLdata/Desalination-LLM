# GPT-6.1 Sol run (`results_gpt61_t30_g3_20261015`)

| File | Contents |
|---|---|
| `gpt61_per_step.csv` | Six-category per-step series (steps 0-30), built from `raw/six_category_summary.csv`. Scaffold rows give mean ± SD over 6 seeds. The manuscript figures use this file. |
| `gpt61_per_step_from_pdf.csv` | The same series recovered from the vector paths of `six_category_comparison.pdf`. It agrees with the raw data to within 0.006 ns^-1 and is kept as a cross-check. |
| `raw/` | Screened copy of the run folder. |

## Contents of `raw/`

- **Top-level files:**
  - protocol;
  - frozen input hashes;
  - verification and plot-integrity records;
  - `REPORT.md`;
  - `six_category_per_step.csv`, `six_category_summary.csv`, `final_metrics.csv`;
  - the comparison figures.
- **`runs/<policy>_seed<k>/`:** `per_step.csv`, `all_trials.csv`, `decisions.csv`, the final pore (PDB and PNG) and the resumable checkpoint.
- **`runs/gpt_tools_seed0/` and `runs/gpt_blind_seed0/`:** additionally contain the model calls (prompts, requests, responses, events).

`raw/MANIFEST.tsv` lists every file in the original folder, whether it was kept, and its SHA-256 hash.

## Settings, from `raw/protocol.json`

- **Model:** `gpt-6.1-sol` with low reasoning effort.
- **Design budget:** 30 decisions; 30 trials, keep included; at most 3 new atoms per trial; 90 atoms in total.
- **Candidates:** k = 30 neighbours; start atom 301.
- **Rejection floor:** 0.90.
- **Seeds:** 20261012-20261017.

Predictions come from the frozen ResNet-50/MLP (all2) checkpoints with `mean_std_all.pickle`. Every recorded state was re-scored, with a maximum difference of 7.6e-5 ns^-1. Negative raw flux outputs are clipped at zero, which is why the smallest pores show zero flux.

## Re-importing from the original machine

```bash
python scripts/import_gpt61_results.py /path/to/results_gpt61_t30_g3_20261015            # dry run
python scripts/import_gpt61_results.py /path/to/results_gpt61_t30_g3_20261015 --apply
```
