# Data documentation

## Canonical experiment

The active data release uses `gemini/Trials/geminit20r3k30`, the archived five-run experiment matching the manuscript settings of 20 trials per step, 30 nearest neighbors, and up to three atom removals per trial.

### `canonical_trials.csv`

One row per archived trial. Core columns:

| Column | Meaning |
|---|---|
| `design_id` | Portable `run/step/trial` identifier |
| `run`, `step`, `trial` | Original experiment indices |
| `flux` | Predicted water flux in `ns⁻¹` as stored in the archive |
| `rejection` | Predicted ion-rejection fraction in `[0,1]` |
| `removed_atom_serials` | PDB serial numbers absent from the pristine structure |
| `source_pdb`, `source_image` | Relative location in the archived experiment |
| geometry columns | Convex-hull, compactness, connectivity, and boundary descriptors reconstructed from PDB files |

### `canonical_selected_steps.csv`

The selected trajectory for each of five runs. It preserves the archived `flux`, `rejection`, atom count, pore area, and pore perimeter.

### `high_performance_designs.csv`

All trial records satisfying:

```text
150 ≤ flux ≤ 170 ns⁻¹
rejection ≥ 0.90
```

There are 28 qualifying records and 23 unique removed-atom signatures. Duplicate records are preserved because they correspond to distinct archived trial events.

### `ablation_summaries.csv`

Combined summary tables for the tested trial-count, atom-removal, and nearest-neighbor settings. The `source` column retains the archived location.

### GPT/Gemini comparison files

- `gpt_agent_selected_steps.csv`
- `gpt_agent_trials.csv`
- `gemini_single_atom_all_trials.csv`
- `gemini_single_atom_summary.csv`

These files preserve the available archived comparison material. They should not be interpreted as a newly rerun benchmark.

## Large files excluded from Git

The original project manifest listed raw trajectories, large NumPy arrays, more than 40,000 PDB files, and more than 44,000 PNG files. The public repository should retain the compact processed tables and selected structures, while depositing bulk data in Zenodo, Figshare, or an institutional repository.
