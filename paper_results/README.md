# Data behind the manuscript: Claude Opus 5.5 vs GPT-6.1 Sol

The files here back the figures and tables of *"Desalination LLM: Large Language Model Agents Design Efficient Graphene Membranes for Water Desalination"* (revised version).

All runs use the same protocol:
- 30 design steps;
- up to 30 trials per step;
- 0-3 atoms removed per trial, drawn from the k = 30 nearest unremoved atoms, plus colonisation, with at most 3 new atoms per step;
- a 90 % ion-rejection floor.

## Layout

| Path | Contents | Used in |
|---|---|---|
| `claude_opus_5_5/opus_tools_per_step.csv` | Claude Opus 5.5 with tools (agent): the pore held after each step, with flux, rejection, geometry, votes and rationale | Fig. 3a,b; Table 1; Table S2 |
| `claude_opus_5_5/opus_notools_per_step.csv` | Claude Opus 5.5 in conversation only: predictor score (`flux`, `rejection`) and the model's own estimate (`claimed_flux`, `claimed_rejection`) | Fig. 3a,b; Fig. S3; Table 1; Table S2 |
| `claude_opus_5_5/scaffold_{greedy,constrained,random}_s{42,1..5}_per_step.csv` | Scaffold-only baselines with no LLM, 6 seeds each | Fig. 3; Fig. S2; Table 1 |
| `claude_opus_5_5/*_trials.csv.gz` | Every evaluated trial pore (flux, rejection, 21 geometric descriptors, removed atoms) | Fig. 4a,b; Fig. 5; Fig. S1 |
| `claude_opus_5_5/tsne_embedding.csv` | t-SNE coordinates of all distinct pores | Fig. 5 |
| `claude_opus_5_5/decision_logs/` | Step-by-step decision logs of both Opus runs (chosen trial or atoms, expert votes, rationale, own estimates) | Methods; Results |
| `claude_opus_5_5/final_structures/` | Final PDB structures of the five Claude-run conditions (scaffold: seed 42) | Fig. S1b |
| `gpt_6_1_sol/gpt61_per_step.csv` | GPT-6.1 Sol per-step series for the six conditions, built from the raw run | Fig. 3c,d; Fig. 4b; Fig. S2c,d; Fig. S3; Table 1; Table S2 |
| `gpt_6_1_sol/raw/` | Screened GPT-6.1 Sol run folder: protocol, per-run trials and decisions, model calls, verification (see `gpt_6_1_sol/README.md`) | Methods; provenance |
| `reference_wang2021/wang2021_dataset_labels.csv` | MD dataset labels of Wang et al., npj 2D Mater. Appl. 5, 66 (2021), from github.com/BaratiLab/Graphene-RL (MIT licence) | Fig. 4b |
| `reference_wang2021/wang2021_drl_hist_digitized.csv` | DRL flux histogram (rejection > 90 %), digitised from Figure 4b of the first manuscript version | Fig. 4b |
| `figures/` | Final figure files (PDF and PNG), `table_metrics.csv`, `fig4b_stats.csv`, `tableS2_per_step.csv` | all |

## Evaluation route
Both models use the CNN-MLP predictor of Wang et al. (2021) as the baseline evaluation method:
- the GPT-6.1 Sol runs queried the frozen ResNet-50/MLP checkpoints directly;
- the Claude Opus 5.5 runs queried the geometry surrogate in `models/geometry_surrogate.joblib`, which was trained on 3,000 CNN-MLP predictions (grouped-CV R² 0.93 for flux and 0.97 for rejection).

## Rebuilding the figures

```bash
pip install -e . && pip install scikit-learn pillow
CLAUDE_OPUS_RESULTS=paper_results/claude_opus_5_5 python -m claude_opus.figures_v3 --out paper_results/figures
```

## Rerunning the experiments
See `claude_opus/README.md`.
