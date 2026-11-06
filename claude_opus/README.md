# `claude_opus`: LLM agent vs scaffold vs tool-use benchmark

This is the code behind the Claude Opus 5.5 / GPT-6.1 Sol comparison in the revised manuscript. Every condition runs through one environment (`env.py`), so the conditions share:
- the starting sheet;
- the k-nearest-neighbour candidate pool;
- the seeded trial generator;
- the predictor.

Only the decision-maker, and what it is allowed to see, changes between conditions.

| File | Purpose |
|---|---|
| `env.py` | Design environment. The `gpt61` protocol: 30 steps, 30 trials, 0-3 atoms + colonisation, 90 % floor. The `fig3_v1` protocol: 20 trials, 1 atom, 60 % floor. |
| `prompts.py` | Figure 2 user-query prompt (floor set by protocol), tool and conversation addenda, three expert prompts |
| `run_tools_agent.py` | LLM + tools: Anthropic Messages API tool-use loop with `evaluate_trials`, `consult_expert_panel`, `commit_step` |
| `run_conversation.py` | LLM no tools: conversation only. The model picks atoms and states its own flux/rejection; the predictor scores are logged but never shown to the model. |
| `run_scaffold_baseline.py` | Scaffold only, no LLM: `greedy`, `constrained` (greedy + manufacturability rules), `random` |
| `session_cli.py` | Stateful CLI version of the tools, for a terminal agent (e.g. Claude Code) |
| `figures_v3.py` (uses `figures_v2.py` loaders) | Manuscript figures 3-5, S1-S3 and Table S2 |
| `extract_gpt_from_pdf.py` | Recovers per-step series from a vector matplotlib PDF |
| `digitize_old_fig4b.py`, `calib_fig4b.py` | Digitises the DRL histogram used in Fig. 4b |

## Reproduce

```bash
pip install -e '.[torch]' anthropic scikit-learn
# optional: the paper's CNN-MLP checkpoints -> models/original_cnn_mlp/ and add --backend cnn-mlp below
for s in 42 1 2 3 4 5; do for p in greedy constrained random; do
  python -m claude_opus.run_scaffold_baseline --policy $p --seed $s; done; done
export ANTHROPIC_API_KEY=...            # never commit keys
python -m claude_opus.run_tools_agent  --model claude-opus-5-5 --tag opus_tools_api
python -m claude_opus.run_conversation --model claude-opus-5-5 --tag opus_notools_api
python -m claude_opus.figures_v3
```

The runs reported in the paper are in `paper_results/claude_opus_5_5/`. The two Opus runs there were driven through `session_cli.py`; their decision logs are in `decision_logs/`.
