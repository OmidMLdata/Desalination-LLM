# Claude Opus 5.5 for graphene-nanopore desalination: agent with tools vs conversation only

This add-on for the `Desalination-LLM` repository measures three contributions separately:

1. the **model** (Claude Opus 5.5's own judgment),
2. the **agent scaffold / tools** (candidate generation, the flux and rejection predictor, the step loop),
3. **tool use vs conversation (CLI chat)**, with the model held fixed.

To do that, every condition runs through one environment (`env.py`). All conditions share the same pristine 4040 sheet, the same k = 30 neighbourhood, the same seeded 20-trial sample per step and the same predictor. The only things that change are **who picks the atom** and **what that decision-maker can see**.

## Settings (paper Fig. 3)

30 steps, 20 trials per step, k = 30, one atom per step, rejection ≥ 60%. The prompt is the Figure 2 "User Query Prompt", copied word for word into `prompts.py`. Each Claude condition adds one short paragraph that only says which interface it has.

## Conditions

| Tag | Who decides | Sees predictor? | Purpose |
|---|---|---|---|
| `scaffold_greedy` | Python rule: highest flux with rejection ≥ 60% (the paper's rule) | yes | scaffold with no LLM |
| `scaffold_constrained` | Same rule, limited to contiguous, island-free trials | yes | tests whether a hand-written rule can do what the LLM adds |
| `scaffold_random` | A random feasible trial | yes | lower bound |
| `opus_tools` | Claude Opus 5.5, 3-expert vote, commits through tools | yes | **model + tools** |
| `opus_notools` | Claude Opus 5.5 in conversation; picks an atom and writes its own flux/rejection estimate | **no** | **model without tools** |

In `opus_notools`, every design is also scored by the predictor after the fact, and Claude never sees that score. So each step has two numbers: `claimed_*` (Claude's estimate) and `flux`/`rejection` (the predictor's value).

**What each comparison measures**

- Model judgment = `opus_tools` − `scaffold_greedy` (and − `scaffold_constrained`)
- Value of tool feedback = `opus_tools` − `opus_notools` (both scored by the predictor)
- Estimation error without tools = `opus_notools` claimed − `opus_notools` predictor score

A design counts as **valid** only if it meets all three conditions: rejection ≥ 60%, a single contiguous vacancy cluster, and no carbon island left inside the convex hull. These are the manuscript's manufacturability rules.

## Results in this package (geometry surrogate, seed 42)

| Condition | Best valid flux at rejection ≥ 90% | Hypervolume* | Invalid steps | Final design |
|---|---|---|---|---|
| Opus 5.5 + tools | **145.0** (step 18, 92.1%) | **52.8** | 0 / 30 | 166.5 ns⁻¹, 64.5%, 24 C |
| Scaffold, rule-constrained (seeds 42 + 1–5) | 128.9 ± 10.2 | 50.7 ± 0.7 | 0 | 161.2 ± 2.2 ns⁻¹, 62.3% |
| Scaffold, greedy (seeds 42 + 1–5) | 65.7 ± 55.8 | 23.1 ± 19.8 | 26 ± 3 | disconnected vacancies |
| Opus 5.5, no tools (predictor score) | 117.8 | 47.0 | 10 / 30 | 220.2 ns⁻¹, **45.9%**, 30 C |
| Opus 5.5, no tools (its own estimate) | 133.0 | 60.3 | – | claimed 188 ns⁻¹, 85.6% |

*Hypervolume is the area of (flux × rejection) space covered by the valid designs, measured from the point (0 ns⁻¹, 60% rejection). Higher is better.

- **Tools matter most.** With the model held fixed, adding predictor feedback raised the best flux at ≥ 90% rejection by about 27 ns⁻¹ and the final rejection by about 19 percentage points (64.5% vs 45.9%). Without tools, the pore passed the 60% limit at step 22 and Claude did not notice.
- **The no-tool estimates were well off.** Claude overestimated rejection by 15 pp on average (MAE 0.154) and underestimated flux by 12 ns⁻¹ (MAE 16.6). At 9 steps Claude believed the design was valid when the predictor put it below 60%. The flux trend was right; the rejection collapse after about 16 atoms was missed entirely.
- **The model's judgment adds value on top of the tools.** The plain greedy scaffold exploits the predictor with scattered, disconnected vacancies (26 of 30 steps invalid). Hand-coding the contiguity and no-island rules recovers most of the gap. Opus + tools still beat that rule on this seed: +16 ns⁻¹ at ≥ 90% rejection (about 1.6 sd of the rule's spread across seeds) and hypervolume 52.8 vs 50.7 ± 0.7. The gain came from giving up 1–2% flux for 1–6 pp of rejection at the rejection cliff (steps 14–21), and from stopping (keeping the pore) once every contiguous option fell below 60%.

## Caveats (read before citing)

- **The predictor is the repository's geometry surrogate, not the paper's ResNet50 + MLP.** The checkpoints are only on the original machine. Rerun with `--backend cnn-mlp` before quoting numbers in the paper.
- **The Opus runs here were done inside a Claude (Cowork) session.** The configured model was `claude-opus-5-5`; the serving model can differ. The agent drove `session_cli.py` as its tool interface and wrote the three expert votes inline rather than as three separate API calls. `run_tools_agent.py` and `run_conversation.py` are the API-based versions to use for the paper. Each makes separate expert calls and keeps a full transcript.
- **Each Opus condition has n = 1 seed.** The scaffold has 6. Run ≥ 5 seeds per Opus condition before claiming significance.
- **The no-tool run was done first, before any predictions for this setup were seen, but it was not fully blind.** The model had read the manuscript (e.g. 159.9 ns⁻¹ / 93.4% at 21 atoms) and the first rows of an archived CSV (step-1 flux ≈ 14–18). A model given the manuscript in the prompt would have the same information.
- The surrogate reports rejection as a fraction. The paper prompt says "Flux (L/m²/h)", but the models output ns⁻¹. The tools addendum tells the model which units it will get.

## How to run on your machine

```bash
cd Desalination-LLM && pip install -e '.[torch]' anthropic
desalination-llm import-assets /home/akshay/Documents/MD/test_gemini --destination models/original_cnn_mlp
export ANTHROPIC_API_KEY=...

# scaffold baselines (no LLM)
for s in 1 2 3 4 5; do for p in greedy constrained random; do
  python -m claude_opus.run_scaffold_baseline --policy $p --seed $s --backend cnn-mlp; done; done
# Opus 5.5 + tools (agent)
python -m claude_opus.run_tools_agent  --backend cnn-mlp --seed 42 --tag opus_tools_api
# Opus 5.5 conversation only
python -m claude_opus.run_conversation --backend cnn-mlp --seed 42 --tag opus_notools_api
# analysis: edit CONDS in analyze.py to point at the *_api tags, then
python -m claude_opus.analyze
```

To use Claude Code as the CLI agent (the counterpart of Gemini CLI in the paper), start it in the repository with the Figure 2 prompt and let it call
`python -m claude_opus.session_cli {init,observe,commit,save} --tag ...`.

## Files

`env.py` (shared environment) · `prompts.py` (Figure 2 prompt + expert prompts) · `run_tools_agent.py` · `run_conversation.py` · `run_scaffold_baseline.py` · `session_cli.py` · `analyze.py`.
Outputs are in `results/claude_opus/`: `*_per_step.csv` (flux and rejection at each step), `comparison_per_step.csv`, `summary_metrics.csv`, `decomposition.csv`, `notools_calibration.csv`, `scaffold_seed_summary.csv`, `fig_*.png`, the final PDBs, and `sessions/*.json` (the full decision logs of the Opus runs, with rationales and votes).
