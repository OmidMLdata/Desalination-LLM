# GPT-6.1 Sol: matched six-category, higher-budget experiment

## Outcome

All six categories are complete under the requested higher-budget protocol.
Both GPT arms use new authenticated `gpt-6.1-sol` calls; neither is a relabeled
`gpt-6-astra` run. All previous experiment data and figures remain unchanged.

[Main figure](six_category_comparison.png) ·
[Rejection zoom](six_category_comparison_zoom.png) ·
[Vector PDF](six_category_comparison.pdf) ·
[Full-range companion](six_category_comparison_fullrange.png)

Final values at decision 30 are below. Scaffold entries are mean ± sample SD
across six seeds; each GPT arm has one rollout and no uncertainty estimate.

| Category | Flux (ns^-1) | Ion rejection (%) |
|---|---:|---:|
| GPT 6.1 with tools | 131.71 | 90.23 |
| GPT 6.1 no tools, predictor score afterward | 47.30 | 86.22 |
| GPT 6.1 no tools, its own estimate | 2.00 | 99.10 |
| Scaffold only, greedy | 157.11 ± 3.09 | 90.72 ± 0.57 |
| Scaffold only, greedy + manufacturability rules | 157.11 ± 3.09 | 90.72 ± 0.57 |
| Scaffold only, random feasible | 60.77 ± 12.56 | 92.86 ± 1.94 |

The with-tools GPT rollout maintains the 90% predicted-rejection floor but
does not reach the 150–170 ns^-1 target. The blind trajectory falls below the
floor at two steps and finishes below it. Its self-estimates overstate final
rejection and understate final flux relative to the frozen predictor; these
are unvalidated estimates, not additional physical evidence.

Checked and unchecked greedy rules select identical states at every step in
all six matched seeds. Some trial proposals fail extra topology checks, but
those proposals do not change the winning greedy choices in these runs. The
two rule curves therefore coincide. A solid green line and dashed gold line
show both without offsetting or changing their values. Their SD bands also
overlap; they are not independent confirmation of two different outcomes.

## Shared protocol

- Pristine source sheet: 680 atoms; start atom ID 301, with no warm start.
- 30 decisions after the scored initial state.
- Up to 30 unique trials INCLUDING KEEP per decision; no padding with duplicates.
- Up to 3 actual new removals per chosen trial INCLUDING cleanup, at most 90 total.
- Common bonded nearest-30 frontier, edge exclusion, explicit cleanup, and budget limits.
- Minimum predicted rejection: 90% for all feedback-based policies.
- Extra pore/topology checks for GPT with tools, checked greedy, and random feasible.
- Uniform random selection among feasible options INCLUDING KEEP, using a
  selection RNG separate from candidate generation.
- Seeds 20261012–20261017 for each scaffold policy; both GPT arms use 20261012.
- GPT-6.1 Sol with low reasoning effort in both GPT arms, preserving the prior
  effective reasoning setting. Model requests, prompts, responses, and events are saved.

Initialization and the first seed were selected by earlier successful searches.
This is a matched comparison at an optimization-selected starting condition,
not an unbiased initialization study or a causal estimate of intrinsic model ability.

To make checked-versus-unchecked greedy meaningful, the source group sampler's
quality filtering was moved from proposal generation to policy selection for
EVERY new arm. Connected growth, cleanup, and atom limits remain common. This
controlled amendment is documented rather than silently mixing candidate
generators. The checked replicate-0 trajectory nevertheless reproduces the
previous high-flux checked trajectory (162.91 ns^-1, 91.58% final rejection).
The new GPT 6.1 trajectory is different; previous Astra values are not substituted.

## No-tools separation

The blind GPT arm receives candidate atom IDs, coordinates, bonds, and its own
prior estimates through the CLI conversation. It receives no CNN properties
or computed pore-quality flags and makes zero tool calls. All 30 decisions and
estimates are committed in `runs/gpt_blind_seed0/plan.json` before constructing
the CNN oracle in that process. The plan hash and scoring-start timestamp are
saved. The plan is neither repaired nor regenerated after seeing predictions.

The predictor-score and self-estimate curves describe the SAME blind geometry
sequence, not independent experiments. The idealized zero-flux/100% rejection
self-estimate at step zero is labeled as a manuscript prompt anchor. The scored
curves retain the actual saved pristine surrogate output instead.

The with-tools arm makes 61 completed native scientific tool calls, including
30 successful commits. It makes no non-domain tool calls. One preliminary CLI
attempt did not expose the graphene tools and produced no scientific choices;
it is preserved under `failed_tool_connection_attempt_01/`. The connection was
corrected with the server explicitly enabled and required, then a fresh valid
tool trajectory was run. No numerical or rule-based fallback stood in for GPT.

## Evaluator and validation

The supplied renderer, RGB conversion, crop 380, resize 224, RGB/255 tensors,
strict all2 ResNet50/MLP checkpoints, and paired `mean_std_all.pickle` are unchanged.
Verified archived cache values are reused only for identical geometries and
matching checkpoint/statistic hashes. All 620 scored recorded states were
independently checked with fresh inference afterward.

- 20 structural trajectories, each with steps 0–30; 651 records including the
  additional self-estimate series for the blind trajectory.
- All actual-removal limits, trial uniqueness, selected trial membership, and
  applicable rejection/topology constraints checked.
- Final PDBs match the logged removed atoms; saved pore images match the renderer.
- Maximum fresh-rescore absolute differences: 0.0000763 ns^-1 in flux and
  0.000000239 in rejection fraction, within the recorded numerical tolerance.
- Blind structural proposals and choices independently replayed; guesses match
  archived model responses exactly. Native commits match accepted GPT records.
- SHA-256 checks confirm all 1,108 historical artifacts are unchanged.
- 23 logic, regression, provenance, and saved-data tests pass.

Pore checks are heuristic design constraints, not a chemical-manufacturability
certification. All transport values are CNN surrogate predictions, not new MD
or experimental measurements. One rollout per GPT arm does not establish
run-to-run reliability, and the two transports have different scaffolding.

## Files and reproduction

`protocol.json`, `frozen_inputs.json`, `verification.json`, and
`plot_integrity.json` contain the protocol, hashes, and verification evidence.
`six_category_per_step.csv` preserves all individual records;
`six_category_summary.csv` contains the six mean/SD curves. No smoothing or
numerical inflation is applied. `runs/` contains all actual trial tables,
decisions, resumable checkpoints, model calls, and final pores.

PNG and LZW-compressed TIFF exports are 600 dpi; PDF embeds TrueType fonts and
SVG retains editable text. The main rejection axis begins at 50%; the zoom and
full-range companion use the same saved data.

From `/home/akshay/Documents/MD/GPT`, regenerate figures without model calls:

```bash
python src/high_budget_six.py plot
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests
```

The `prepare`, `baselines`, `tools`, `blind`, and `verify` subcommands provide
the separate experiment phases. Completed model arms are retained; unfinished
runs resume committed state rather than synthesizing missing results.

Official model documentation:
[GPT-6.1 Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol).
