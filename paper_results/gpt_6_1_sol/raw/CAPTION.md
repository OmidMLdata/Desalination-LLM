# GPT-6.1 Sol: six-category matched higher-budget comparison

Every GPT curve comes from NEW authenticated `gpt-6.1-sol` calls (low reasoning
effort), not renamed Astra results. The three scaffold baselines each have six
seeds; each GPT arm has one rollout and no run-to-run uncertainty estimate.
The two no-tools curves describe the SAME blind geometry sequence: GPT's own
unvalidated numerical estimates, and frozen-CNN scores computed only after
all 30 decisions and estimates were committed. Step-zero self-estimates use
the idealized manuscript prompt anchor, not an inferred transport measurement.

All policies start from the pristine 680-atom sheet with central start atom
ID 301. There are 30 decisions, up to 30 UNIQUE trials INCLUDING KEEP, up to
3 ACTUAL newly removed atoms per trial INCLUDING cleanup, and at most 90 total
removals. KEEP consumes a decision. The first seed is 20261012, matched between
the GPT arms and baseline replicate 0; baseline seeds are 20261012–20261017.
The start/first seed were selected by earlier successful optimization, so
this is not an unbiased initialization search or proof of model superiority.

The bonded nearest-30 frontier, explicit under-coordination cleanup, edge
exclusion, and removal limits are common structural scaffolding. Relative to
the earlier publication plot's sampler, quality filtering has been moved
FROM proposal generation TO selection, consistently for EVERY policy. This
controlled amendment permits a real checked-versus-unchecked rule comparison:
greedy enforces the 90% rejection floor but not the extra pore checks; checked
greedy and random feasible enforce both. Random selection is uniform over
feasible trials INCLUDING KEEP using a separate documented RNG stream. Blind
choices are not repaired or filtered using post-hoc scores. Pore checks are
heuristics, not chemical manufacturability or MD validation. Saved per-step
tables retain all geometry validity and rejection-floor flags.

The supplied source renderer, crop 380, resize 224, RGB/255, exact all2
ResNet50/MLP checkpoints, and paired mean_std_all.pickle remain unchanged.
Verified cache values may be reused for identical geometries; all 620 scored
recorded states were independently checked with fresh inference. All original
experiment files and publication-plot data remain unchanged. These new GPT
6.1 curves may differ from Astra, even under similar search budgets.

Each baseline line is the mean and its band is sample SD (six seeds), not a
confidence interval. No smoothing, numerical inflation, guessed predictor
values, or uncertainty for single GPT rollouts is introduced. The main plot
starts rejection at 50%; the full-range companion shows every recorded
rejection mean. All transport scores are SURROGATE PREDICTIONS, not MD or
experimental measurements. PNG/TIFF are 600 dpi; PDF/SVG are vector exports.

See protocol.json, frozen_inputs.json, verification.json, the original CLI
prompts/responses/event logs, and plot_integrity.json for full provenance.
Model documentation: https://developers.openai.com/api/docs/models/gpt-6.1-sol
