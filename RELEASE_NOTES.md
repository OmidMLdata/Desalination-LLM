# Release notes

## v0.2.0: LLM benchmark for the revised manuscript

- **Benchmark module `claude_opus/`:** a shared design environment that compares six conditions per model under one protocol:
  - LLM + tools;
  - LLM in conversation only, scored by the predictor and as estimated by the model;
  - three scaffold-only baselines.

  The protocol is 30 steps, up to 30 trials per step, at most 3 atoms per trial and a 90 % rejection floor.
- **Claude Opus 5.5 runs:** per-step and per-trial results, decision logs, final structures, and 18 scaffold baseline runs (`paper_results/claude_opus_5_5/`).
- **GPT-6.1 Sol run:** the screened raw run folder (`paper_results/gpt_6_1_sol/raw/`) and the six-category per-step table built from it (`gpt61_per_step.csv`).
- **Reference data for Figure 4b:** MD dataset labels and the digitized DRL histogram of Wang et al. (2021).
- **Figures:** all revised-manuscript figures and tables, reproducible from the committed data with `claude_opus/figures_v3.py`.
- **`scripts/import_gpt61_results.py`:** screens a run folder before publication, skipping per-trial files, binaries and anything that looks like a secret.
- **Metadata:** README, CITATION.cff and package metadata updated, and a new `llm` optional dependency.

## v0.1.0

- Converted the uploaded 86,111-member archive into a compact GitHub-ready repository.
- Added a portable Python package, CLI, configuration files, tests, and CI.
- Reconstructed 3,000 canonical trial records and removed-atom signatures from PDB files.
- Included 28 qualifying high-performance trial assets, representing 23 unique structures.
- Added an immediately runnable geometry surrogate with grouped cross-validation metadata.
- Added an exact ResNet50 + residual-MLP inference adapter and original-asset checksum manifest.
- Preserved representative original scripts and prompts under `legacy/`.
- Removed credentials, caches, absolute paths from active code, and bulk intermediate assets.
