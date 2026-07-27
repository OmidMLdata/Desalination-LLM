# Repository audit and reorganization

## Source archive reviewed

- `gemini.tar.xz`: 86,111 members, approximately 2.37 GiB uncompressed
- Dominant file types: 44,257 PNG files and 41,553 PDB files
- Active canonical experiment: `gemini/Trials/geminit20r3k30`

## Problems corrected

- Replaced hard-coded `/home/akshay/...` paths with configuration-driven relative paths.
- Removed dependency on a graphical Matplotlib backend.
- Removed duplicate active scripts while retaining representative originals under `legacy/`.
- Separated active code, archived results, prompts, models, figures, and manuscript files.
- Reconstructed removed-atom identities from PDB serial numbers.
- Added a testable predictor interface and an immediately runnable surrogate backend.
- Added deterministic random seeds, validation, portable CSV paths, packaging metadata, and CI.
- Excluded `.env`, notebook checkpoints, `__pycache__`, and large raw trajectories from the Git repository.

## Items requiring author review before public release

1. Replace the placeholder GitHub URL in `CITATION.cff`.
2. Select an explicit code/data license; the current file is conservative and all-rights-reserved.
3. Confirm the journal policy for sharing `paper/manuscript.pdf`.
4. Decide whether original CNN/MLP checkpoints may be redistributed through Git LFS or an external DOI-backed archive.
5. Add the final publication DOI and versioned release tag.
