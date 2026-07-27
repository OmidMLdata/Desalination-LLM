# Contributing

Please open an issue before making substantial scientific changes. Contributions should:

- preserve physical units and identify prediction versus MD/experimental values;
- include a deterministic seed and configuration file;
- add or update tests;
- avoid committing credentials or large raw trajectories;
- document model/checkpoint provenance and hashes;
- retain archived results unchanged, placing new experiments under a new versioned directory.

Run before submitting:

```bash
pytest
ruff check src tests scripts
```
