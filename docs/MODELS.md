# Predictor models

## Paper-matched predictor

The original workflow used separate ResNet50 + residual-MLP pipelines for water flux and ion rejection. The active implementation is `desalination_llm.predictors.cnn_mlp.CNNMLPPredictor`.

Required files are documented in `models/original_asset_manifest.csv`. The uploaded archive did not contain their bytes, so the repository does not claim to bundle them.

Import the assets from the original local project:

```bash
desalination-llm import-assets /home/akshay/Documents/MD/test_gemini \
  --destination models/original_cnn_mlp
```

Verify file hashes against `models/original_asset_manifest.csv` before publishing or running scientific reproduction.

## Included geometry surrogate

`models/geometry_surrogate.joblib` is trained on 3,000 archived CNN/MLP trial outputs using geometry and connectivity descriptors reconstructed from trial PDBs. It exists to make the repository functional without proprietary/large checkpoints.

Grouped by run, its recorded cross-validation performance is approximately:

- Flux: `R² = 0.929`, `MAE = 20.77 ns⁻¹`, `RMSE = 29.50 ns⁻¹`
- Rejection: `R² = 0.971`, `MAE = 0.0427`, `RMSE = 0.0604`

These metrics describe agreement with the archived predictors, not agreement with new MD simulations or experiments.
