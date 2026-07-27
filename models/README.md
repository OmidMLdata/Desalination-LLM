# Models

- `geometry_surrogate.joblib`: runnable secondary surrogate trained from archived trial predictions.
- `geometry_surrogate.metadata.json`: training provenance and grouped cross-validation metrics.
- `original_asset_manifest.csv`: filenames, sizes, timestamps, and SHA-256 hashes for the original paper predictor assets.
- `original_cnn_mlp/`: expected location for the four original checkpoints and `noisy_train.csv`; intentionally absent from Git by default.
