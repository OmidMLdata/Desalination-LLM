from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import pandas as pd

from .agents import GeminiCLIPanel, NoAgentPanel, RuleBasedPanel
from .analysis import qualifying_designs, reproduce_figures
from .config import load_config
from .optimizer import GraphenePoreOptimizer
from .pdb import GrapheneStructure
from .predictors.cnn_mlp import CNNMLPPredictor
from .predictors.geometry_surrogate import GeometrySurrogatePredictor


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def build_predictor(config, structure: GrapheneStructure, base: Path):
    backend = config.predictor.backend.lower()
    if backend in {"geometry", "geometry-surrogate", "surrogate"}:
        return GeometrySurrogatePredictor(structure.xy, _resolve(base, config.predictor.model_path))
    if backend in {"cnn", "cnn-mlp", "paper"}:
        return CNNMLPPredictor(
            structure.xy,
            _resolve(base, config.predictor.model_dir),
            _resolve(base, config.predictor.labels_csv),
            config.predictor.device,
        )
    raise ValueError(f"Unknown predictor backend: {config.predictor.backend}")


def build_agents(config):
    mode = config.agents.mode.lower()
    if mode in {"none", "off"}:
        return NoAgentPanel()
    if mode in {"rule", "rule-based"}:
        return RuleBasedPanel(config.optimization.minimum_rejection)
    if mode in {"gemini", "gemini-cli"}:
        return GeminiCLIPanel(config.agents.command, config.optimization.minimum_rejection)
    raise ValueError(f"Unknown agent mode: {config.agents.mode}")


def command_optimize(args) -> None:
    config_path = Path(args.config).resolve()
    base = config_path.parent.parent if config_path.parent.name == "configs" else Path.cwd()
    config = load_config(config_path)
    structure = GrapheneStructure.from_pdb(_resolve(base, config.structure))
    predictor = build_predictor(config, structure, base)
    agents = build_agents(config)
    root_output = Path(args.output or _resolve(base, config.output_dir))
    all_results = []
    for run in range(config.runs):
        output = root_output / f"run_{run}"
        optimizer = GraphenePoreOptimizer(structure, predictor, config.optimization, agents)
        results, _ = optimizer.run(output, run_number=run, seed=config.optimization.random_seed + run)
        all_results.append(results)
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(root_output / "all_selected_steps.csv", index=False)
    print(f"Optimization outputs written to {root_output}")


def command_reproduce(args) -> None:
    paths = reproduce_figures(args.data, args.output)
    print("Created:")
    for path in paths:
        print(f"  {path}")


def command_summary(args) -> None:
    trials = pd.read_csv(args.trials)
    qualified = qualifying_designs(trials, args.flux_min, args.flux_max, args.rejection_min)
    signatures = qualified.get("removed_atom_serials", pd.Series(dtype=str)).nunique()
    print(f"Trial records: {len(trials)}")
    print(f"Qualifying records: {len(qualified)}")
    print(f"Unique removed-atom signatures: {signatures}")
    if len(qualified):
        best = qualified.sort_values(["rejection", "flux"], ascending=False).iloc[0]
        print(f"Highest-rejection qualifying design: {best['design_id']} | flux={best['flux']:.3f} | rejection={best['rejection']:.2%}")


def command_import_assets(args) -> None:
    source = Path(args.source)
    destination = Path(args.destination)
    destination.mkdir(parents=True, exist_ok=True)
    names = list(CNNMLPPredictor.REQUIRED_FILES) + ["noisy_train.csv"]
    for name in names:
        matches = list(source.rglob(name))
        if not matches:
            print(f"Missing: {name}")
            continue
        target = destination / name
        shutil.copy2(matches[0], target)
        print(f"Copied {matches[0]} -> {target}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="desalination-llm")
    sub = root.add_subparsers(dest="command", required=True)
    optimize = sub.add_parser("optimize", help="Run nanopore optimization")
    optimize.add_argument("--config", default="configs/demo.yaml")
    optimize.add_argument("--output")
    optimize.set_defaults(func=command_optimize)

    reproduce = sub.add_parser("reproduce", help="Regenerate figures from archived data")
    reproduce.add_argument("--data", default="data/processed")
    reproduce.add_argument("--output", default="figures/reproduced")
    reproduce.set_defaults(func=command_reproduce)

    summary = sub.add_parser("summary", help="Summarize the paper performance window")
    summary.add_argument("--trials", default="data/processed/canonical_trials.csv")
    summary.add_argument("--flux-min", type=float, default=150.0)
    summary.add_argument("--flux-max", type=float, default=170.0)
    summary.add_argument("--rejection-min", type=float, default=0.90)
    summary.set_defaults(func=command_summary)

    assets = sub.add_parser("import-assets", help="Copy original checkpoints from a local project tree")
    assets.add_argument("source")
    assets.add_argument("--destination", default="models/original_cnn_mlp")
    assets.set_defaults(func=command_import_assets)
    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
