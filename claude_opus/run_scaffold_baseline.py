"""Scaffold-only baselines (no LLM in the loop).

greedy      : the paper's rule - highest-flux trial meeting the rejection floor
              (keep the pore if none does).
constrained : greedy restricted to contiguous, island-free trials - tests whether a
              hand-written manufacturability rule reproduces what the LLM adds.
random      : a uniformly random feasible trial (lower bound for the scaffold).

  python -m claude_opus.run_scaffold_baseline --policy greedy --seed 42 [--protocol gpt61] [--backend cnn-mlp]
"""
from __future__ import annotations

import argparse

import numpy as np

from .env import DesalEnv, DesignConfig, REPO


def pick(trials, policy, rng):
    feas = [t for t in trials if t["feasible"] and t["n_new"] > 0]
    if policy == "constrained":
        feas = [t for t in feas if t["connected_components"] == 1 and t["internal_unremoved_count"] == 0]
    if not feas:
        return -1
    if policy == "random":
        return int(rng.choice([t["trial"] for t in feas]))
    return max(feas, key=lambda t: (t["flux"], t["rejection"]))["trial"]


def run(policy="greedy", seed=42, protocol="gpt61", backend="geometry-surrogate", out=None, tag=None):
    env = DesalEnv(DesignConfig.preset(protocol, seed=seed, backend=backend))
    rng = np.random.default_rng(seed + 1000)
    while not env.done():
        env.commit_trial(pick(env.evaluate_trials(), policy, rng), rationale=f"scaffold:{policy}")
    tag = tag or f"scaffold_{policy}_s{seed}"
    return env.save(out or REPO / "results" / "claude_opus", tag), env


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["greedy", "random", "constrained"], default="greedy")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--protocol", default="gpt61")
    ap.add_argument("--backend", default="geometry-surrogate")
    a = ap.parse_args()
    print(run(a.policy, a.seed, a.protocol, a.backend)[0])
