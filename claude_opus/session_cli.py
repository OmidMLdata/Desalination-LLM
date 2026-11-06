"""Stateful command-line front end to DesalEnv: the interface an LLM running in a terminal
agent (Claude Code / Cowork, like Gemini CLI in the paper) uses as its tools.
The environment is pickled between calls; every decision is also appended to a JSON log.

  python -m claude_opus.session_cli init    --tag T --mode tools|blind [--protocol gpt61] [--backend cnn-mlp]
  python -m claude_opus.session_cli observe --tag T                 # tools: trials; blind: candidates only
  python -m claude_opus.session_cli commit  --tag T --trial N       # tools mode (-1 = keep)
  python -m claude_opus.session_cli commit  --tag T --atoms "a b c" --claimed-flux F --claimed-rejection R   # blind
  python -m claude_opus.session_cli save    --tag T
"""
from __future__ import annotations

import argparse, json, pickle
from pathlib import Path

from .env import DesalEnv, DesignConfig, REPO

RUNS = REPO / "results" / "claude_opus" / "sessions"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["init", "observe", "commit", "save"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mode", choices=["tools", "blind"], default="tools")
    ap.add_argument("--protocol", default="gpt61")
    ap.add_argument("--backend", default="geometry-surrogate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trial", type=int)
    ap.add_argument("--atoms", default="")
    ap.add_argument("--votes", default="[]")
    ap.add_argument("--rationale", default="")
    ap.add_argument("--claimed-flux", type=float)
    ap.add_argument("--claimed-rejection", type=float)
    ap.add_argument("--out", default=str(REPO / "results" / "claude_opus"))
    a = ap.parse_args(argv)
    RUNS.mkdir(parents=True, exist_ok=True)
    pk, lg = RUNS / f"{a.tag}.pkl", RUNS / f"{a.tag}.json"
    if a.cmd == "init":
        env = DesalEnv(DesignConfig.preset(a.protocol, seed=a.seed, backend=a.backend), blind=a.mode == "blind")
        pickle.dump(env, open(pk, "wb"))
        lg.write_text(json.dumps({"mode": a.mode, "protocol": a.protocol, "seed": a.seed, "commits": []}, indent=1))
        print(f"initialised {a.tag} ({a.mode}, {a.protocol})"); return
    env = pickle.load(open(pk, "rb")); log = json.loads(lg.read_text())
    if a.cmd == "observe":
        out = {"state": env.state()}
        if env.blind:
            out["candidates_k"] = env.describe_candidates()
        else:
            out["trials"] = env.evaluate_trials()
        print(json.dumps(out, separators=(",", ":")))
    elif a.cmd == "commit":
        votes = json.loads(a.votes)
        if env.blind:
            atoms = [int(x) for x in a.atoms.replace(",", " ").split()]
            res = env.commit_atoms(atoms, votes=votes, rationale=a.rationale,
                                   claimed_flux=a.claimed_flux, claimed_rejection=a.claimed_rejection)
            entry = {"atoms": atoms}
        else:
            res = env.commit_trial(a.trial, votes=votes, rationale=a.rationale)
            entry = {"trial": a.trial, "new_atoms": res.get("new_atoms", [])}
        log["commits"].append({**entry, "votes": votes, "rationale": a.rationale,
                               "claimed_flux": a.claimed_flux, "claimed_rejection": a.claimed_rejection})
        lg.write_text(json.dumps(log, indent=1))
    if a.cmd in ("commit",):
        pickle.dump(env, open(pk, "wb"))
        print(json.dumps(res, separators=(",", ":")))
    if a.cmd == "observe" and not env.blind:
        pickle.dump(env, open(pk, "wb"))       # keep the evaluated trials (rng already advanced)
    if a.cmd == "save":
        print(env.save(a.out, a.tag))


if __name__ == "__main__":
    main()
