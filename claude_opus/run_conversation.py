"""Condition C: Claude Opus 5.5 in conversation only (no tools; like chatting in a CLI).

Same paper prompt, same starting sheet, same k=30 candidate neighbourhood. Claude never
sees the predictor. Each turn it picks an atom and writes its own flux/rejection
estimate. The harness still scores every design with the predictor, but only saves
that score to the log, so each step has both the model's claim and the predictor's value.

  python -m claude_opus.run_conversation --tag opus_notools_api [--backend cnn-mlp] [--model claude-opus-5-5]
"""
from __future__ import annotations

import argparse
import json
import re

import anthropic

from .env import DesalEnv, DesignConfig, REPO
from .prompts import CONVERSATION_ADDENDUM, paper_prompt


def observation(env: DesalEnv) -> str:
    s = env.state()   # blind env: contains no flux/rejection
    cands = " ".join(f"{c['serial']}({c['x']},{c['y']},b{c['bonds_to_pore']})" for c in env.describe_candidates())
    return (f"Step {s['step'] + 1} of {s['max_steps']}. Removed atoms so far ({s['removed_count']}): {s['removed_serials']}.\n"
            f"Pore geometry: {s['shape']}\n"
            f"Candidate atoms serial(x,y,bonds_to_pore): {cands}\nReply with the JSON object only.")


def parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backend", default="geometry-surrogate")
    ap.add_argument("--tag", default="opus_notools_api")
    ap.add_argument("--protocol", default="gpt61")
    a = ap.parse_args()
    env = DesalEnv(DesignConfig.preset(a.protocol, seed=a.seed, backend=a.backend), blind=True)
    client = anthropic.Anthropic()
    messages, transcript, feedback = [], [], ""
    while not env.done():
        messages.append({"role": "user", "content": (feedback + "\n" if feedback else "") + observation(env)})
        r = client.messages.create(model=a.model, max_tokens=1500,
                                   system=paper_prompt(env.cfg.min_rejection) + CONVERSATION_ADDENDUM, messages=messages)
        text = "".join(b.text for b in r.content if b.type == "text")
        messages.append({"role": "assistant", "content": text})
        d = parse(text)
        votes = [{"agent": k, "decision": str(v).upper()} for k, v in (d.get("votes") or {}).items()]
        try:
            atoms = d.get("atom_serials", d.get("atom_serial", []))
            atoms = [atoms] if isinstance(atoms, int) else list(atoms or [])
            env.commit_atoms([int(x) for x in atoms], votes=votes, rationale=d.get("rationale", ""),
                       claimed_flux=d.get("predicted_flux"), claimed_rejection=d.get("predicted_rejection"))
            feedback = "Accepted."
        except (ValueError, TypeError) as e:   # invalid atom -> step is spent, pore unchanged
            env.keep(rationale=f"invalid: {e}")
            feedback = f"Invalid choice ({e}); pore unchanged."
        transcript.append({"step": env.step, "reply": text, "feedback": feedback})
    out = REPO / "results" / "claude_opus"
    json.dump(transcript, open(out / f"{a.tag}_transcript.json", "w"), indent=1)
    print(env.save(out, a.tag))


if __name__ == "__main__":
    main()
