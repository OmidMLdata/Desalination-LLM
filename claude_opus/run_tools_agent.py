"""Condition B: Claude Opus 5.5 as a tool-using agent (Anthropic Messages API tool use).

The system prompt is the paper's Figure 2 prompt unchanged, plus a note on which tools
are available. The tools wrap the same DesalEnv that the scaffold and conversation
conditions use. The three expert agents are three separate Claude calls using the
repository's multi_agent_prompts.

  export ANTHROPIC_API_KEY=...
  python -m claude_opus.run_tools_agent --seed 42 --tag opus_tools_api [--backend cnn-mlp] [--model claude-opus-5-5]
"""
from __future__ import annotations

import argparse
import json
import re
import time

import anthropic

from .env import DesalEnv, DesignConfig, REPO
from .prompts import EXPERT_PROMPTS, TOOLS_ADDENDUM, paper_prompt

TOOLS = [
    {"name": "get_pore_state",
     "description": "Current step, removed atom serials, current predicted flux/rejection and pore geometry (hull area, perimeter, circularity, aspect ratio, connected components, internal unremoved atoms).",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "evaluate_trials",
     "description": "Run this step's trials (up to 30): each removes 0-3 atoms sampled from the k=30 nearest unremoved atoms, plus colonisation, and is scored by the predictor. Returns trial number, new_atoms, flux (ns^-1), rejection (fraction), feasible (rejection >= floor) and the geometry of the pore after the removal (hull area, circularity, aspect ratio, connected_components, internal_unremoved_count). Repeat calls return the same trials.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "consult_expert_panel",
     "description": "Ask the science, geometry and flux expert agents whether to remove this trial atom. Returns three REMOVE/KEEP votes with rationales.",
     "input_schema": {"type": "object", "properties": {"trial": {"type": "integer"}}, "required": ["trial"]}},
    {"name": "commit_step",
     "description": "Finish the step by accepting one of this step's trials. The expert panel must already have been consulted on it; the pore is updated only if >=2 experts voted REMOVE. Pass -1 to keep the pore unchanged.",
     "input_schema": {"type": "object", "properties": {"trial": {"type": "integer"}, "rationale": {"type": "string"}},
                      "required": ["trial", "rationale"]}},
]


class ToolAgent:
    def __init__(self, env: DesalEnv, model: str, expert_model: str | None = None):
        self.env, self.model = env, model
        self.expert_model = expert_model or model
        self.client = anthropic.Anthropic()
        self.panel_cache: dict[int, list[dict]] = {}
        self.usage = {"input_tokens": 0, "output_tokens": 0, "api_calls": 0}

    def _call(self, **kw):
        for attempt in range(6):
            try:
                r = self.client.messages.create(**kw)
                self.usage["input_tokens"] += r.usage.input_tokens
                self.usage["output_tokens"] += r.usage.output_tokens
                self.usage["api_calls"] += 1
                return r
            except (anthropic.RateLimitError, anthropic.APIStatusError) as e:  # back off and retry
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)

    # ---------------------------------------------------------------- tools
    def _panel(self, trial: int) -> list[dict]:
        trials = {t["trial"]: t for t in self.env.evaluate_trials()}
        if trial not in trials:
            return [{"agent": "system", "decision": "KEEP", "rationale": "not a trial of this step"}]
        t, cur = trials[trial], self.env.current
        serial = t["new_atoms"]
        shape = {k: t[k] for k in ("hull_area", "circularity", "aspect_ratio", "connected_components", "internal_unremoved_count")}
        votes = []
        for name, tmpl in EXPERT_PROMPTS.items():
            prompt = tmpl.format(atom_serials=serial, flux=t["flux"], rejection=t["rejection"],
                                 prev_flux=cur.flux, prev_rejection=cur.rejection, shape=shape)
            r = self._call(model=self.expert_model, max_tokens=400, messages=[{"role": "user", "content": prompt}])
            text = "".join(b.text for b in r.content if b.type == "text")
            d = re.search(r"DECISION\s*:\s*(REMOVE|KEEP)", text, re.I)
            ra = re.search(r"RATIONALE\s*:\s*(.*?)(?:\n\s*DECISION|$)", text, re.I | re.S)
            votes.append({"agent": name, "decision": d.group(1).upper() if d else "KEEP",
                          "rationale": (ra.group(1) if ra else text).strip()[:400]})
        self.panel_cache[trial] = votes
        return votes

    def dispatch(self, name: str, args: dict):
        if name == "get_pore_state":
            return self.env.state()
        if name == "evaluate_trials":
            return self.env.evaluate_trials()
        if name == "consult_expert_panel":
            return self._panel(int(args["trial"]))
        if name == "commit_step":
            serial = int(args["trial"])
            votes = [] if serial < 0 else self.panel_cache.get(serial)
            if votes is None:
                return {"error": "consult_expert_panel on this trial before committing"}
            try:
                res = self.env.commit_trial(serial, votes=votes, rationale=args.get("rationale", ""))
            except ValueError as e:
                return {"error": str(e)}
            self.panel_cache.clear()
            return res
        return {"error": f"unknown tool {name}"}

    # ---------------------------------------------------------------- loop
    def run(self, max_turns: int = 400, log_path=None):
        messages = [{"role": "user", "content": "Begin the design now. Step 1: the sheet is pristine (flux = 0, rejection = 100%)."}]
        transcript = []
        for _ in range(max_turns):
            if self.env.done():
                break
            r = self._call(model=self.model, max_tokens=4000, system=paper_prompt(self.env.cfg.min_rejection) + TOOLS_ADDENDUM,
                           tools=TOOLS, messages=messages)
            messages.append({"role": "assistant", "content": r.content})
            calls = [b for b in r.content if b.type == "tool_use"]
            transcript.append({"text": "".join(b.text for b in r.content if b.type == "text"),
                               "tool_calls": [{"name": c.name, "input": c.input} for c in calls]})
            if not calls:   # the model stopped without finishing; tell it to continue
                messages.append({"role": "user", "content": f"Continue with step {self.env.step + 1}."})
                continue
            results = []
            for c in calls:
                out = self.dispatch(c.name, c.input)
                results.append({"type": "tool_result", "tool_use_id": c.id, "content": json.dumps(out)})
            messages.append({"role": "user", "content": results})
            messages = self._trim(messages)
        if log_path:
            json.dump({"transcript": transcript, "usage": self.usage,
                       "predictor_calls": self.env.n_predictor_calls}, open(log_path, "w"), indent=1, default=str)

    @staticmethod
    def _trim(messages, keep_last=24):
        """Keep the first message plus the most recent turns so 30 steps fit in context.
        The cut is placed just before an assistant turn so tool_use/tool_result pairs stay together."""
        if len(messages) <= keep_last + 1:
            return messages
        tail = messages[-keep_last:]
        while tail and tail[0]["role"] != "assistant":
            tail = tail[1:]
        return [messages[0]] + tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--expert-model", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backend", default="geometry-surrogate")
    ap.add_argument("--tag", default="opus_tools_api")
    ap.add_argument("--protocol", default="gpt61")
    a = ap.parse_args()
    env = DesalEnv(DesignConfig.preset(a.protocol, seed=a.seed, backend=a.backend))
    out = REPO / "results" / "claude_opus"
    out.mkdir(parents=True, exist_ok=True)
    agent = ToolAgent(env, a.model, a.expert_model)
    agent.run(log_path=out / f"{a.tag}_transcript.json")
    print(env.save(out, a.tag), agent.usage)


if __name__ == "__main__":
    main()
