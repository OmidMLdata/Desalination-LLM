from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
import shlex
import subprocess
from typing import Protocol

from .predictors.base import Prediction


@dataclass(frozen=True, slots=True)
class AgentVote:
    agent: str
    decision: str
    rationale: str


class AgentPanel(Protocol):
    def review(self, atom_serials: list[int], prediction: Prediction, previous: Prediction) -> list[AgentVote]:
        ...


class NoAgentPanel:
    def review(self, atom_serials: list[int], prediction: Prediction, previous: Prediction) -> list[AgentVote]:
        return []


class RuleBasedPanel:
    """Deterministic stand-in for the three paper prompt roles."""

    def __init__(self, minimum_rejection: float = 0.60):
        self.minimum_rejection = minimum_rejection

    def review(self, atom_serials: list[int], prediction: Prediction, previous: Prediction) -> list[AgentVote]:
        feasible = prediction.rejection >= self.minimum_rejection
        improves_flux = prediction.flux >= previous.flux
        modest_loss = prediction.rejection >= previous.rejection - 0.08
        decisions = [
            ("science", feasible and modest_loss, "Preserves the rejection constraint with a limited selectivity loss."),
            ("geometry", feasible, "Candidate remains inside the geometry-controlled optimization loop."),
            ("flux", feasible and improves_flux, "Improves or preserves throughput while meeting rejection."),
        ]
        return [
            AgentVote(name, "REMOVE" if approved else "KEEP", rationale)
            for name, approved, rationale in decisions
        ]


class GeminiCLIPanel:
    """Optional Gemini CLI reviewer.

    The executable is configurable because Gemini CLI invocation details may differ
    by installation. The command must accept a prompt on stdin and return text.
    """

    def __init__(self, command: str = "gemini", minimum_rejection: float = 0.60):
        self.command = command
        self.minimum_rejection = minimum_rejection

    @staticmethod
    def _parse(text: str, agent: str) -> AgentVote:
        decision_match = re.search(r"DECISION\s*:\s*(REMOVE|KEEP)", text, re.I)
        rationale_match = re.search(r"RATIONALE\s*:\s*(.*?)(?:\n\s*DECISION|$)", text, re.I | re.S)
        decision = decision_match.group(1).upper() if decision_match else "KEEP"
        rationale = rationale_match.group(1).strip() if rationale_match else text.strip()[:500]
        return AgentVote(agent, decision, rationale)

    def _query(self, prompt: str, agent: str) -> AgentVote:
        command = self.command
        if "{prompt}" in command:
            args = shlex.split(command.format(prompt=prompt))
            stdin = None
        else:
            args = shlex.split(command)
            if "-p" in args or "--prompt" in args:
                args.append(prompt)
                stdin = None
            else:
                stdin = prompt
        process = subprocess.run(
            args, input=stdin, text=True, capture_output=True, check=False, timeout=120
        )
        if process.returncode != 0:
            return AgentVote(agent, "KEEP", f"Gemini CLI error: {process.stderr.strip()[:300]}")
        return self._parse(process.stdout, agent)

    def review(self, atom_serials: list[int], prediction: Prediction, previous: Prediction) -> list[AgentVote]:
        context = (
            f"Candidate atom serials: {atom_serials}. Predicted flux={prediction.flux:.3f} ns^-1, "
            f"rejection={prediction.rejection:.2%}. Previous flux={previous.flux:.3f}, "
            f"previous rejection={previous.rejection:.2%}. Minimum rejection={self.minimum_rejection:.2%}."
        )
        prompts = {
            "science": "You are a nanoscience researcher. Assess literature-consistent desalination performance.\n" + context,
            "geometry": "You are a materials physicist. Prefer cohesive polygonal/Ozark-like pore growth.\n" + context,
            "flux": "You are a desalination engineer. Prioritize flux without violating rejection.\n" + context,
        }
        suffix = "\nRespond exactly as:\nRATIONALE: <brief reason>\nDECISION: REMOVE or KEEP"
        return [self._query(prompt + suffix, name) for name, prompt in prompts.items()]


def votes_to_json(votes: list[AgentVote]) -> str:
    return json.dumps([asdict(vote) for vote in votes], ensure_ascii=False)
