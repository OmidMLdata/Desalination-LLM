# User query prompt used in the revised benchmark

The Figure 2 prompt, with the rejection floor set to 90 %. Both models received this text. The interface note that follows it depends on the condition.

## Prompt

```text
You are a nanoscience research agent working to optimize a graphene nanopore for water desalination. Your goal is to iteratively remove carbon atoms from a graphene sheet to create a pore that:
1. Start from the center of the graphene sheet, where no atoms are removed yet.
2. At each step select candidate atoms around the current pore boundary and evaluate up to 20 trials using a CNN + MLP model that predicts:
     - Flux (L/m²/h)
     - Ion rejection (%)
   - Only accept atoms if rejection ≥ 90% and flux improves.
3. Use multiple expert agents with different decision philosophies:
   - A science-based agent referencing published nanopore research.
   - A geometry-aware agent ensuring polygonal pore evolution.
   - A flux-prioritizing agent optimizing throughput.
Repeat this process until 30 atoms are removed or no better atom can be found.
```

## LLM + tools (agent): interface note

```text
Interface: you are running as an agent with tools. Settings: 30 steps; each step the scaffold generates up to 30 trials, each removing 0-3 atoms drawn from the k=30 nearest unremoved atoms (atoms trapped inside the pore's convex hull are colonised, at most 3 new atoms per step).
Each step: call evaluate_trials, choose one trial number (or -1 to keep the pore unchanged), call consult_expert_panel on it, then call commit_step. A removal is applied only if at least 2 of the 3 experts vote REMOVE. Flux is reported in ns^-1 (water molecules per ns) and rejection as a fraction.
```

## LLM no tools: interface note

```text
Interface: conversation only. You have NO access to the CNN + MLP predictor or to any other tool. At each step you will be given the current pore (removed atom serials) and the k=30 candidate atoms with their coordinates (Angstrom) and number of bonds to the pore. Settings: 30 steps, at most 3 atoms removed per step (include any atom that would be left inside the pore yourself).
Act as all three expert agents yourself, then reply with ONLY a JSON object:
{"atom_serials": [<up to 3 ints>, or [] to keep the pore], "votes": {"science": "REMOVE|KEEP", "geometry": "REMOVE|KEEP", "flux": "REMOVE|KEEP"},
 "predicted_flux": <your estimate for the pore after this step, ns^-1>, "predicted_rejection": <your estimate, fraction 0-1>, "rationale": "<one sentence>"}
Your predicted_flux and predicted_rejection are your own estimates; nothing will tell you the true values.
```

The three expert prompts are in `claude_opus/prompts.py` and `prompts/multi_agent_prompts.yaml`.
