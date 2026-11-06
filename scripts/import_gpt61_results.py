"""Screen the GPT-6.1 Sol run folder and copy the items the manuscript relies on into
paper_results/gpt_6_1_sol/raw/ (dry run by default).

Kept      : tables and logs (csv, tsv, json, jsonl, yaml, txt, md), the scripts that produced
            the run (py, ipynb), final/selected PDB structures, and summary figures.
Skipped   : per-trial images and PDBs, pickles, arrays, model checkpoints, files > 20 MB,
            and any text file that looks like it contains an API key or other secret.
CSV/JSON files larger than 1 MB are stored gzip-compressed. A MANIFEST.tsv with SHA-256
hashes records every copied file and why anything else was left out.

  python scripts/import_gpt61_results.py /home/akshay/Documents/MD/GPT/results_gpt61_t30_g3_20261015
  python scripts/import_gpt61_results.py <folder> --apply
"""
from __future__ import annotations

import argparse, gzip, hashlib, re, shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEST = REPO / "paper_results" / "gpt_6_1_sol" / "raw"
TEXT = {".csv", ".tsv", ".json", ".jsonl", ".yaml", ".yml", ".txt", ".md", ".log", ".py", ".ipynb", ".cfg", ".toml"}
FIG = {".pdf", ".png", ".svg", ".gif"}
SKIP_EXT = {".pkl", ".pickle", ".npy", ".npz", ".ckpt", ".pt", ".pth", ".h5", ".joblib", ".dcd", ".xtc", ".trr"}
SECRET = re.compile(r"(sk-[A-Za-z0-9_\-]{20,}|sk-proj-[A-Za-z0-9_\-]{20,}|api[_-]?key\s*[=:]\s*['\"][^'\"]{12,}|"
                    r"AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,}|xox[bap]-[A-Za-z0-9-]{10,})", re.I)
PER_TRIAL = re.compile(r"(trial[_-]?\d+|_t\d+_|step_\d+_\d+)", re.I)
MAX_BYTES = 20 * 1024 * 1024


def classify(p: Path, root: Path) -> tuple[str, str]:
    ext, name, size = p.suffix.lower(), p.name.lower(), p.stat().st_size
    rel = str(p.relative_to(root)).lower()
    if any(part.startswith(".") for part in p.relative_to(root).parts) or name in {".env"}:
        return "skip", "hidden file"
    if ext in SKIP_EXT:
        return "skip", f"binary/model file ({ext})"
    if size > MAX_BYTES:
        return "skip", f"larger than 20 MB ({size / 1e6:.1f} MB)"
    if ext in TEXT:
        try:
            if SECRET.search(p.read_text(errors="ignore")):
                return "skip", "possible API key or secret inside - review and redact before publishing"
        except OSError:
            return "skip", "unreadable"
        return "keep", "table/log/script"
    if ext == ".pdb":
        if PER_TRIAL.search(name) and not re.search(r"final|best|selected", name):
            return "skip", "per-trial structure"
        return "keep", "selected/final structure"
    if ext in FIG:
        if PER_TRIAL.search(rel) or re.search(r"/step_\d+", rel):
            return "skip", "per-step/per-trial image"
        return "keep", "summary figure"
    return "skip", f"unrecognised type ({ext or 'none'})"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("--apply", action="store_true"); a = ap.parse_args()
    root = Path(a.src).expanduser().resolve()
    files = sorted(f for f in root.rglob("*") if f.is_file())
    rows, kept = [], 0
    for f in files:
        verdict, why = classify(f, root)
        rel = f.relative_to(root)
        target = DEST / rel
        if verdict == "keep" and f.suffix.lower() in {".csv", ".json", ".jsonl", ".tsv"} and f.stat().st_size > 1024 * 1024:
            target = target.with_name(target.name + ".gz")
        if verdict == "keep":
            kept += 1
            if a.apply:
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.suffix == ".gz":
                    with open(f, "rb") as s, gzip.open(target, "wb") as d:
                        shutil.copyfileobj(s, d)
                else:
                    shutil.copy2(f, target)
        h = hashlib.sha256(f.read_bytes()).hexdigest() if verdict == "keep" else ""
        rows.append((verdict, str(rel), f.stat().st_size, why, h))
        print(f"{verdict:4s}  {rel}  ({f.stat().st_size / 1e3:.1f} kB)  {why}")
    print(f"\n{kept} of {len(files)} files selected.")
    if a.apply:
        DEST.mkdir(parents=True, exist_ok=True)
        with open(DEST / "MANIFEST.tsv", "w") as m:
            m.write("verdict\tpath\tbytes\treason\tsha256\n")
            for r in rows:
                m.write("\t".join(map(str, r)) + "\n")
        print(f"Copied to {DEST}. Review MANIFEST.tsv, then: git add paper_results/gpt_6_1_sol && git commit")
    else:
        print("Dry run - nothing copied. Re-run with --apply to copy.")


if __name__ == "__main__":
    main()
