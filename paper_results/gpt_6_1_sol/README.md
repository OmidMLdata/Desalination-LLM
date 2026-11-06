# GPT-6.1 Sol run (`results_gpt61_t30_g3_20261015`)

`gpt61_per_step_from_pdf.csv` holds the per-step water flux and ion rejection of the six conditions:
- GPT-6.1 Sol + tools;
- no tools, predictor score;
- no tools, own estimate;
- scaffold greedy, scaffold greedy + rules and scaffold random, each as the mean ± SD of 6 seeds.

These values were recovered from the vector paths of `six_category_comparison.pdf` with `claude_opus/extract_gpt_from_pdf.py`, so they are exact to plotting precision.

## Adding the raw run folder
Run this on the machine that holds the run:

```bash
python scripts/import_gpt61_results.py /home/akshay/Documents/MD/GPT/results_gpt61_t30_g3_20261015          # dry run: lists keep/skip
python scripts/import_gpt61_results.py /home/akshay/Documents/MD/GPT/results_gpt61_t30_g3_20261015 --apply  # copies into raw/
git add paper_results/gpt_6_1_sol && git commit -m "Add GPT-6.1 Sol raw results" && git push
```

**What the script keeps:** tables, logs, configs, the scripts that produced the run, final or selected structures, and summary figures.

**What it skips:**
- per-trial images and PDB files;
- pickles, arrays and model checkpoints;
- files larger than 20 MB;
- any text file that appears to contain an API key.

Every decision is recorded in `raw/MANIFEST.tsv`.

If the raw folder includes a per-step table, save it as `gpt61_per_step.csv`, with the columns of the `_from_pdf` file. The figure scripts then use it in place of the PDF-recovered values.
