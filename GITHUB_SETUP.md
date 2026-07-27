# GitHub publishing checklist

```bash
cd Desalination-LLM
git init
git add .
git commit -m "Initial reproducible Desalination LLM release"
git branch -M main
git remote add origin git@github.com:REPLACE_WITH_OWNER/Desalination-LLM.git
git push -u origin main
```

Before pushing:

1. Review `LICENSE` and the journal-sharing status of `paper/manuscript.pdf`.
2. Replace `REPLACE_WITH_OWNER` in `README.md` and `CITATION.cff`.
3. Run `pytest` and `desalination-llm summary`.
4. Do not commit `.env`, API keys, or provider authentication files.
5. Store original `.ckpt` files with Git LFS or an external research-data repository.

For Git LFS:

```bash
git lfs install
git lfs track '*.ckpt' '*.pth' '*.pt'
git add .gitattributes
```
