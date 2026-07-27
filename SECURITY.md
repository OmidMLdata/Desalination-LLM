# Security and credentials

Never commit API keys, `.env` files, Gemini/OpenAI authentication state, SSH keys, or provider tokens. The active code reads no credentials from tracked files. Use environment variables or the provider's local credential store.

Before publishing a release, scan the repository and its Git history. Example:

```bash
git grep -nE 'sk-[A-Za-z0-9]{20,}|AIza[A-Za-z0-9_-]{20,}'
```

Report a discovered credential privately to the repository owners and revoke it immediately.
