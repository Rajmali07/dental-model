# One-Time Setup — GitHub + Hugging Face

Do these once before CI/CD in `.github/workflows/` will work end-to-end.

## 1. GitHub repo

1. Create a new repo (e.g. `dental-model`) on GitHub — don't initialize with a README, since
   this scaffold already has one.
2. Push this scaffold:
   ```bash
   cd dental-model
   git init
   git add .
   git commit -m "Initial scaffold: uv, CI/CD, configs"
   git branch -M main
   git remote add origin https://github.com/<your-username>/dental-model.git
   git push -u origin main
   ```
3. In the GitHub repo, go to **Settings → Branches** and add a branch protection rule on
   `main` requiring the `lint-and-test` CI check to pass before merging (optional but
   recommended once you have collaborators or just want a safety net).

## 2. Hugging Face account

1. Create an account at [huggingface.co/join](https://huggingface.co/join) if you don't have one.
2. Go to **Settings → Access Tokens** → create a new token with **write** permission.
   Save it somewhere safe — you'll need it in three places below.

## 3. Hugging Face model repo (for weights)

1. Create a new model repo: [huggingface.co/new](https://huggingface.co/new) → type "Model".
   Name it e.g. `<your-username>/dental-model`.
2. Locally, authenticate once:
   ```bash
   uv run huggingface-cli login
   # paste the write token when prompted
   ```
3. After training, push weights with `huggingface_hub.upload_folder()` (called from
   `src/dental_model/*/train.py` or a small `scripts/push_model.py` — not committed to git,
   uploaded straight to the Hub).

## 4. Hugging Face Space (for the deployed app)

1. Create a new Space: [huggingface.co/new-space](https://huggingface.co/new-space).
   - SDK: **Gradio**
   - Hardware: start with the free CPU tier; upgrade to a small GPU tier later only if
     detector inference latency is a problem in practice — record whichever you land on
     in the README.
2. Note the Space's repo id, e.g. `<your-username>/dental-model-demo`.
3. In the Space's own **Settings → Repository secrets**, add:
   - `HF_TOKEN` — not usually needed inside the Space itself unless `app.py` pulls a
     private model repo; if the model repo is public, this can be skipped in the Space.
4. In your **GitHub repo's** Settings → Secrets and variables → Actions, add:
   - `HF_TOKEN` — the same write token from step 2, so CI can push to the Space.
   - `HF_SPACE_REPO` — the Space repo id, e.g. `<your-username>/dental-model-demo`.
5. Once these two GitHub secrets exist, `.github/workflows/deploy-space.yml` will push
   `app/` to the Space automatically on every merge to `main`.

## 5. Local dev

```bash
cp .env.example .env
# fill in HF_TOKEN locally for ad-hoc pushes/testing; this file is never read by the Space itself
```

## Checklist
- [ ] GitHub repo created and scaffold pushed
- [ ] HF account created, write token generated
- [ ] HF model repo created
- [ ] HF Space created (Gradio, hardware tier chosen)
- [ ] `HF_TOKEN` + `HF_SPACE_REPO` added to GitHub Actions secrets
- [ ] `.env` created locally from `.env.example`
