# Submission runbook — lock the guaranteed entry

Goal: get the verified image into a **public** registry and prove the judges can
pull + run it with **no dependency on your local Docker cache**. `PULL_ERROR` is
the FAQ's #1 killer — this sequence rules it out.

Run everything in your bash shell, from the repo root. Replace `<USER>` with your
Docker Hub username (or use the GHCR variant at the bottom).

---

## 0. Build the FINAL image (already done here, but rebuild if you changed code)

```bash
docker build --platform linux/amd64 -t dynamic-model-router:submit .
```
`--platform linux/amd64` is mandatory — the judging VM is amd64.

## 1. Log in and push to a PUBLIC repo (Docker Hub)

```bash
docker login                                   # your Docker Hub account
docker tag dynamic-model-router:submit <USER>/dynamic-model-router:v1
docker push <USER>/dynamic-model-router:v1
```
Use an immutable version tag (`v1`), not `:latest` — the FAQ wants an exact,
stable tag. **Copy the `sha256:...` digest** the push prints; that's the most
unambiguous reference to put on the submission form.

Then make it public: Docker Hub → your repo → **Settings → Make public**.

## 2. Prove it's PUBLIC and cache-independent (the real test)

```bash
docker logout                                  # judges pull anonymously
docker rmi <USER>/dynamic-model-router:v1 dynamic-model-router:submit
docker image prune -af                         # nuke local cache

docker pull <USER>/dynamic-model-router:v1     # MUST succeed while logged OUT
```
If this pull fails, the image isn't public yet — fix visibility before submitting.

## 3. Run the freshly-pulled image and validate output

```bash
mkdir -p out
MSYS_NO_PATHCONV=1 docker run --rm \
  -v "$(pwd -W)/fixtures:/input:ro" \
  -v "$(pwd -W)/out:/output" \
  -e FIREWORKS_API_KEY="$FIREWORKS_API_KEY" \
  -e FIREWORKS_BASE_URL="$FIREWORKS_BASE_URL" \
  -e ALLOWED_MODELS="accounts/fireworks/models/minimax-m3,accounts/fireworks/models/kimi-k2p7-code,accounts/fireworks/models/gemma-4-31b-it,accounts/fireworks/models/gemma-4-26b-a4b-it,accounts/fireworks/models/gemma-4-31b-it-nvfp4"

python -c "import json; d=json.load(open('out/results.json')); assert all('task_id' in r and isinstance(r['answer'], str) for r in d); print('OK:', len(d), 'rows, all valid')"
```
With the key set you'll see real answers + the gemmas self-heal at startup; the
container should finish well under the limits. (Without a key it still writes
valid JSON — the contract holds either way.)

## 4. Before you hit submit

- [ ] Image **pulls while logged out** (step 2 passed).
- [ ] `results.json` valid, one row per task, answers are strings (step 3).
- [ ] **GitHub repo is public** (submission also takes a repo URL).
- [ ] Record the exact image ref: `<USER>/dynamic-model-router:v1@sha256:...`.
- [ ] Form fields: **Categories** = Agent Builder track + Assistant;
      **Technologies** = AMD Developer Cloud, Fireworks AI, Python, Docker.

---

### GHCR variant (if you prefer GitHub Container Registry)
```bash
echo "$GITHUB_TOKEN" | docker login ghcr.io -u <GH_USER> --password-stdin
docker tag dynamic-model-router:submit ghcr.io/<GH_USER>/dynamic-model-router:v1
docker push ghcr.io/<GH_USER>/dynamic-model-router:v1
# then: GitHub -> Packages -> this package -> Package settings -> Change visibility -> Public
```
The `GITHUB_TOKEN` needs `write:packages` scope. Verify with the same logged-out
pull (`docker logout ghcr.io` first).
