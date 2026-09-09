---
name: colab-operator
description: Operate Google Colab via the colab CLI — provision GPU sessions, run scripts, sync files. Use when work needs Colab compute, checking GPU entitlement, or fetching remote artifacts.
---

# Skill: Colab Session Operator

Operate Google Colab environments via the `colab` CLI: provision GPU/TPU sessions, run Python/shell on the VM, sync files, and capture work as notebooks.

Upstream source: `googlecolab/google-colab-cli` (`colab skill` output, v0.6.0).
Do not edit the reference sections below — project wiring lives at the bottom.

## When to activate
- Creating or managing TPU/GPU sessions.
- Running Python or shell on a remote Colab VM.
- Syncing files between local and remote.
- Automating environment setup (packages, auth, Drive).
- Exporting session history as a Jupyter notebook.

## Mental model (read this first)
- **A session == a live Jupyter kernel on a rented VM.** `colab new` allocates a billable VM; `colab stop` releases it. Nothing reclaims it automatically except a 24h keep-alive cap, so an unstopped session burns compute units indefinitely.
- **Kernel state PERSISTS across `colab exec` / `colab repl` calls in the same session.** Each invocation reattaches to the *same* kernel (the kernel ID is cached in local state) and only closes the websocket on exit — it does **not** shut the kernel down. So imports, variables, and defined functions survive between separate `colab exec` commands. Build up state incrementally; don't re-import everything each call. (`colab stop` and `colab restart-kernel` are what actually reset it.)
- **Default working directory is `/content`.** Every `exec`/`repl`/`run` `cd`s there first; prefer absolute paths (`/content/...`) for file work. For `colab ls/rm/upload/download`, absolute `/content/...` paths work and the default `ls` path is `content` (VM root).
- **`colab` is fire-and-forget.** Each command authenticates, does one thing, and exits. A detached background daemon (spawned by `colab new`) handles keep-alive; you don't manage it.

## Authentication (the #1 thing that blocks agents)
- The global flag is `--auth={adc,oauth2}` and the **default is `adc`** (Application Default Credentials). It must come *before* the subcommand: `colab --auth=adc new -s x`.
- **oauth2 setup**: `colab --auth=oauth2 <anything>` triggers a browser consent flow on first use (token cached at `~/.config/colab-cli/token.json`). Requires a client config at `~/.colab-cli-oauth-config.json` (or `-c PATH`). The browser step means it usually needs a human; prefer ADC for agents.
- **Verify auth in one shot**: `colab sessions` (read-only, lists server assignments).
- **Do NOT confuse `colab auth` with CLI authentication.** `colab auth` injects *VM-side* GCP credentials into the running kernel (so notebook code can call BigQuery/GCS); it is orthogonal to how the CLI itself authenticates.

## Workflow

### Provision
- `colab new -s <name>` (CPU). Add `--gpu T4` for accelerators. **Always pass `-s <name>`** — an omitted name is auto-generated as a random 6-hex string, which makes later commands ambiguous.
- Supported `--gpu`: `T4`, `L4`, `G4`, `H100`, `A100`. Supported `--tpu`: `v5e1`, `v6e1`.
- **Gotcha**: an unrecognized `--gpu` value silently falls back to **A100** (which then usually fails the next step). A `400` on `colab new` with an accelerator means no quota/entitlement for it on this account — fall back to `--gpu T4` or omit the flag for CPU.
- Accelerator availability is tier-gated; most accounts can only get CPU. Don't assume a GPU/TPU will allocate.

### Execute
- **Preferred**: `colab exec -s <name> -f <script.py>` runs a local script on the remote VM (read locally, sent to the kernel — no manual upload needed).
- **Piped code**: `echo "print(1)" | colab exec -s <name>` or `cat script.py | colab exec -s <name>`.
- **Notebooks**: `colab exec -s <name> -f nb.ipynb` runs each code cell and writes results to `<basename>_output.ipynb` next to the input.
- **Plots/images**: use `--output-image <path>` on `exec`/`repl` to save to a known location.
- **Shell**: `echo "cmd" | colab console -s <name>` for batch shell. `exec` is faster when you don't need a real shell.
- **Never run `colab repl`, `colab console`, `colab auth`, or `colab drivemount` interactively from an agent** — they expect a TTY and will hang. `repl`/`console` accept piped stdin and exit on EOF; `auth`/`drivemount` genuinely require a human at the terminal.

### Ephemeral one-shot jobs (`colab run`)
- `colab run [--gpu T4] [--tpu v6e1] [--keep] [-s NAME] script.py [args...]` = `new` + `exec` + `stop` in one command. It provisions a fresh VM, runs the script with `sys.argv` and `__name__ == "__main__"` set like native `python script.py args`, then tears the VM down (unless `--keep`).
- **Exit codes propagate**: an uncaught exception or `sys.exit(N)` in the script makes `colab run` exit non-zero.
- **Stream separation**: `colab run` writes its own `[colab] ...` chatter to **stderr** and the script's output to **stdout**.
- A nonexistent script path exits non-zero **before** allocating a VM (no wasted compute).

### Automate
- `colab install -s <name> pkg1 pkg2` — installs via `uv pip install --system`, falling back to `pip`. Also `colab install -s <name> -r requirements.txt`.

### Inspect & report
- `colab sessions` lists server-side assignments and auto-prunes stale local entries.
- `colab status [-s <name>]` shows hardware, IDLE/BUSY, and last execution.
- `colab log -s <name> [-n 20]` shows recent structured events; invaluable when a task fails.
- `colab log -s <name> -o summary.ipynb` exports the session as a notebook (also `.md`, `.txt`, `.jsonl` by suffix).
- `colab url -s <name>` attaches the Colab web UI to an existing CLI session instead of allocating a new VM.

## Safety
- **Always `colab stop -s <name>` when done** — idle VMs burn compute units. `colab run` (without `--keep`) self-cleans even if the script errors.
- Local state lives in `~/.config/colab-cli/sessions.json`. Don't edit by hand.
- **Isolate parallel/agent runs** with the global `--config <path>` flag to point session state at a scratch file.

## Recovery
- "Session not found" / 404 / 401 on exec: the backend pruned the VM. Re-create with `colab new`.
- Execution timeout or wedged kernel: `colab restart-kernel -s <name>` (keeps the VM, resets the kernel), or `colab stop` then `colab new`.

## Project wiring (finetune — agent must read this)
- **Binary**: `/home/salla/miniconda3/bin/colab` (miniconda3). It is NOT on a non-interactive shell's PATH — always invoke via this absolute path through `wsl`, e.g. `wsl /home/salla/miniconda3/bin/colab sessions`. Never bare `colab`.
- **Repo mount**: `F:\finetune` == `/mnt/f/finetune` inside WSL. Scripts run from there are read locally and sent to the kernel — no manual upload needed.
- **Session naming**: always `-s <name>`, prefixed `pb-` (e.g. `-s pb-train`). Never rely on auto-generated names.
- **GPU policy**: T4 only on free tier. A `400` on `--gpu T4` means no entitlement — stop and report to the user (Kaggle fallback), don't retry other accelerators blindly.
- **Checkpoint discipline**: every training run pushes adapters to the Hub mid-run; a preempted VM must never cost more than its tail.
- **Secrets**: HF token / W&B key are passed at runtime from user-supplied env, never written into the repo.
- **Dependency pin (2026-09-09)**: `google-colab-cli==0.6.0` requires the OLD
  `jupyter-kernel-client==0.15.0` API (`KernelClient`, `wsclient`,
  `client.output_hook`). pip resolves 1.0.2 by default, which breaks every
  `exec`/`run` with `AttributeError: ... has no attribute 'KernelClient'`.
  Pinned in WSL miniconda3. If a fresh env breaks the same way, downgrade first.
