# Agent notes (media-download)

Operational context for coding agents working in this repository.

## Default ports and services

- media-pipeline API default port: **8875** (not 8765; 8765 is often used by `personal_telemetry` on this machine).
- Dashboard: `http://127.0.0.1:8875/`
- Config: `~/.config/media-pipeline/config.yaml`

## Keep the API always running

Install (or reinstall) the LaunchAgent:

```bash
./scripts/install-launch-agent.sh
```

| Detail | Value |
|--------|--------|
| Label | `com.media-pipeline.server` |
| Entrypoint | `.venv` python `-m media_pipeline serve --host 127.0.0.1 --port 8875` |
| Bind | **`127.0.0.1:8875` only** |
| KeepAlive | yes |
| ThrottleInterval | 30s (avoids crash loops after bind failures) |

Do **not** use `--lan` / `0.0.0.0` in the LaunchAgent while Tailscale Serve also publishes `:8875`.
Tailscale already listens on the Tailscale IP for that port; binding `0.0.0.0:8875` causes `address already in use`, KeepAlive thrashing, then launchd gives up.

Remote access (phone, other machines): Tailscale URL `http://<machine>.ts.net:8875/`.
Local access: `http://127.0.0.1:8875/`.

## LaunchAgents

| Label | Install script | What it runs |
|-------|----------------|--------------|
| `com.media-pipeline.server` | `./scripts/install-launch-agent.sh` | `media-pipeline serve --host 127.0.0.1 --port 8875` |
| `com.tailscale.serve` | `./scripts/tailscale/install.sh` | `python3 ~/.config/tailscale/apply-serve.py` (login + every 5 min) |
| `com.media-pipeline.bulk.<job>` | `./scripts/install-bulk-agent.sh <job>` (`--remove` to stop) | `media-pipeline bulk run <job>` (downloads + ASR); restarts after a crash, stays stopped once the job is finished |
| `com.media-pipeline.watchdog` | `./scripts/install-watchdog.sh` (`--remove` to stop) | `media-pipeline watchdog` every 5 min: API health, bulk heartbeats, disk; restarts what is stuck after two bad checks |

Logs: `~/AIContent/logs/launchd.out.log`, `launchd.err.log`, `tailscale-serve.out.log`, `tailscale-serve.err.log`, `bulk-<job>.log`, `watchdog.log` (latest result in `watchdog.json`).

Bulk jobs live in `~/AIContent/bulk/<job>/` (`progress.sqlite3` + `media/` staging). The staging folder is outside
`paths.videos` on purpose, so the storage sweep never archives half-finished bulk downloads to iCloud.

Do **not** pass `--tailscale` to `media-pipeline serve` in LaunchAgent.
Tailscale Serve is machine-wide, not project-specific.

## Tailscale Serve (machine-wide)

Runtime files live under `~/.config/tailscale/`, not in this repo.

| Path | Role |
|------|------|
| `~/.config/tailscale/serve.json` | Declarative routes for all local backends |
| `~/.config/tailscale/apply-serve.py` | Applies routes via `tailscale serve` CLI |
| `scripts/tailscale/serve.example.json` | Example route for media-pipeline on :8875 |
| `scripts/tailscale/install.sh` | Deploys the above and loads `com.tailscale.serve` |

Install or refresh:

```bash
./scripts/tailscale/install.sh
python3 ~/.config/tailscale/apply-serve.py
```

`install.sh` copies `apply-serve.py` and creates `serve.json` from the example only if missing.
It removes the legacy `com.media-pipeline.tailscale-serve` LaunchAgent if present.

To add another backend, edit `~/.config/tailscale/serve.json` (see `serve.example.json` for shape), then re-run `apply-serve.py`.
Use host keys like `:8875`; MagicDNS is filled in at apply time.
`//` and `/* */` comments in `serve.json` are allowed.

Tailscale Serve is tailnet-only.
Never enable Funnel from this project.

## Git worktrees

Create worktrees inside the repo only:

```bash
git worktree add .worktrees/<short-name> <branch>
```

Never create sibling worktrees such as `../media-download-<name>`.

## Web UI

- `/` is the reader (`dashboard/static/index.html`): mobile-first library + note view, one row per video, page-level scrolling, PWA manifest and icons under `/static/`.
  It reads `/v1/tasks/{id}/note` (task, summary, speaker transcript, frames) and polls only while a task or summary is running, never repainting under a text selection.
- Tasks with `extra.collection` (made by bulk jobs) are left out of `GET /v1/tasks` unless `?collection=<name>` is given; the reader shows each collection as a 📚 chip (`/v1/collections`).
- `/bulk` is the bulk download progress table (`bulk.html`), polling `/v1/bulk/{job}` every 10s and skipping repaints under a text selection.
- `/classic` is the previous dashboard (`classic.html`): settings, concurrency, keyframe keep/drop, prompt editing, stage reruns.

## Dashboard polling (text selection)

The classic detail page polls every 4s (`loadTasks` / `loadWorker` / `refreshSelected`).

`refreshSelected` must **not** rewrite Extracted / Summary / gallery via `innerHTML` when:

1. The task `status` is `completed` **and** `summary.status` is not `running`, or
2. The user currently has a text selection inside `#main`.

Rewriting those nodes clears the browser selection and breaks selecting long passages.

Still poll the sidebar and worker chips.
Still refresh detail body while the pipeline is in progress, or while summary regeneration is `running`.
Actions like Regenerate summary / Rerun / Keep-Drop call `openTask` or `paintSummary` directly and do not rely on the idle completed poll path.
