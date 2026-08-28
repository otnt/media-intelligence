# Agent notes (media-download)

Operational context for coding agents working in this repository.

## Default ports and services

- media-pipeline API default port: **8875** (not 8765; 8765 is often used by `personal_telemetry` on this machine).
- Dashboard: `http://127.0.0.1:8875/`
- Config: `~/.config/media-pipeline/config.yaml`

## LaunchAgents

| Label | Install script | What it runs |
|-------|----------------|--------------|
| `com.media-pipeline.server` | `./scripts/install-launch-agent.sh` | `media-pipeline serve --lan --port 8875` |
| `com.tailscale.serve` | `./scripts/tailscale/install.sh` | `python3 ~/.config/tailscale/apply-serve.py` |

Logs: `~/AIContent/logs/launchd.out.log`, `launchd.err.log`, `tailscale-serve.out.log`, `tailscale-serve.err.log`.

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

Tailscale Serve is tailnet-only.
Never enable Funnel from this project.

## Git worktrees

Create worktrees inside the repo only:

```bash
git worktree add .worktrees/<short-name> <branch>
```

Never create sibling worktrees such as `../media-download-<name>`.
