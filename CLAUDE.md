# Vehicle Tracker — Development Rules

## Environment
- **ALL development happens inside Docker containers.** Never install dependencies on the host.
- Backend dev container: `nvcr.io/nvidia/deepstream:8.0-gc-triton-devel` (custom Dockerfile bakes in ffmpeg + pip deps)
- Frontend dev container: `node:22-alpine`
- GPU driver on host: 580.126.09 (compatible with DS 8.0, NOT DS 9.0 which needs 590.x)
- All commands go through the **Makefile** — never call `docker compose` directly (files will be root-owned)

## Common Commands
- Start dev environment: `make dev`
- Start backend server: `make start`
- Stop backend server: `make stop-server`
- Run tests: `make test`
- Run linting: `make lint`
- Auto-format: `make format`
- Shell into backend: `make shell`
- Shell into frontend: `make shell-frontend`
- Check status: `make status`
- Frontend dev server: `http://localhost:5173` (Vite, proxies `/api` -> backend:8000)
- Install frontend deps: `make shell-frontend` then `npm install <pkg>`
- See all targets: `make help`

## Versioning
- Version comes from **git tags only** via `setuptools-scm`
- No version field in `pyproject.toml` or `package.json` to bump manually
- To release: `git tag vX.Y.Z` — that's the version source of truth
- Between tags, version auto-generates as `X.Y.Z.devN+gHASH`

## Pipeline Architecture
- Two backends: DeepStream (primary) and Custom (secondary) — never mixed
- Both implement `PipelineBackend` Protocol (see design doc Section 3.3)
- FastAPI layer imports ONLY through the Protocol interface
- Shared modules (`direction.py`, `roi.py`, `alerts.py`, etc.) live in `backend/pipeline/`, not inside `deepstream/` or `custom/`

## Key Conventions
- All time-based configs in seconds, never frame counts
- Inference normalized to 30fps regardless of source frame rate
- Commit messages: never mention Claude or Anthropic
- Use axel with parallel connections for file downloads
- Never delete downloaded/captured data without asking first
