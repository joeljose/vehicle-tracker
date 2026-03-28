# Vehicle Tracker — Development Rules

## Environment
- **ALL development happens inside Docker containers.** Never install dependencies on the host.
- Backend dev container: `nvcr.io/nvidia/deepstream:7.0-triton-multiarch`
- Frontend dev container: `node:22-alpine`
- Start dev environment: `docker compose -f docker-compose.dev.yml up`
- Run tests: `docker compose exec backend pytest`
- Run linting: `docker compose exec backend ruff check .`

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
