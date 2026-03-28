#!/usr/bin/env bash
# Start the development environment.
# Sets the container user to match the host user so all
# files created on bind mounts are owned by you, not root.

set -euo pipefail

export DS_UID="$(id -u)"
export DS_GID="$(id -g)"

# Ensure cache dirs exist (bind-mounted into container)
mkdir -p .ds-cache/.local .ds-cache/.cache

exec docker compose -f docker-compose.dev.yml "$@"
