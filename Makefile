# Vehicle Tracker — Development Makefile
# Run `make help` to see all available targets.

SHELL := /bin/bash
export DS_UID := $(shell id -u)
export DS_GID := $(shell id -g)

COMPOSE := docker compose -f docker-compose.dev.yml
BACKEND := $(COMPOSE) exec backend
FRONTEND := $(COMPOSE) exec frontend

# Clip extraction defaults
START    ?= 0
DURATION ?= 60
FPS      ?= 30

.PHONY: help setup dev down clean rebuild \
        start stop-server restart logs \
        shell shell-frontend test test-v lint format \
        status clip tag version

## —— Setup & Lifecycle ——————————————————————————

help: ## Show this help message
	@echo "Usage: make <target>"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## First-time setup: create .env, cache dirs, build images
	@test -f .env || (cp .env.example .env && echo "Created .env from .env.example")
	@mkdir -p .ds-cache/.local .ds-cache/.cache .node-cache/.npm snapshots
	$(COMPOSE) build
	@echo ""
	@echo "Setup complete. Run 'make dev' to start containers."

dev: ## Start dev containers (backend idle, frontend hot-reload)
	$(COMPOSE) up -d
	@echo ""
	@echo "Containers started."
	@echo "  Backend:  run 'make start' to launch the server"
	@echo "  Frontend: http://localhost:5173"

down: ## Stop all containers
	$(COMPOSE) down

clean: ## Stop containers and remove caches, snapshots, build artifacts
	$(COMPOSE) down -v --remove-orphans 2>/dev/null || true
	rm -rf .ds-cache .node-cache snapshots frontend/.vite frontend/node_modules/.vite
	@echo "Cleaned up. Run 'make setup' to start fresh."

rebuild: ## Force rebuild Docker images (use after Dockerfile changes)
	$(COMPOSE) build --no-cache

## —— Backend Server ——————————————————————————————

start: ## Start backend server (uvicorn) inside container
	@$(MAKE) -s stop-server 2>/dev/null || true
	@sleep 1
	@$(BACKEND) bash -c 'cd /app && python3 -m backend.main > /tmp/backend.log 2>&1 &'
	@echo "Waiting for backend..."
	@for i in 1 2 3 4 5; do \
		$(BACKEND) bash -c 'curl -sf http://localhost:8000/channels > /dev/null 2>&1' && break || sleep 1; \
	done
	@echo "Backend running at http://localhost:8000"

stop-server: ## Stop backend server (container keeps running)
	@$(BACKEND) bash -c 'kill $$(ps aux | grep "[b]ackend.main" | awk "{print \$$2}") 2>/dev/null \
		&& echo "Backend server stopped" || echo "No server running"'

restart: ## Restart backend server
	@$(MAKE) -s stop-server
	@sleep 1
	@$(MAKE) -s start

logs: ## Tail backend server logs
	$(BACKEND) bash -c "tail -f /tmp/backend.log 2>/dev/null || echo 'No logs yet — run make start first'"

## —— Development —————————————————————————————————

shell: ## Open bash shell in backend container
	$(BACKEND) bash

shell-frontend: ## Open sh shell in frontend container
	$(FRONTEND) sh

test: ## Run pytest
	$(BACKEND) pytest backend/tests/ --tb=short -q

test-v: ## Run pytest verbose (show individual test names)
	$(BACKEND) pytest backend/tests/ --tb=short -v

lint: ## Run ruff check (report issues)
	$(BACKEND) ruff check --no-cache backend/

format: ## Run ruff format (auto-fix)
	$(BACKEND) ruff format --no-cache backend/

## —— Data & Debugging ————————————————————————————

status: ## Show container status, server health, GPU memory
	@echo "=== Containers ==="
	@$(COMPOSE) ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || echo "No containers running"
	@echo ""
	@echo "=== Backend Server ==="
	@$(BACKEND) bash -c "curl -sf http://localhost:8000/channels 2>/dev/null && echo ' (healthy)' || echo 'Not running'" 2>/dev/null || echo "Container not running"
	@echo ""
	@echo "=== GPU ==="
	@nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"

clip: ## Extract a test clip. Usage: make clip SRC="path/to/video.mp4" START=30 DURATION=60 OUT="clip.mp4"
	@test -n "$(SRC)" || (echo "Usage: make clip SRC=\"path/to/video.mp4\" START=30 DURATION=60 OUT=\"clip.mp4\"" && exit 1)
	@test -n "$(OUT)" || (echo "Error: OUT is required" && exit 1)
	$(BACKEND) ffmpeg -ss $(START) -t $(DURATION) -i "$(SRC)" -c copy "$(OUT)" -y

## —— Release —————————————————————————————————————

version: ## Show current version from git tags
	@git describe --tags --always 2>/dev/null || echo "No tags found"

tag: ## Show current version and prompt for new tag
	@echo "Current version: $$(git describe --tags --always 2>/dev/null || echo 'none')"
	@echo "Recent tags:"
	@git tag -l --sort=-v:refname | head -5
	@echo ""
	@echo "To create a new release:"
	@echo "  git tag vX.Y.Z"
	@echo "  git push origin vX.Y.Z"
