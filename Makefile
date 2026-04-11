# Vehicle Tracker — Development Makefile
# Run `make help` to see all available targets.

SHELL := /bin/bash
export DS_UID := $(shell id -u)
export DS_GID := $(shell id -g)

BACKEND_TYPE ?= custom
COMPOSE := docker compose -f docker-compose.dev.yml -f docker-compose.$(BACKEND_TYPE).yml
BACKEND := $(COMPOSE) exec backend
FRONTEND := $(COMPOSE) exec frontend

TRAIN_COMPOSE := docker compose -f docker-compose.training.yml
TRAIN := $(TRAIN_COMPOSE) exec training

LS_COMPOSE := docker compose -f docker-compose.label-studio.yml

# Clip extraction defaults
START    ?= 0
DURATION ?= 60
FPS      ?= 30

.PHONY: help setup dev down clean rebuild \
        start stop-server restart logs \
        shell shell-frontend test test-v lint format \
        status clip tag version \
        training-build training-up training-down training-shell \
        train-extract train-dedup train-label train-split train-student train-student-continue train-export-onnx \
        train-compare-sample train-compare-run train-compare-run-all \
        train-compare-stats train-compare \
        train-ls-up train-ls-down train-ls-compare-setup train-ls-export

## —— Setup & Lifecycle ——————————————————————————

help: ## Show this help message
	@echo "Usage: make <target> [BACKEND_TYPE=custom|deepstream]"
	@echo "  Default backend: $(BACKEND_TYPE)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## First-time setup: create .env, cache dirs, build images
	@test -f .env || (cp .env.example .env && echo "Created .env from .env.example")
	@mkdir -p .ds-cache/.local .ds-cache/.cache .custom-cache/.local .custom-cache/.cache .node-cache/.npm snapshots
	$(COMPOSE) build
	@echo ""
	@echo "Setup complete. Run 'make dev' to start containers."

dev: ## Start dev containers (backend idle, frontend hot-reload)
	$(COMPOSE) up -d
	@echo ""
	@echo "Containers started (backend=$(BACKEND_TYPE))."
	@echo "  Backend:  run 'make start' to launch the server"
	@echo "  Frontend: http://localhost:5173"

down: ## Stop all containers
	$(COMPOSE) down

clean: ## Stop containers and remove caches, snapshots, build artifacts
	$(COMPOSE) down -v --remove-orphans 2>/dev/null || true
	$(TRAIN_COMPOSE) down -v --remove-orphans 2>/dev/null || true
	rm -rf .ds-cache .custom-cache .node-cache .training-cache snapshots frontend/.vite frontend/node_modules/.vite
	@echo "Cleaned up. Run 'make setup' to start fresh."

rebuild: ## Force rebuild Docker images (use after Dockerfile changes)
	$(COMPOSE) build --no-cache

## —— Backend Server ——————————————————————————————

start: ## Start backend server (uvicorn) inside container
	@$(MAKE) -s stop-server 2>/dev/null || true
	@sleep 1
	@$(COMPOSE) exec -d backend bash -c 'cd /app && exec python3 -m backend.main > /tmp/backend.log 2>&1'
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
	@echo "=== Containers (backend=$(BACKEND_TYPE)) ==="
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

## —— Training ————————————————————————————————————

training-build: ## Build training image
	$(TRAIN_COMPOSE) build

training-up: ## Start training container (idle)
	@mkdir -p .training-cache/.config/Ultralytics
	$(TRAIN_COMPOSE) up -d

training-down: ## Stop training container
	$(TRAIN_COMPOSE) down

training-shell: ## Shell into training container
	$(TRAIN) bash

train-extract: ## Extract frames from site videos (1 FPS)
	$(TRAIN) python3 scripts/extract_frames.py

train-dedup: ## Dedupe extracted frames (CNN perceptual hash)
	$(TRAIN) python3 scripts/dedup_frames.py

train-label: ## Auto-label deduped frames with the chosen teacher (TEACHER=yolov8x default)
	$(TRAIN) python3 scripts/auto_label.py --teacher $(or $(TEACHER),yolov8x)

train-split: ## Build stratified train/val/test split + dataset.yaml
	$(TRAIN) python3 scripts/split_dataset.py

train-student: ## Fine-tune YOLOv8s on the RF-DETR auto-labels (fresh run)
	$(TRAIN) python3 scripts/train_student.py --mode fresh $(ARGS)

train-student-continue: ## Continue an existing run. Usage: make train-student-continue FROM=yolov8s_rfdetr_v1
	@test -n "$(FROM)" || (echo "Usage: make train-student-continue FROM=<run_name> [ARGS=...]" && exit 1)
	$(TRAIN) python3 scripts/train_student.py --mode continue --from $(FROM) $(ARGS)

train-export-onnx: ## Export a trained run's best.pt to ONNX for backend consumption. Usage: make train-export-onnx RUN=yolov8s_rfdetr_v1_cont
	@test -n "$(RUN)" || (echo "Usage: make train-export-onnx RUN=<run_name>" && exit 1)
	$(TRAIN) python3 scripts/export_onnx.py --run $(RUN)

## —— P1.5 v2 Teacher comparison ————————————————————

train-compare-sample: ## Sample 1000 frames into data/comparison/
	$(TRAIN) python3 scripts/build_comparison_set.py

train-compare-run: ## Run one teacher on the comparison set. Usage: make train-compare-run TEACHER=yolo26x
	@test -n "$(TEACHER)" || (echo "Usage: make train-compare-run TEACHER=<name>" && exit 1)
	$(TRAIN) python3 scripts/run_comparison.py --teacher $(TEACHER)

train-compare-run-all: ## Run all 5 teachers on the comparison set
	$(TRAIN) python3 scripts/run_comparison.py --teacher yolov8x
	$(TRAIN) python3 scripts/run_comparison.py --teacher yolo26x
	$(TRAIN) python3 scripts/run_comparison.py --teacher yolo12x
	$(TRAIN) python3 scripts/run_comparison.py --teacher rfdetr_m
	$(TRAIN) python3 scripts/run_comparison.py --teacher grounding_dino_t

train-compare-stats: ## Aggregate per-teacher stats into data/comparison/stats.json
	$(TRAIN) python3 scripts/compute_comparison_stats.py

train-compare: ## Orchestrate: sample + run all + stats + LS upload (end-to-end)
	$(MAKE) train-compare-sample
	$(MAKE) train-compare-run-all
	$(MAKE) train-compare-stats
	$(MAKE) train-ls-compare-setup

## —— P1.5 Label Studio annotation ————————————————

train-ls-up: ## Start Label Studio at http://localhost:8080 (idempotent)
	$(LS_COMPOSE) up -d
	@echo "Waiting for Label Studio to come up..."
	@for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do \
		curl -sf http://localhost:8080/health > /dev/null && break || sleep 2; \
	done
	@echo "Enabling legacy API token auth + setting deterministic token..."
	@docker exec vehicle_tracker-label-studio bash -c '\
	  cd /label-studio && python label_studio/manage.py shell -c "\
	from organizations.models import Organization; \
	from rest_framework.authtoken.models import Token; \
	from users.models import User; \
	o = Organization.objects.first(); \
	o.jwt.legacy_api_tokens_enabled = True; o.jwt.save(); \
	u = User.objects.get(email=\"joel@vt.local\"); \
	Token.objects.filter(user=u).delete(); \
	Token.objects.create(user=u, key=\"vt_p1_5_token_2026\"); \
	print(\"LS_INIT_OK\")"' 2>&1 | tail -3
	@echo ""
	@echo "Label Studio ready at http://localhost:8080"
	@echo "  Login: joel@vt.local / vtlabels2026"
	@echo "  Then run: make train-ls-compare-setup"

train-ls-down: ## Stop Label Studio
	$(LS_COMPOSE) down

train-ls-compare-setup: ## Create LS comparison project + post all 5 teachers' predictions
	$(TRAIN) python3 scripts/ls_setup.py --mode comparison

train-ls-export: ## Export human-corrected labels from LS into data/benchmark/labels_gt
	$(TRAIN) python3 scripts/ls_export.py

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
