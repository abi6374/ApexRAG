# ═══════════════════════════════════════════════════════════════════
# ApexRAG — Makefile
# ═══════════════════════════════════════════════════════════════════
# Common development commands.  Requires Python >= 3.10 and GNU Make.
#
# Quick reference:
#   make install          Install dev dependencies
#   make lint             Run ruff linter
#   make format           Auto-format code with ruff
#   make typecheck        Run mypy type checker
#   make test             Run all tests
#   make check            Run lint + format-check + typecheck + test (CI equivalent)
#   make build            Build sdist + wheel
#   make serve            Start the FastAPI dev server
#   make clean            Remove build artifacts and caches
# ═══════════════════════════════════════════════════════════════════

.DEFAULT_GOAL := help

# ── Detect Python & OS ──────────────────────────────────────────────────
PYTHON   := python3
PIP      := $(PYTHON) -m pip

# Colour output helpers (auto-disable on non-TTY)
# NOTE: This Makefile assumes a Unix-like environment (rm, find, etc.).
#       On Windows, use Git Bash, WSL, or equivalent.
ifeq ($(shell test -t 1 && echo yes),yes)
  GREEN  := \033[32m
  CYAN   := \033[36m
  YELLOW := \033[33m
  RED    := \033[31m
  BOLD   := \033[1m
  RESET  := \033[0m
else
  GREEN  :=
  CYAN   :=
  YELLOW :=
  RED    :=
  BOLD   :=
  RESET  :=
endif

INFO    := @printf "$(CYAN)➜$(RESET) $(BOLD)%s$(RESET)\n"
OK      := @printf "$(GREEN)✔$(RESET) %s\n"
WARN    := @printf "$(YELLOW)⚠$(RESET) %s\n"
ERR     := @printf "$(RED)✖$(RESET) %s\n"

# ══════════════════════════════════════════════════════════════════════════
# Installation
# ══════════════════════════════════════════════════════════════════════════

.PHONY: install install-dev install-all install-docs

install:                          ## Install package in editable mode with dev & web extras
	$(INFO) "Installing ApexRAG [dev,web]…"
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev,web]"
	$(OK) "Installation complete.  Run 'make test' to verify."

install-all:                      ## Install with ALL extras (web, postgres, docling, vectors, telemetry, monitoring, docs)
	$(INFO) "Installing ApexRAG with all extras…"
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[all]"
	$(OK) "Full installation complete."

install-docs:                     ## Install documentation dependencies
	$(PIP) install -e ".[docs]"

install-precommit:                ## Install pre-commit hooks
	$(PIP) install pre-commit
	pre-commit install
	$(OK) "Pre-commit hooks installed."

# ══════════════════════════════════════════════════════════════════════════
# Quality — Lint, Format, Typecheck
# ══════════════════════════════════════════════════════════════════════════

.PHONY: lint format format-check typecheck

lint:                             ## Run ruff linter on source code
	$(INFO) "Running ruff linter…"
	ruff check apex_rag/
	$(OK) "No lint errors."

format:                           ## Auto-format source code with ruff
	$(INFO) "Formatting with ruff…"
	ruff format apex_rag/
	$(OK) "Formatting applied."

format-check:                     ## Check formatting without modifying files (CI)
	$(INFO) "Checking formatting…"
	ruff format --check apex_rag/
	$(OK) "Formatting is clean."

typecheck:                        ## Run mypy type checker on source code
	$(INFO) "Running mypy type checker…"
	mypy apex_rag/ --ignore-missing-imports --python-version=3.10
	$(OK) "No type errors."

typecheck-strict:                 ## Run mypy in strict mode (verbose)
	$(INFO) "Running mypy (strict, all modules)…"
	mypy apex_rag/ --strict --ignore-missing-imports --python-version=3.10
	$(OK) "Strict type check passed."

# ══════════════════════════════════════════════════════════════════════════
# Testing
# ══════════════════════════════════════════════════════════════════════════

.PHONY: test test-cov test-quick test-bench

test:                             ## Run all tests (quick, no coverage)
	$(INFO) "Running tests…"
	$(PYTHON) -m pytest tests/ -v --tb=short
	$(OK) "All tests passed."

test-cov:                         ## Run all tests with coverage report
	$(INFO) "Running tests with coverage…"
	$(PYTHON) -m pytest tests/ -v --tb=short --cov=apex_rag --cov-report=term-missing
	$(OK) "Tests passed.  See coverage report above."

test-quick:                       ## Run tests, stopping on first failure (fast feedback)
	$(INFO) "Running tests (fast-fail mode)…"
	$(PYTHON) -m pytest tests/ -x --tb=short -q
	$(OK) "Quick tests passed."

test-bench:                       ## Run benchmark tests (may be slow)
	$(INFO) "Running benchmark tests…"
	$(PYTHON) -m pytest tests/ -v --tb=short -m benchmark
	$(OK) "Benchmarks complete."

# ══════════════════════════════════════════════════════════════════════════
# Full CI Check
# ══════════════════════════════════════════════════════════════════════════

.PHONY: check

check: lint format-check typecheck test-cov  ## Run lint + format-check + typecheck + tests (CI equivalent)
	$(OK) "All checks passed."

# ══════════════════════════════════════════════════════════════════════════
# Build & Publish
# ══════════════════════════════════════════════════════════════════════════

.PHONY: build clean publish

build: clean                      ## Build sdist and wheel packages
	$(INFO) "Building sdist and wheel…"
	$(PYTHON) -m build
	$(OK) "Build complete: ./dist/"

clean:                            ## Remove build artifacts, caches, and generated files
	$(INFO) "Cleaning build artifacts…"
	rm -rf dist/ build/ *.egg-info/ .eggs/
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -rf __pycache__/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	$(OK) "Clean."

publish: build                    ## Build and upload to PyPI (requires credentials)
	$(INFO) "Uploading to PyPI…"
	$(PYTHON) -m twine upload dist/*
	$(OK) "Published to PyPI."

# ══════════════════════════════════════════════════════════════════════════
# Serve — Local API Server
# ══════════════════════════════════════════════════════════════════════════

.PHONY: serve serve-reload

serve:                            ## Start the FastAPI development server (port 8000)
	$(INFO) "Starting ApexRAG API server on http://0.0.0.0:8000…"
	$(PYTHON) -m apex_rag serve

serve-reload:                     ## Start the API server with auto-reload (dev only)
	$(INFO) "Starting ApexRAG API server with hot-reload…"
	$(PYTHON) -m apex_rag serve --reload

# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

.PHONY: info list

info:                             ## Show system information and project version
	$(INFO) "ApexRAG system info…"
	$(PYTHON) -m apex_rag info

list:                             ## List all indexed documents
	$(INFO) "Listing documents…"
	$(PYTHON) -m apex_rag list

# ══════════════════════════════════════════════════════════════════════════
# Tox — Multi-Python Testing
# ══════════════════════════════════════════════════════════════════════════

.PHONY: tox tox-py310 tox-py311 tox-py312 tox-lint tox-docs

tox:                              ## Run tox across all environments
	$(INFO) "Running tox…"
	tox

tox-py310:                        ## Run tests on Python 3.10
	tox -e py310

tox-py311:                        ## Run tests on Python 3.11
	tox -e py311

tox-py312:                        ## Run tests on Python 3.12
	tox -e py312

tox-lint:                         ## Run lint environment (ruff + mypy)
	tox -e lint

tox-docs:                         ## Build documentation with mkdocs
	tox -e docs

# ══════════════════════════════════════════════════════════════════════════
# Documentation
# ══════════════════════════════════════════════════════════════════════════

.PHONY: docs docs-serve

docs:                             ## Build documentation site
	$(INFO) "Building documentation…"
	mkdocs build --strict
	$(OK) "Documentation built in ./site/"

docs-serve:                       ## Serve documentation locally with hot-reload
	$(INFO) "Serving documentation on http://127.0.0.1:8000…"
	mkdocs serve

# ══════════════════════════════════════════════════════════════════════════
# Docker
# ══════════════════════════════════════════════════════════════════════════

.PHONY: docker-build docker-up docker-down docker-logs

docker-build:                     ## Build Docker images
	$(INFO) "Building Docker images…"
	docker compose build

docker-up:                        ## Start all Docker services
	$(INFO) "Starting Docker services…"
	docker compose up -d
	$(OK) "Services running.  API at http://localhost:8000"

docker-down:                      ## Stop all Docker services
	$(INFO) "Stopping Docker services…"
	docker compose down

docker-logs:                      ## Tail Docker logs
	docker compose logs -f

# ══════════════════════════════════════════════════════════════════════════
# Utility
# ══════════════════════════════════════════════════════════════════════════

.PHONY: help version

version:                          ## Show installed ApexRAG version
	@$(PYTHON) -c "from apex_rag import __version__; print(__version__)"

help:                             ## Show this help message
	@printf "\n$(BOLD)ApexRAG Development Commands$(RESET)\n"
	@printf "$(CYAN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(RESET)\n"
	@grep -Eh '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-22s$(RESET) %s\n", $$1, $$2}'
	@printf "\n"
	@printf "  $(CYAN)Quick start:$(RESET)  make install  &&  make check\n\n"
