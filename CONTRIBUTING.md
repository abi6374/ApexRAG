# Contributing to ApexRAG

Thank you for considering contributing! We welcome bug reports, feature suggestions, documentation improvements, and pull requests.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/abi6374/apexrag.git
cd apex-rag

# Install in dev mode with all extras (or use `make install-all`)
pip install -e ".[dev,web,postgres,docling]"

# Run tests (or use `make test`)
pytest tests/ -v
```

## ⚙️ Makefile Commands

A `Makefile` is provided to speed up common development tasks.
Requires **GNU Make** and a Unix-like shell (Git Bash / WSL on Windows).

```bash
# Show all available commands
make
```

| Command | What it does |
|---------|--------------|
| `make install` | Install package with dev & web extras |
| `make install-all` | Install with all extras |
| `make lint` | Run ruff linter on `apex_rag/` |
| `make format` | Auto-format code with ruff |
| `make format-check` | Check formatting without modifying (CI) |
| `make typecheck` | Run mypy type checker |
| `make test` | Run all tests with verbose output |
| `make test-cov` | Run tests with coverage report |
| `make test-quick` | Run tests, stop on first failure |
| `make check` | **Full CI equivalent** — lint + format-check + typecheck + test-cov |
| `make build` | Build sdist + wheel |
| `make clean` | Remove build artifacts and caches |
| `make serve` | Start the FastAPI dev server |
| `make docker-up` | Start all Docker services |
| `make docs` | Build documentation site |
| `make tox` | Run tests across Python 3.10–3.12 |

```bash
# Typical workflow
echo "==> Install"
make install

echo "==> Check quality"
make check

echo "==> Start dev server"
make serve
```

## 📋 Development Guidelines

### Code Style

- **Formatting:** We use [Ruff](https://docs.astral.sh/ruff/). Run `ruff check .` and `ruff format .` before committing.
- **Line length:** 100 characters max.
- **Type hints:** All public functions must have full type annotations. Aim for `strict` MyPy compliance.
- **Naming:** Follow PEP 8. Private methods use `_leading_underscore`.

### Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

This runs Ruff and MyPy automatically on each commit.

### Testing

- All tests must pass on Python 3.10, 3.11, and 3.12.
- Tests MUST NOT require external services (Ollama, OpenAI, databases).
- Use in-memory SQLite (`sqlite+aiosqlite:///:memory:`) for storage tests.
- Use `DummyLLM` or `MagicMock` for LLM tests.
- Add tests for every new feature — aim for 90%+ coverage.

```bash
# Run all tests with coverage
pytest tests/ -v --cov=apex_rag --cov-report=term-missing

# Run against all Python versions
tox
```

### Pull Request Process

1. Open an issue describing the change before starting work.
2. Fork the repo and create a feature branch (`git checkout -b feat/amazing-feature`).
3. Make your changes, add tests, and ensure CI passes.
4. Update `CHANGELOG.md` with your changes.
5. Open a PR against the `main` branch.

### Versioning

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR** — breaking API changes
- **MINOR** — new features, backward-compatible
- **PATCH** — bug fixes, backward-compatible

## 📦 Publishing (Maintainers Only)

```bash
# 1. Update version in pyproject.toml
# 2. Update CHANGELOG.md
# 3. Build and publish
python -m build
python -m twine upload dist/*

# Or let GitHub Actions handle it — create a Release on GitHub.
```

## 🐛 Reporting Issues

Include:
- Python version and OS
- Full traceback (if applicable)
- Minimal reproduction code
- Expected vs actual behavior
