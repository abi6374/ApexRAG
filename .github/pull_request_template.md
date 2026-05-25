# 📋 Pull Request

## Description

<!-- A clear and concise description of the changes in this PR. What problem does it solve? -->

Fixes #<!-- (issue number) -->

## 🧪 Type of Change

<!-- Mark the relevant option(s) with an `x`. -->

- [ ] 🐛 Bug fix (non-breaking change that fixes an issue)
- [ ] ✨ New feature (non-breaking change that adds functionality)
- [ ] 💥 Breaking change (fix or feature that changes existing API behaviour)
- [ ] 📝 Documentation update
- [ ] 🔧 Refactor / code quality improvement
- [ ] ⚡ Performance improvement
- [ ] 🧪 Test addition or improvement
- [ ] 🔨 Build / CI / dependency change

## ✅ Checklist

<!-- Mark completed items with an `x`. -->

### Code Quality

- [ ] My code follows the project's code style (`ruff check apex_rag/ && ruff format --check apex_rag/`)
- [ ] I've added type hints to all public functions and methods
- [ ] My changes pass `mypy apex_rag/ --ignore-missing-imports --python-version=3.10`
- [ ] I've removed any debug/print statements and TODO comments

### Testing

- [ ] I've added tests that cover my changes
- [ ] New and existing tests pass (`pytest tests/ -v --tb=short`)
- [ ] Tests do NOT require external services (Ollama, OpenAI, databases)

### Documentation

- [ ] I've updated the README or relevant docs if needed
- [ ] I've added a docstring to any new public functions/classes
- [ ] I've updated `CHANGELOG.md` with my changes

### Integration

- [ ] My changes are backward-compatible (or I've documented breaking changes)
- [ ] I've verified the feature works with both SQLite and PostgreSQL (if applicable)

## 📸 Screenshots (if applicable)

<!-- For UI changes, include before/after screenshots or screen recordings. -->

## 💬 Additional Notes

<!-- Any additional context, edge cases considered, or design decisions made. -->
