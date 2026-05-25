---
name: 🐛 Bug Report
about: Report a reproducible bug in ApexRAG
title: "[Bug] "
labels: bug
assignees: ""
---

## 🐛 Description

<!-- A clear and concise description of what the bug is. -->

## 🔁 Steps to Reproduce

<!-- Minimal reproduction steps. Include code snippets if possible. -->

```python
# Example:
import asyncio
from apex_rag import ApexIndex

async def reproduce():
    index = await ApexIndex.create(db_url="sqlite+aiosqlite:///:memory:")
    ...
```

1. ...
2. ...
3. ...

## ✅ Expected Behavior

<!-- What should have happened? -->

## ❌ Actual Behavior

<!-- What actually happened? Include the full error traceback if applicable. -->

```
(Paste error traceback here)
```

## 🌍 Environment

- **OS:** <!-- e.g., macOS 14.5, Ubuntu 22.04, Windows 11 -->
- **Python version:** <!-- e.g., 3.10.12, 3.11.9 -->
- **ApexRAG version:** <!-- e.g., 0.1.8 (run `python -m apex_rag info`) -->
- **Install method:** <!-- `pip install apex-rag`, `pip install -e .`, Docker, etc. -->
- **LLM provider:** <!-- Ollama, OpenAI, Groq, Anthropic -->
- **Database:** <!-- SQLite, PostgreSQL -->

## 📋 Logs

<!-- Enable debug logging and attach relevant logs: set APEX_LOG_LEVEL=DEBUG -->

```
(Paste logs here)
```

## 💡 Possible Solution

<!-- (Optional) If you have an idea of what might be causing the issue or how to fix it. -->

## 📸 Screenshots

<!-- (Optional) If applicable, add screenshots to help explain the problem. -->
