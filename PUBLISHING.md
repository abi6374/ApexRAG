# Publishing ApexRAG to PyPI

This guide outlines the exact steps required to publish new versions of the **ApexRAG** library to the Python Package Index (PyPI).

---

## Prerequisites

Ensure you have the required publishing tools installed in your environment:

```bash
pip install build twine
```

---

## Step 1: Bump the Version

Before publishing a new release, you must increment the version number. PyPI will reject uploads if the version number already exists.

1. Open `pyproject.toml`.
2. Locate the `[project]` section at the top.
3. Update the `version` string (e.g., from `"0.1.0"` to `"0.1.1"` or `"0.2.0"`):
### python -m twine upload dist/* -u __token__ -p <token> ###
```toml
[project]
name = "apex-rag"
version = "0.1.1"  # <--- Update this!
description = "A production-grade, local-first Agentic RAG library..."
```

---

## Step 2: Build the Package

Before uploading, you need to compile the source code into distribution formats (a source archive `.tar.gz` and a compiled wheel `.whl`).

It is highly recommended to delete old distribution files first so you don't accidentally upload older versions:

```bash
# Windows (PowerShell)
Remove-Item -Recurse -Force dist\*

# Mac/Linux
rm -rf dist/*
```

Run the following command in the root directory (where `pyproject.toml` is located):

```bash
python -m build
```

This will create a `dist/` directory containing two files, for example:
- `apex_rag-0.1.1-py3-none-any.whl`
- `apex_rag-0.1.1.tar.gz`

---

## Step 3: Upload to PyPI

Use `twine` to securely upload the generated distribution files to PyPI using your API token. 

Run the following command, replacing `<YOUR_TOKEN>` with your actual PyPI token (which starts with `pypi-`):

```bash
python -m twine upload dist/* -u __token__ -p <YOUR_TOKEN>
```

### Example:
```bash
python -m twine upload dist/* -u __token__ -p pypi-AgEIcHlwaS5vcmcCJGE3Y...
```

---

## Step 4: Verify the Release

Once the upload is complete, `twine` will output a link to your project page. 

1. Visit **[https://pypi.org/project/apex-rag/](https://pypi.org/project/apex-rag/)** to ensure the latest version is displayed.
2. In a clean terminal or a new Python environment, test the installation:

```bash
pip install --upgrade apex-rag
```

---

## Optional: Automating with GitHub Actions

If you want to automate this process so that PyPI publishes automatically every time you create a GitHub Release, you can add a `.github/workflows/publish.yml` file. Let me know if you would like the configuration file for that!
