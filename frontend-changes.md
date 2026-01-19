# Code Quality Tools Implementation

This document describes the code quality tools added to the development workflow.

## Changes Made

### 1. Added Black Code Formatter

**File Modified:** `pyproject.toml`

Added black as a development dependency and configured its settings:

```toml
[dependency-groups]
dev = [
    "black>=24.0.0",
    "pytest>=8.0.0",
]

[tool.black]
line-length = 88
target-version = ["py313"]
include = '\.pyi?$'
exclude = '''
/(
    \.git
    | \.venv
    | venv
    | __pycache__
    | \.eggs
    | build
    | dist
    | chroma_db
)/
'''
```

**Configuration Details:**
- Line length: 88 characters (black's default)
- Target Python version: 3.13
- Excludes common non-source directories (git, venv, pycache, build artifacts, chroma_db)

### 2. Formatted All Python Files

Ran black formatter across the entire codebase. The following 13 files were reformatted:

- `backend/config.py`
- `backend/models.py`
- `backend/ai_generator.py`
- `backend/app.py`
- `backend/session_manager.py`
- `backend/rag_system.py`
- `backend/search_tools.py`
- `backend/document_processor.py`
- `backend/vector_store.py`
- `backend/tests/conftest.py`
- `backend/tests/test_course_search_tool.py`
- `backend/tests/test_ai_generator.py`
- `backend/tests/test_rag_system.py`

### 3. Created Development Scripts

**New Directory:** `scripts/`

**New Files:**

#### `scripts/quality.sh`
Runs code quality checks to verify formatting compliance.

```bash
./scripts/quality.sh
```

This script:
- Checks if black formatting is correct
- Exits with error code if issues are found
- Provides colored output for better readability

#### `scripts/format.sh`
Automatically formats all Python code.

```bash
./scripts/format.sh
```

This script:
- Runs black to format all Python files
- Should be run before committing code

## Usage

### Install Development Dependencies

```bash
uv sync
```

### Check Code Formatting

```bash
./scripts/quality.sh
# or
uv run black --check .
```

### Format Code

```bash
./scripts/format.sh
# or
uv run black .
```

## Recommended Workflow

1. Before committing, run `./scripts/quality.sh` to check formatting
2. If issues are found, run `./scripts/format.sh` to auto-fix them
3. Commit your changes

## Files Added/Modified

| File | Change Type | Description |
|------|-------------|-------------|
| `pyproject.toml` | Modified | Added dev dependencies and black configuration |
| `scripts/quality.sh` | Added | Quality check script |
| `scripts/format.sh` | Added | Code formatting script |
| `backend/*.py` | Modified | Formatted with black |
| `backend/tests/*.py` | Modified | Formatted with black |
