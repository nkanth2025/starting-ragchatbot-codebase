# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A RAG (Retrieval-Augmented Generation) chatbot for querying course materials. Uses ChromaDB for vector storage, Anthropic's Claude for AI responses, and FastAPI for the web interface.

## Commands

```bash
# Install dependencies
uv sync

# Run the application (from project root)
./run.sh
# Or manually:
cd backend && uv run uvicorn app:app --reload --port 8000

# Access the app
# Web UI: http://localhost:8000
# API docs: http://localhost:8000/docs
```

## Environment Setup

Create `.env` in project root:
```
ANTHROPIC_API_KEY=your_key_here
```

## Architecture

The system follows a layered RAG architecture where queries flow through tool-based search:

```
User Query → FastAPI (app.py) → RAGSystem → AIGenerator
                                    ↓
                              ToolManager → CourseSearchTool → VectorStore → ChromaDB
```

### Key Components

**RAGSystem** ([backend/rag_system.py](backend/rag_system.py)) - Main orchestrator that:
- Initializes all components with config
- Manages query flow through the tool-based search system
- Handles document ingestion from the `docs/` folder on startup

**AIGenerator** ([backend/ai_generator.py](backend/ai_generator.py)) - Claude API wrapper that:
- Sends queries with tool definitions
- Handles tool execution loop (calls tools, feeds results back to Claude)
- Maintains conversation context via SessionManager

**VectorStore** ([backend/vector_store.py](backend/vector_store.py)) - ChromaDB interface with two collections:
- `course_catalog`: Course metadata for semantic course name resolution
- `course_content`: Chunked course content for retrieval

**CourseSearchTool** ([backend/search_tools.py](backend/search_tools.py)) - Anthropic tool that:
- Provides `search_course_content` tool definition
- Handles course name resolution via vector similarity
- Filters by course and lesson number

**DocumentProcessor** ([backend/document_processor.py](backend/document_processor.py)) - Parses course documents with expected format:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [name]

Lesson 0: [title]
Lesson Link: [url]
[content...]

Lesson 1: [title]
...
```

### Data Models ([backend/models.py](backend/models.py))

- `Course`: Title, link, instructor, list of lessons
- `Lesson`: Number, title, link
- `CourseChunk`: Content chunk with course/lesson metadata

### Frontend

Static HTML/CSS/JS served from `frontend/` directory. Communicates with `/api/query` and `/api/courses` endpoints.

## Configuration ([backend/config.py](backend/config.py))

Key settings:
- `CHUNK_SIZE`: 800 chars per vector chunk
- `CHUNK_OVERLAP`: 100 chars overlap between chunks
- `MAX_RESULTS`: 5 search results returned
- `MAX_HISTORY`: 2 conversation exchanges remembered
- `EMBEDDING_MODEL`: all-MiniLM-L6-v2
- `ANTHROPIC_MODEL`: claude-sonnet-4-20250514

## Adding Course Documents

Place `.pdf`, `.docx`, or `.txt` files in the `docs/` folder. Documents are loaded automatically on server startup. New courses are added; existing courses (matched by title) are skipped.
