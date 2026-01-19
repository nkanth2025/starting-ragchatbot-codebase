"""Shared test fixtures for RAG chatbot tests."""

import sys
import os
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, MagicMock, patch, AsyncMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel

# Add backend to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_store import SearchResults


@dataclass
class TestConfig:
    """Test configuration with proper MAX_RESULTS value."""

    ANTHROPIC_API_KEY: str = "test-api-key"
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    MAX_RESULTS: int = 5  # Proper value for testing
    MAX_HISTORY: int = 2
    CHROMA_PATH: str = "./test_chroma_db"


@dataclass
class BrokenConfig:
    """Configuration that mirrors the broken production config."""

    ANTHROPIC_API_KEY: str = "test-api-key"
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    MAX_RESULTS: int = 0  # This is the bug!
    MAX_HISTORY: int = 2
    CHROMA_PATH: str = "./test_chroma_db"


@pytest.fixture
def test_config():
    """Provide a working test configuration."""
    return TestConfig()


@pytest.fixture
def broken_config():
    """Provide the broken configuration to test failure cases."""
    return BrokenConfig()


@pytest.fixture
def sample_search_results():
    """Provide sample search results for testing."""
    return SearchResults(
        documents=[
            "RAG stands for Retrieval-Augmented Generation. It combines retrieval with generation.",
            "MCP is the Model Context Protocol for building AI applications.",
        ],
        metadata=[
            {
                "course_title": "Advanced Retrieval for AI with Chroma",
                "lesson_number": 1,
            },
            {"course_title": "MCP: Build Rich-Context AI Apps", "lesson_number": 3},
        ],
        distances=[0.25, 0.35],
    )


@pytest.fixture
def empty_search_results():
    """Provide empty search results."""
    return SearchResults(documents=[], metadata=[], distances=[])


@pytest.fixture
def mock_vector_store(sample_search_results):
    """Create a mock vector store that returns test data."""
    mock_store = Mock()
    mock_store.search.return_value = sample_search_results
    mock_store.max_results = 5
    mock_store.get_lesson_link.return_value = "https://example.com/lesson1"
    mock_store.get_course_outline.return_value = {
        "course_title": "Test Course",
        "course_link": "https://example.com/course",
        "instructor": "Test Instructor",
        "lessons": [
            {"lesson_number": 1, "lesson_title": "Introduction"},
            {"lesson_number": 2, "lesson_title": "Advanced Topics"},
        ],
    }
    return mock_store


@pytest.fixture
def mock_vector_store_empty(empty_search_results):
    """Create a mock vector store that returns empty results."""
    mock_store = Mock()
    mock_store.search.return_value = empty_search_results
    mock_store.max_results = 5
    mock_store.get_lesson_link.return_value = None
    return mock_store


@pytest.fixture
def mock_vector_store_zero_results():
    """Create a mock vector store simulating MAX_RESULTS=0 bug."""
    mock_store = Mock()
    # Simulates what happens when n_results=0
    mock_store.search.return_value = SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error=None,  # ChromaDB may return empty without error
    )
    mock_store.max_results = 0  # The bug!
    return mock_store


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client."""
    mock_client = Mock()
    return mock_client


@pytest.fixture
def mock_tool_use_response():
    """Create a mock response that includes tool use."""
    response = Mock()
    response.stop_reason = "tool_use"

    tool_use_block = Mock()
    tool_use_block.type = "tool_use"
    tool_use_block.name = "search_course_content"
    tool_use_block.id = "tool_123"
    tool_use_block.input = {"query": "What is RAG?"}

    response.content = [tool_use_block]
    return response


@pytest.fixture
def mock_text_response():
    """Create a mock text response (no tool use)."""
    response = Mock()
    response.stop_reason = "end_turn"

    text_block = Mock()
    text_block.type = "text"
    text_block.text = "RAG stands for Retrieval-Augmented Generation."

    response.content = [text_block]
    return response


# ============================================================
# API Testing Fixtures
# ============================================================

class QueryRequest(BaseModel):
    """Request model for course queries."""
    query: str
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    """Response model for course queries."""
    answer: str
    sources: List[Dict[str, Any]]
    session_id: str


class CourseStats(BaseModel):
    """Response model for course statistics."""
    total_courses: int
    course_titles: List[str]


@pytest.fixture
def mock_rag_system():
    """Create a mock RAG system for API testing."""
    mock_system = Mock()

    # Mock session manager
    mock_system.session_manager = Mock()
    mock_system.session_manager.create_session.return_value = "test-session-123"
    mock_system.session_manager.clear_session.return_value = None

    # Mock query method
    mock_system.query.return_value = (
        "RAG stands for Retrieval-Augmented Generation.",
        [{"text": "Test Course - Lesson 1", "url": "https://example.com/lesson1"}]
    )

    # Mock get_course_analytics
    mock_system.get_course_analytics.return_value = {
        "total_courses": 3,
        "course_titles": ["Course A", "Course B", "Course C"]
    }

    return mock_system


@pytest.fixture
def mock_rag_system_error():
    """Create a mock RAG system that raises errors."""
    mock_system = Mock()
    mock_system.session_manager = Mock()
    mock_system.session_manager.create_session.return_value = "test-session-123"
    mock_system.query.side_effect = Exception("RAG system error")
    mock_system.get_course_analytics.side_effect = Exception("Analytics error")
    return mock_system


@pytest.fixture
def mock_rag_system_empty():
    """Create a mock RAG system with empty results."""
    mock_system = Mock()
    mock_system.session_manager = Mock()
    mock_system.session_manager.create_session.return_value = "test-session-456"
    mock_system.query.return_value = (
        "I don't have information about that topic.",
        []
    )
    mock_system.get_course_analytics.return_value = {
        "total_courses": 0,
        "course_titles": []
    }
    return mock_system


def create_test_app(mock_rag_system):
    """
    Create a test FastAPI app with mocked RAG system.

    This avoids importing the main app which mounts static files
    that don't exist in the test environment.
    """
    app = FastAPI(title="Test Course Materials RAG System")

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        """Process a query and return response with sources."""
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()

            answer, sources = mock_rag_system.query(request.query, session_id)

            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        """Get course analytics and statistics."""
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/session/{session_id}")
    async def delete_session(session_id: str):
        """Clear a conversation session."""
        try:
            mock_rag_system.session_manager.clear_session(session_id)
            return {"status": "ok"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/")
    async def root():
        """Root endpoint for health check."""
        return {"status": "healthy", "service": "RAG Chatbot API"}

    return app


@pytest.fixture
def test_client(mock_rag_system):
    """Create a test client with mocked RAG system."""
    app = create_test_app(mock_rag_system)
    return TestClient(app)


@pytest.fixture
def test_client_error(mock_rag_system_error):
    """Create a test client with error-raising RAG system."""
    app = create_test_app(mock_rag_system_error)
    return TestClient(app)


@pytest.fixture
def test_client_empty(mock_rag_system_empty):
    """Create a test client with empty results RAG system."""
    app = create_test_app(mock_rag_system_empty)
    return TestClient(app)


@pytest.fixture
def sample_query_request():
    """Provide a sample query request."""
    return {"query": "What is RAG?", "session_id": None}


@pytest.fixture
def sample_query_request_with_session():
    """Provide a sample query request with session ID."""
    return {"query": "Tell me more about that", "session_id": "existing-session-789"}
