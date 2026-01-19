"""Shared test fixtures for RAG chatbot tests."""

import sys
import os
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, MagicMock

import pytest

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
