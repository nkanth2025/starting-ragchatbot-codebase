"""Tests for CourseSearchTool.execute() method."""

import sys
import os
from unittest.mock import Mock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_tools import CourseSearchTool
from vector_store import SearchResults


class TestCourseSearchToolExecute:
    """Test suite for CourseSearchTool.execute() method."""

    def test_execute_returns_formatted_results(
        self, mock_vector_store, sample_search_results
    ):
        """Test that execute returns properly formatted search results."""
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="What is RAG?")

        # Should contain course context headers
        assert "[Advanced Retrieval for AI with Chroma - Lesson 1]" in result
        assert "[MCP: Build Rich-Context AI Apps - Lesson 3]" in result
        # Should contain actual content
        assert "RAG stands for Retrieval-Augmented Generation" in result

    def test_execute_populates_last_sources(self, mock_vector_store):
        """Test that execute populates last_sources correctly."""
        tool = CourseSearchTool(mock_vector_store)

        tool.execute(query="What is RAG?")

        assert len(tool.last_sources) == 2
        assert (
            tool.last_sources[0]["text"]
            == "Advanced Retrieval for AI with Chroma - Lesson 1"
        )
        assert tool.last_sources[0]["url"] == "https://example.com/lesson1"

    def test_execute_with_course_filter(self, mock_vector_store):
        """Test that execute passes course_name filter to vector store."""
        tool = CourseSearchTool(mock_vector_store)

        tool.execute(query="What is RAG?", course_name="Chroma")

        mock_vector_store.search.assert_called_once_with(
            query="What is RAG?", course_name="Chroma", lesson_number=None
        )

    def test_execute_with_lesson_filter(self, mock_vector_store):
        """Test that execute passes lesson_number filter to vector store."""
        tool = CourseSearchTool(mock_vector_store)

        tool.execute(query="What is RAG?", lesson_number=3)

        mock_vector_store.search.assert_called_once_with(
            query="What is RAG?", course_name=None, lesson_number=3
        )

    def test_execute_with_both_filters(self, mock_vector_store):
        """Test that execute passes both filters to vector store."""
        tool = CourseSearchTool(mock_vector_store)

        tool.execute(query="What is RAG?", course_name="MCP", lesson_number=2)

        mock_vector_store.search.assert_called_once_with(
            query="What is RAG?", course_name="MCP", lesson_number=2
        )

    def test_execute_returns_error_message_on_search_error(self):
        """Test that execute returns error message when search fails."""
        mock_store = Mock()
        mock_store.search.return_value = SearchResults.empty(
            "Search error: Connection failed"
        )

        tool = CourseSearchTool(mock_store)
        result = tool.execute(query="What is RAG?")

        assert result == "Search error: Connection failed"

    def test_execute_returns_no_results_message(self, mock_vector_store_empty):
        """Test that execute returns appropriate message when no results found."""
        tool = CourseSearchTool(mock_vector_store_empty)

        result = tool.execute(query="nonexistent topic")

        assert "No relevant content found" in result

    def test_execute_returns_no_results_with_course_filter(
        self, mock_vector_store_empty
    ):
        """Test no results message includes course filter info."""
        tool = CourseSearchTool(mock_vector_store_empty)

        result = tool.execute(query="nonexistent", course_name="MCP")

        assert "No relevant content found" in result
        assert "MCP" in result

    def test_execute_returns_no_results_with_lesson_filter(
        self, mock_vector_store_empty
    ):
        """Test no results message includes lesson filter info."""
        tool = CourseSearchTool(mock_vector_store_empty)

        result = tool.execute(query="nonexistent", lesson_number=5)

        assert "No relevant content found" in result
        assert "lesson 5" in result

    def test_execute_with_invalid_course_name(self):
        """Test that execute handles invalid course name gracefully."""
        mock_store = Mock()
        mock_store.search.return_value = SearchResults.empty(
            "No course found matching 'InvalidCourse'"
        )

        tool = CourseSearchTool(mock_store)
        result = tool.execute(query="What is RAG?", course_name="InvalidCourse")

        assert "No course found matching 'InvalidCourse'" in result


class TestCourseSearchToolWithZeroMaxResults:
    """Tests that reveal the MAX_RESULTS=0 bug."""

    def test_execute_with_zero_max_results_returns_empty(
        self, mock_vector_store_zero_results
    ):
        """
        This test reveals the bug: when MAX_RESULTS=0, search returns no results.

        In production, config.py has MAX_RESULTS=0, which causes VectorStore.search()
        to call ChromaDB with n_results=0, returning empty results for all queries.
        """
        tool = CourseSearchTool(mock_vector_store_zero_results)

        result = tool.execute(query="What is RAG?")

        # With MAX_RESULTS=0, we get no results even for valid queries
        assert "No relevant content found" in result
        # This confirms the bug - valid queries return empty results

    def test_max_results_zero_causes_empty_sources(
        self, mock_vector_store_zero_results
    ):
        """Test that zero max_results causes empty sources list."""
        tool = CourseSearchTool(mock_vector_store_zero_results)

        tool.execute(query="What is RAG?")

        # No sources populated due to zero results
        assert tool.last_sources == []


class TestCourseSearchToolFormatting:
    """Test result formatting in CourseSearchTool."""

    def test_format_results_includes_lesson_numbers(self, mock_vector_store):
        """Test that formatted results include lesson numbers."""
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="test")

        assert "Lesson 1" in result
        assert "Lesson 3" in result

    def test_format_results_separates_documents(self, mock_vector_store):
        """Test that multiple documents are separated properly."""
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="test")

        # Results should be separated by double newlines
        assert "\n\n" in result
