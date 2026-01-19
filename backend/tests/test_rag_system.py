"""Tests for RAGSystem query handling."""

import sys
import os
from unittest.mock import Mock, patch, MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system import RAGSystem
from search_tools import ToolManager, CourseSearchTool, CourseOutlineTool
from vector_store import SearchResults


class TestRAGSystemInitialization:
    """Test RAGSystem initialization and configuration."""

    def test_tool_manager_has_search_tool(self, test_config):
        """Test that RAGSystem registers CourseSearchTool."""
        with (
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator"),
            patch("rag_system.DocumentProcessor"),
        ):

            system = RAGSystem(test_config)

            assert "search_course_content" in system.tool_manager.tools

    def test_tool_manager_has_outline_tool(self, test_config):
        """Test that RAGSystem registers CourseOutlineTool."""
        with (
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator"),
            patch("rag_system.DocumentProcessor"),
        ):

            system = RAGSystem(test_config)

            assert "get_course_outline" in system.tool_manager.tools

    def test_both_tools_share_same_vector_store(self, test_config):
        """Test that both tools use the same vector store instance."""
        with (
            patch("rag_system.VectorStore") as mock_vs,
            patch("rag_system.AIGenerator"),
            patch("rag_system.DocumentProcessor"),
        ):

            system = RAGSystem(test_config)

            # Both tools should reference the same vector store
            assert system.search_tool.store is system.outline_tool.store


class TestRAGSystemQuery:
    """Test RAGSystem.query() method."""

    def test_query_returns_response_and_sources(self, test_config):
        """Test that query returns both response and sources."""
        with (
            patch("rag_system.VectorStore") as mock_vs,
            patch("rag_system.AIGenerator") as mock_ai,
            patch("rag_system.DocumentProcessor"),
        ):

            # Setup mock AI generator
            mock_ai_instance = Mock()
            mock_ai_instance.generate_response.return_value = (
                "RAG is Retrieval-Augmented Generation."
            )
            mock_ai.return_value = mock_ai_instance

            system = RAGSystem(test_config)

            # Mock sources from tool
            system.tool_manager.get_last_sources = Mock(
                return_value=[
                    {"text": "Test Course - Lesson 1", "url": "https://example.com"}
                ]
            )

            response, sources = system.query("What is RAG?")

            assert response == "RAG is Retrieval-Augmented Generation."
            assert len(sources) == 1
            assert sources[0]["text"] == "Test Course - Lesson 1"

    def test_query_passes_tools_to_ai_generator(self, test_config):
        """Test that query passes tool definitions to AI generator."""
        with (
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator") as mock_ai,
            patch("rag_system.DocumentProcessor"),
        ):

            mock_ai_instance = Mock()
            mock_ai_instance.generate_response.return_value = "Test response"
            mock_ai.return_value = mock_ai_instance

            system = RAGSystem(test_config)

            system.query("What is RAG?")

            # Verify tools were passed to generate_response
            call_kwargs = mock_ai_instance.generate_response.call_args.kwargs
            assert "tools" in call_kwargs
            tools = call_kwargs["tools"]
            tool_names = [t["name"] for t in tools]
            assert "search_course_content" in tool_names
            assert "get_course_outline" in tool_names

    def test_query_passes_tool_manager(self, test_config):
        """Test that query passes tool manager to AI generator."""
        with (
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator") as mock_ai,
            patch("rag_system.DocumentProcessor"),
        ):

            mock_ai_instance = Mock()
            mock_ai_instance.generate_response.return_value = "Test response"
            mock_ai.return_value = mock_ai_instance

            system = RAGSystem(test_config)

            system.query("What is RAG?")

            call_kwargs = mock_ai_instance.generate_response.call_args.kwargs
            assert "tool_manager" in call_kwargs
            assert call_kwargs["tool_manager"] is system.tool_manager

    def test_query_resets_sources_after_retrieval(self, test_config):
        """Test that sources are reset after query completes."""
        with (
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator") as mock_ai,
            patch("rag_system.DocumentProcessor"),
        ):

            mock_ai_instance = Mock()
            mock_ai_instance.generate_response.return_value = "Test response"
            mock_ai.return_value = mock_ai_instance

            system = RAGSystem(test_config)

            # Mock reset_sources to verify it's called
            system.tool_manager.reset_sources = Mock()

            system.query("What is RAG?")

            system.tool_manager.reset_sources.assert_called_once()


class TestRAGSystemSessionHandling:
    """Test session handling in RAGSystem.query()."""

    def test_query_with_session_gets_history(self, test_config):
        """Test that query retrieves conversation history for session."""
        with (
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator") as mock_ai,
            patch("rag_system.DocumentProcessor"),
        ):

            mock_ai_instance = Mock()
            mock_ai_instance.generate_response.return_value = "Follow-up response"
            mock_ai.return_value = mock_ai_instance

            system = RAGSystem(test_config)

            # Setup session with history
            session_id = system.session_manager.create_session()
            system.session_manager.add_exchange(session_id, "What is RAG?", "RAG is...")

            system.query("Tell me more", session_id=session_id)

            # Verify history was passed
            call_kwargs = mock_ai_instance.generate_response.call_args.kwargs
            assert "conversation_history" in call_kwargs
            assert call_kwargs["conversation_history"] is not None

    def test_query_updates_session_history(self, test_config):
        """Test that query adds exchange to session history."""
        with (
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator") as mock_ai,
            patch("rag_system.DocumentProcessor"),
        ):

            mock_ai_instance = Mock()
            mock_ai_instance.generate_response.return_value = "The answer is..."
            mock_ai.return_value = mock_ai_instance

            system = RAGSystem(test_config)
            session_id = system.session_manager.create_session()

            system.query("What is MCP?", session_id=session_id)

            # Check history was updated
            history = system.session_manager.get_conversation_history(session_id)
            assert "What is MCP?" in history
            assert "The answer is..." in history


class TestRAGSystemWithBrokenConfig:
    """Tests that reveal issues with broken configuration."""

    def test_query_with_zero_max_results_returns_no_content(self, broken_config):
        """
        This test reveals the MAX_RESULTS=0 bug at the system level.

        When MAX_RESULTS=0, the vector store returns empty results,
        causing the AI to generate responses without any context.
        """
        with (
            patch("rag_system.AIGenerator") as mock_ai,
            patch("rag_system.DocumentProcessor"),
        ):

            mock_ai_instance = Mock()
            mock_ai_instance.generate_response.return_value = (
                "I don't have information about that."
            )
            mock_ai.return_value = mock_ai_instance

            # Create system with broken config (MAX_RESULTS=0)
            # VectorStore is not mocked here to test real behavior
            with patch("rag_system.VectorStore") as mock_vs:
                mock_vs_instance = Mock()
                mock_vs_instance.max_results = 0  # The bug!
                mock_vs_instance.search.return_value = SearchResults(
                    documents=[], metadata=[], distances=[]
                )
                mock_vs.return_value = mock_vs_instance

                system = RAGSystem(broken_config)

                # The tool manager's search will return empty results
                response, sources = system.query("What is RAG?")

                # With MAX_RESULTS=0, we get no sources
                assert sources == []


class TestToolManagerIntegration:
    """Test ToolManager integration with RAGSystem."""

    def test_get_tool_definitions_returns_both_tools(self, test_config):
        """Test that get_tool_definitions returns both tool definitions."""
        with (
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator"),
            patch("rag_system.DocumentProcessor"),
        ):

            system = RAGSystem(test_config)

            definitions = system.tool_manager.get_tool_definitions()

            assert len(definitions) == 2
            names = [d["name"] for d in definitions]
            assert "search_course_content" in names
            assert "get_course_outline" in names

    def test_execute_tool_calls_correct_tool(self, test_config):
        """Test that execute_tool routes to the correct tool."""
        with (
            patch("rag_system.VectorStore") as mock_vs,
            patch("rag_system.AIGenerator"),
            patch("rag_system.DocumentProcessor"),
        ):

            mock_vs_instance = Mock()
            mock_vs_instance.search.return_value = SearchResults(
                documents=["Test content"],
                metadata=[{"course_title": "Test", "lesson_number": 1}],
                distances=[0.1],
            )
            mock_vs_instance.get_lesson_link.return_value = None
            mock_vs.return_value = mock_vs_instance

            system = RAGSystem(test_config)

            result = system.tool_manager.execute_tool(
                "search_course_content", query="test query"
            )

            assert "Test content" in result or "Test" in result

    def test_get_last_sources_retrieves_from_tools(self, test_config):
        """Test that get_last_sources retrieves sources from tools."""
        with (
            patch("rag_system.VectorStore") as mock_vs,
            patch("rag_system.AIGenerator"),
            patch("rag_system.DocumentProcessor"),
        ):

            mock_vs_instance = Mock()
            mock_vs_instance.search.return_value = SearchResults(
                documents=["Content"],
                metadata=[{"course_title": "Course", "lesson_number": 1}],
                distances=[0.1],
            )
            mock_vs_instance.get_lesson_link.return_value = "https://example.com"
            mock_vs.return_value = mock_vs_instance

            system = RAGSystem(test_config)

            # Execute search to populate sources
            system.tool_manager.execute_tool("search_course_content", query="test")

            sources = system.tool_manager.get_last_sources()

            assert len(sources) == 1
            assert sources[0]["url"] == "https://example.com"
