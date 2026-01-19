"""Tests for AIGenerator tool calling functionality."""

import sys
import os
from unittest.mock import Mock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator
from search_tools import ToolManager, CourseSearchTool


class TestAIGeneratorToolCalling:
    """Test suite for AIGenerator tool calling."""

    def test_generate_response_passes_tools_to_api(self):
        """Test that tools are passed to the Anthropic API."""
        with patch("ai_generator.anthropic.Anthropic") as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            # Create mock response
            text_block = Mock()
            text_block.type = "text"
            text_block.text = "Test response"
            mock_response = Mock()
            mock_response.stop_reason = "end_turn"
            mock_response.content = [text_block]
            mock_client.messages.create.return_value = mock_response

            generator = AIGenerator("test-key", "test-model")
            tools = [
                {
                    "name": "search_course_content",
                    "description": "Search",
                    "input_schema": {},
                }
            ]

            generator.generate_response("What is RAG?", tools=tools)

            # Verify tools were passed to the API
            call_kwargs = mock_client.messages.create.call_args.kwargs
            assert "tools" in call_kwargs
            assert call_kwargs["tools"] == tools

    def test_generate_response_handles_tool_use_response(self):
        """Test that tool use responses trigger tool execution."""
        with patch("ai_generator.anthropic.Anthropic") as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            # First response: tool use
            tool_use_block = Mock()
            tool_use_block.type = "tool_use"
            tool_use_block.name = "search_course_content"
            tool_use_block.id = "tool_123"
            tool_use_block.input = {"query": "What is RAG?"}

            tool_response = Mock()
            tool_response.stop_reason = "tool_use"
            tool_response.content = [tool_use_block]

            # Second response: final text
            text_block = Mock()
            text_block.type = "text"
            text_block.text = "RAG is Retrieval-Augmented Generation."

            final_response = Mock()
            final_response.stop_reason = "end_turn"
            final_response.content = [text_block]

            mock_client.messages.create.side_effect = [tool_response, final_response]

            # Create mock tool manager
            mock_tool_manager = Mock()
            mock_tool_manager.execute_tool.return_value = "Search results here"

            generator = AIGenerator("test-key", "test-model")
            tools = [
                {
                    "name": "search_course_content",
                    "description": "Search",
                    "input_schema": {},
                }
            ]

            result = generator.generate_response(
                "What is RAG?", tools=tools, tool_manager=mock_tool_manager
            )

            # Verify tool was executed
            mock_tool_manager.execute_tool.assert_called_once_with(
                "search_course_content", query="What is RAG?"
            )

            # Verify final response is returned
            assert result == "RAG is Retrieval-Augmented Generation."

    def test_generate_response_without_tools(self):
        """Test that responses work without tools."""
        with patch("ai_generator.anthropic.Anthropic") as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            text_block = Mock()
            text_block.type = "text"
            text_block.text = "General knowledge answer"

            mock_response = Mock()
            mock_response.stop_reason = "end_turn"
            mock_response.content = [text_block]
            mock_client.messages.create.return_value = mock_response

            generator = AIGenerator("test-key", "test-model")

            result = generator.generate_response("What is AI?")

            assert result == "General knowledge answer"
            # Verify no tools were passed
            call_kwargs = mock_client.messages.create.call_args.kwargs
            assert "tools" not in call_kwargs or call_kwargs.get("tools") is None

    def test_generate_response_includes_conversation_history(self):
        """Test that conversation history is included in system prompt."""
        with patch("ai_generator.anthropic.Anthropic") as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            text_block = Mock()
            text_block.type = "text"
            text_block.text = "Follow-up answer"

            mock_response = Mock()
            mock_response.stop_reason = "end_turn"
            mock_response.content = [text_block]
            mock_client.messages.create.return_value = mock_response

            generator = AIGenerator("test-key", "test-model")

            history = "User: What is RAG?\nAssistant: RAG is..."
            generator.generate_response("Tell me more", conversation_history=history)

            call_kwargs = mock_client.messages.create.call_args.kwargs
            assert "Previous conversation" in call_kwargs["system"]
            assert history in call_kwargs["system"]


class TestAIGeneratorToolExecution:
    """Test tool execution handling in AIGenerator."""

    def test_handle_tool_execution_builds_correct_messages(self):
        """Test that tool execution builds correct message chain."""
        with patch("ai_generator.anthropic.Anthropic") as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            # Tool use response
            tool_use_block = Mock()
            tool_use_block.type = "tool_use"
            tool_use_block.name = "search_course_content"
            tool_use_block.id = "tool_456"
            tool_use_block.input = {"query": "MCP protocol"}

            tool_response = Mock()
            tool_response.stop_reason = "tool_use"
            tool_response.content = [tool_use_block]

            # Final response
            text_block = Mock()
            text_block.type = "text"
            text_block.text = "MCP is the Model Context Protocol."

            final_response = Mock()
            final_response.stop_reason = "end_turn"
            final_response.content = [text_block]

            mock_client.messages.create.side_effect = [tool_response, final_response]

            mock_tool_manager = Mock()
            mock_tool_manager.execute_tool.return_value = "MCP content from search"

            generator = AIGenerator("test-key", "test-model")
            tools = [
                {
                    "name": "search_course_content",
                    "description": "Search",
                    "input_schema": {},
                }
            ]

            generator.generate_response(
                "What is MCP?", tools=tools, tool_manager=mock_tool_manager
            )

            # Check second API call has tool results
            second_call = mock_client.messages.create.call_args_list[1]
            messages = second_call.kwargs["messages"]

            # Should have: user message, assistant tool use, user tool result
            assert len(messages) == 3
            assert messages[0]["role"] == "user"
            assert messages[1]["role"] == "assistant"
            assert messages[2]["role"] == "user"

            # Tool result should be in the last user message
            tool_result = messages[2]["content"][0]
            assert tool_result["type"] == "tool_result"
            assert tool_result["tool_use_id"] == "tool_456"
            assert tool_result["content"] == "MCP content from search"

    def test_tool_manager_not_called_without_tool_use(self):
        """Test that tool manager is not called for non-tool responses."""
        with patch("ai_generator.anthropic.Anthropic") as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            text_block = Mock()
            text_block.type = "text"
            text_block.text = "Direct answer"

            mock_response = Mock()
            mock_response.stop_reason = "end_turn"
            mock_response.content = [text_block]
            mock_client.messages.create.return_value = mock_response

            mock_tool_manager = Mock()

            generator = AIGenerator("test-key", "test-model")
            tools = [
                {
                    "name": "search_course_content",
                    "description": "Search",
                    "input_schema": {},
                }
            ]

            generator.generate_response(
                "Hello", tools=tools, tool_manager=mock_tool_manager
            )

            # Tool manager should not be called
            mock_tool_manager.execute_tool.assert_not_called()


class TestAIGeneratorSystemPrompt:
    """Test system prompt configuration."""

    def test_system_prompt_mentions_search_tool(self):
        """Test that system prompt references search_course_content tool."""
        assert "search_course_content" in AIGenerator.SYSTEM_PROMPT

    def test_system_prompt_mentions_outline_tool(self):
        """Test that system prompt references get_course_outline tool."""
        assert "get_course_outline" in AIGenerator.SYSTEM_PROMPT

    def test_system_prompt_includes_usage_guidelines(self):
        """Test that system prompt has tool usage guidelines."""
        assert "Tool Usage Guidelines" in AIGenerator.SYSTEM_PROMPT
        assert "Course outline" in AIGenerator.SYSTEM_PROMPT
        assert "Content-specific" in AIGenerator.SYSTEM_PROMPT
