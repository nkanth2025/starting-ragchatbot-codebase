"""Tests for FastAPI endpoints."""
import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestQueryEndpoint:
    """Test POST /api/query endpoint."""

    def test_query_returns_200_with_valid_request(self, test_client, sample_query_request):
        """Test that a valid query returns 200 with expected response structure."""
        response = test_client.post("/api/query", json=sample_query_request)

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data

    def test_query_creates_session_when_not_provided(self, test_client):
        """Test that a new session is created when session_id is not provided."""
        response = test_client.post("/api/query", json={"query": "What is RAG?"})

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session-123"

    def test_query_uses_provided_session_id(self, test_client, mock_rag_system):
        """Test that provided session_id is used."""
        response = test_client.post(
            "/api/query",
            json={"query": "Tell me more", "session_id": "my-session-456"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "my-session-456"

        # Verify RAG system was called with the provided session ID
        mock_rag_system.query.assert_called_with("Tell me more", "my-session-456")

    def test_query_returns_answer_from_rag_system(self, test_client):
        """Test that the answer comes from the RAG system."""
        response = test_client.post("/api/query", json={"query": "What is RAG?"})

        assert response.status_code == 200
        data = response.json()
        assert "RAG stands for Retrieval-Augmented Generation" in data["answer"]

    def test_query_returns_sources(self, test_client):
        """Test that sources are returned from the RAG system."""
        response = test_client.post("/api/query", json={"query": "What is RAG?"})

        assert response.status_code == 200
        data = response.json()
        assert len(data["sources"]) == 1
        assert data["sources"][0]["text"] == "Test Course - Lesson 1"
        assert data["sources"][0]["url"] == "https://example.com/lesson1"

    def test_query_returns_500_on_rag_error(self, test_client_error):
        """Test that RAG system errors return 500."""
        response = test_client_error.post("/api/query", json={"query": "What is RAG?"})

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "RAG system error" in data["detail"]

    def test_query_with_empty_string_still_processes(self, test_client):
        """Test that empty query string is still processed."""
        response = test_client.post("/api/query", json={"query": ""})

        assert response.status_code == 200

    def test_query_requires_query_field(self, test_client):
        """Test that query field is required."""
        response = test_client.post("/api/query", json={})

        assert response.status_code == 422  # Validation error

    def test_query_with_empty_results(self, test_client_empty):
        """Test query when RAG system returns no sources."""
        response = test_client_empty.post("/api/query", json={"query": "Unknown topic"})

        assert response.status_code == 200
        data = response.json()
        assert data["sources"] == []
        assert "don't have information" in data["answer"]


class TestCoursesEndpoint:
    """Test GET /api/courses endpoint."""

    def test_courses_returns_200(self, test_client):
        """Test that courses endpoint returns 200."""
        response = test_client.get("/api/courses")

        assert response.status_code == 200

    def test_courses_returns_total_count(self, test_client):
        """Test that courses endpoint returns total course count."""
        response = test_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 3

    def test_courses_returns_course_titles(self, test_client):
        """Test that courses endpoint returns list of course titles."""
        response = test_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert data["course_titles"] == ["Course A", "Course B", "Course C"]

    def test_courses_returns_500_on_error(self, test_client_error):
        """Test that analytics errors return 500."""
        response = test_client_error.get("/api/courses")

        assert response.status_code == 500
        data = response.json()
        assert "Analytics error" in data["detail"]

    def test_courses_with_empty_catalog(self, test_client_empty):
        """Test courses endpoint when no courses exist."""
        response = test_client_empty.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []


class TestSessionEndpoint:
    """Test DELETE /api/session/{session_id} endpoint."""

    def test_delete_session_returns_200(self, test_client):
        """Test that deleting a session returns 200."""
        response = test_client.delete("/api/session/test-session-123")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_delete_session_calls_clear_session(self, test_client, mock_rag_system):
        """Test that session manager's clear_session is called."""
        test_client.delete("/api/session/my-session-to-delete")

        mock_rag_system.session_manager.clear_session.assert_called_with("my-session-to-delete")

    def test_delete_nonexistent_session_still_returns_ok(self, test_client):
        """Test that deleting nonexistent session doesn't error."""
        response = test_client.delete("/api/session/nonexistent-session")

        assert response.status_code == 200


class TestRootEndpoint:
    """Test GET / endpoint (health check)."""

    def test_root_returns_200(self, test_client):
        """Test that root endpoint returns 200."""
        response = test_client.get("/")

        assert response.status_code == 200

    def test_root_returns_health_status(self, test_client):
        """Test that root endpoint returns health status."""
        response = test_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "RAG Chatbot API"


class TestRequestValidation:
    """Test request validation and edge cases."""

    def test_query_with_very_long_text(self, test_client):
        """Test query with very long text."""
        long_query = "What is RAG? " * 1000
        response = test_client.post("/api/query", json={"query": long_query})

        assert response.status_code == 200

    def test_query_with_special_characters(self, test_client):
        """Test query with special characters."""
        response = test_client.post(
            "/api/query",
            json={"query": "What is <script>alert('xss')</script>?"}
        )

        assert response.status_code == 200

    def test_query_with_unicode(self, test_client):
        """Test query with unicode characters."""
        response = test_client.post(
            "/api/query",
            json={"query": "What is 日本語 and émojis 🎉?"}
        )

        assert response.status_code == 200

    def test_invalid_content_type(self, test_client):
        """Test request with invalid content type."""
        response = test_client.post(
            "/api/query",
            content="query=What is RAG?",
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )

        assert response.status_code == 422

    def test_malformed_json(self, test_client):
        """Test request with malformed JSON."""
        response = test_client.post(
            "/api/query",
            content='{"query": "incomplete',
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422
