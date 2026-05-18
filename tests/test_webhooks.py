"""Tests for webhook notifications module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from ragforge.deployment.webhooks import WebhookNotifier


class TestWebhookNotifier:
    """Tests for WebhookNotifier class."""

    def _create_notifier(self) -> tuple:
        """Create a notifier with a temp file."""
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.close()
        path = Path(tmp.name)
        path.unlink()
        notifier = WebhookNotifier(webhook_path=path)
        return notifier, path

    def test_register_webhook(self):
        """Test registering a webhook."""
        notifier, path = self._create_notifier()
        try:
            result = notifier.register_webhook(
                url="https://example.com/hook",
                events=["ingestion_complete", "error"],
            )

            assert result["url"] == "https://example.com/hook"
            assert result["events"] == ["ingestion_complete", "error"]
        finally:
            if path.exists():
                path.unlink()

    def test_register_webhook_invalid_event(self):
        """Test that invalid event types raise ValueError."""
        notifier, path = self._create_notifier()
        try:
            with pytest.raises(ValueError, match="Invalid event types"):
                notifier.register_webhook(
                    url="https://example.com/hook",
                    events=["invalid_event"],
                )
        finally:
            if path.exists():
                path.unlink()

    def test_register_webhook_updates_existing(self):
        """Test that registering same URL updates events."""
        notifier, path = self._create_notifier()
        try:
            notifier.register_webhook(
                url="https://example.com/hook",
                events=["ingestion_complete"],
            )
            notifier.register_webhook(
                url="https://example.com/hook",
                events=["error", "budget_exceeded"],
            )

            webhooks = notifier.get_registered_webhooks()
            assert len(webhooks) == 1
            assert webhooks[0]["events"] == ["error", "budget_exceeded"]
        finally:
            if path.exists():
                path.unlink()

    def test_unregister_webhook(self):
        """Test unregistering a webhook."""
        notifier, path = self._create_notifier()
        try:
            notifier.register_webhook(
                url="https://example.com/hook",
                events=["ingestion_complete"],
            )

            result = notifier.unregister_webhook("https://example.com/hook")
            assert result is True

            webhooks = notifier.get_registered_webhooks()
            assert len(webhooks) == 0
        finally:
            if path.exists():
                path.unlink()

    def test_unregister_nonexistent_webhook(self):
        """Test unregistering a webhook that doesn't exist."""
        notifier, path = self._create_notifier()
        try:
            result = notifier.unregister_webhook("https://example.com/nonexistent")
            assert result is False
        finally:
            if path.exists():
                path.unlink()

    def test_get_registered_webhooks(self):
        """Test getting all registered webhooks."""
        notifier, path = self._create_notifier()
        try:
            notifier.register_webhook("https://a.com/hook", ["ingestion_complete"])
            notifier.register_webhook("https://b.com/hook", ["error"])

            webhooks = notifier.get_registered_webhooks()
            assert len(webhooks) == 2
            urls = [w["url"] for w in webhooks]
            assert "https://a.com/hook" in urls
            assert "https://b.com/hook" in urls
        finally:
            if path.exists():
                path.unlink()

    def test_notify_invalid_event(self):
        """Test that notify with invalid event raises ValueError."""
        notifier, path = self._create_notifier()
        try:
            with pytest.raises(ValueError, match="Invalid event type"):
                notifier.notify("invalid_event", {"data": "test"})
        finally:
            if path.exists():
                path.unlink()

    def test_notify_no_subscribers(self):
        """Test notify with no subscribers returns empty list."""
        notifier, path = self._create_notifier()
        try:
            notifier.register_webhook("https://a.com/hook", ["error"])

            # Notify an event that no one is subscribed to
            results = notifier.notify("ingestion_complete", {"docs": 10})
            assert results == []
        finally:
            if path.exists():
                path.unlink()

    @patch("urllib.request.urlopen")
    def test_notify_success(self, mock_urlopen):
        """Test successful webhook notification."""
        notifier, path = self._create_notifier()
        try:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            notifier.register_webhook(
                "https://example.com/hook",
                ["ingestion_complete"],
            )

            results = notifier.notify("ingestion_complete", {"docs": 5})

            assert len(results) == 1
            assert results[0]["url"] == "https://example.com/hook"
            assert results[0]["success"] is True
            assert results[0]["error"] is None
        finally:
            if path.exists():
                path.unlink()

    @patch("urllib.request.urlopen")
    def test_notify_failure(self, mock_urlopen):
        """Test webhook notification failure handling."""
        import urllib.error

        notifier, path = self._create_notifier()
        try:
            mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

            notifier.register_webhook(
                "https://example.com/hook",
                ["error"],
            )

            results = notifier.notify("error", {"message": "something broke"})

            assert len(results) == 1
            assert results[0]["success"] is False
            assert results[0]["error"] is not None
        finally:
            if path.exists():
                path.unlink()

    def test_persistence(self):
        """Test that webhook registrations persist to file."""
        notifier, path = self._create_notifier()
        try:
            notifier.register_webhook("https://a.com/hook", ["ingestion_complete"])

            # Create new notifier from same file
            notifier2 = WebhookNotifier(webhook_path=path)
            webhooks = notifier2.get_registered_webhooks()
            assert len(webhooks) == 1
            assert webhooks[0]["url"] == "https://a.com/hook"
        finally:
            if path.exists():
                path.unlink()

    def test_valid_events_list(self):
        """Test all valid event types can be registered."""
        notifier, path = self._create_notifier()
        try:
            all_events = list(WebhookNotifier.VALID_EVENTS)
            result = notifier.register_webhook("https://a.com/hook", all_events)
            assert result["events"] == all_events
        finally:
            if path.exists():
                path.unlink()
