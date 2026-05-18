"""Webhook notifications for RAGForge events."""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional


class WebhookNotifier:
    """Sends HTTP POST notifications on RAGForge events.

    Supported events:
    - ingestion_complete: Fired after successful ingestion
    - quality_alert: Fired when quality metrics drop below threshold
    - budget_exceeded: Fired when cost budget is exceeded
    - error: Fired on pipeline errors

    Webhook registrations are stored in a JSON file.
    """

    VALID_EVENTS = (
        "ingestion_complete",
        "quality_alert",
        "budget_exceeded",
        "error",
    )

    def __init__(self, webhook_path: str | Path = "ragforge_webhooks.json"):
        """Initialize the webhook notifier.

        Args:
            webhook_path: Path to the JSON file storing webhook registrations.
        """
        self._webhook_path = Path(webhook_path)
        self._webhooks: List[Dict[str, Any]] = []
        self._load_webhooks()

    def _load_webhooks(self) -> None:
        """Load webhook registrations from file."""
        if self._webhook_path.exists():
            try:
                data = json.loads(self._webhook_path.read_text(encoding="utf-8"))
                self._webhooks = data.get("webhooks", [])
            except (json.JSONDecodeError, IOError):
                self._webhooks = []

    def _save_webhooks(self) -> None:
        """Save webhook registrations to file."""
        self._webhook_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"webhooks": self._webhooks}
        self._webhook_path.write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )

    def register_webhook(self, url: str, events: List[str]) -> Dict[str, Any]:
        """Register a webhook URL for specific events.

        Args:
            url: The HTTP(S) URL to send POST requests to.
            events: List of event types to subscribe to.

        Returns:
            The webhook registration record.

        Raises:
            ValueError: If any event type is invalid.
        """
        invalid_events = [e for e in events if e not in self.VALID_EVENTS]
        if invalid_events:
            raise ValueError(
                f"Invalid event types: {invalid_events}. "
                f"Valid events: {list(self.VALID_EVENTS)}"
            )

        registration = {
            "url": url,
            "events": events,
        }

        # Update existing registration for same URL, or add new
        existing = next((w for w in self._webhooks if w["url"] == url), None)
        if existing:
            existing["events"] = events
        else:
            self._webhooks.append(registration)

        self._save_webhooks()
        return registration

    def unregister_webhook(self, url: str) -> bool:
        """Remove a webhook registration.

        Args:
            url: The webhook URL to unregister.

        Returns:
            True if the webhook was found and removed, False otherwise.
        """
        original_count = len(self._webhooks)
        self._webhooks = [w for w in self._webhooks if w["url"] != url]

        if len(self._webhooks) < original_count:
            self._save_webhooks()
            return True
        return False

    def get_registered_webhooks(self) -> List[Dict[str, Any]]:
        """Get all registered webhooks.

        Returns:
            List of webhook registration records.
        """
        return list(self._webhooks)

    def notify(self, event_type: str, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Send notifications to all webhooks subscribed to an event.

        Args:
            event_type: The type of event that occurred.
            payload: Event data to include in the notification.

        Returns:
            List of notification results (url, success, error).
        """
        if event_type not in self.VALID_EVENTS:
            raise ValueError(
                f"Invalid event type: {event_type}. "
                f"Valid events: {list(self.VALID_EVENTS)}"
            )

        results = []
        notification_body = {
            "event": event_type,
            "payload": payload,
        }
        body_bytes = json.dumps(notification_body).encode("utf-8")

        for webhook in self._webhooks:
            if event_type not in webhook["events"]:
                continue

            url = webhook["url"]
            result = {"url": url, "success": False, "error": None}

            try:
                req = urllib.request.Request(
                    url,
                    data=body_bytes,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=10) as response:
                    result["success"] = response.status < 400
            except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
                result["error"] = str(e)
            except Exception as e:
                result["error"] = str(e)

            results.append(result)

        return results
