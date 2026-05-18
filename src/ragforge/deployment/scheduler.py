"""Scheduled ingestion configuration for RAGForge."""

from __future__ import annotations

from typing import Any, Dict

from ragforge.core.config import RAGConfig


class IngestionScheduler:
    """Generates EventBridge cron expressions from configuration.

    Supports common frequencies (hourly, daily, weekly) and custom cron expressions.
    """

    # Mapping of frequency names to EventBridge cron expressions
    FREQUENCY_MAP = {
        "hourly": "cron(0 * * * ? *)",
        "daily": "cron(0 0 * * ? *)",
        "weekly": "cron(0 0 ? * MON *)",
        "every_6_hours": "cron(0 */6 * * ? *)",
        "every_12_hours": "cron(0 */12 * * ? *)",
        "twice_daily": "cron(0 0,12 * * ? *)",
        "weekdays": "cron(0 0 ? * MON-FRI *)",
    }

    # Mapping to EventBridge rate expressions (simpler alternative)
    RATE_MAP = {
        "hourly": "rate(1 hour)",
        "daily": "rate(1 day)",
        "weekly": "rate(7 days)",
        "every_6_hours": "rate(6 hours)",
        "every_12_hours": "rate(12 hours)",
    }

    def get_cron_expression(self, frequency: str) -> str:
        """Get an EventBridge cron expression for a given frequency.

        Args:
            frequency: One of: hourly, daily, weekly, every_6_hours,
                      every_12_hours, twice_daily, weekdays, or a custom
                      cron expression prefixed with 'cron('.

        Returns:
            EventBridge-compatible cron expression.

        Raises:
            ValueError: If frequency is not recognized and not a valid cron.
        """
        # If it's already a cron expression, return as-is
        if frequency.startswith("cron(") or frequency.startswith("rate("):
            return frequency

        if frequency in self.FREQUENCY_MAP:
            return self.FREQUENCY_MAP[frequency]

        raise ValueError(
            f"Unknown frequency: {frequency}. "
            f"Valid options: {', '.join(self.FREQUENCY_MAP.keys())}, "
            f"or a custom cron/rate expression."
        )

    def get_rate_expression(self, frequency: str) -> str:
        """Get an EventBridge rate expression for a given frequency.

        Args:
            frequency: One of: hourly, daily, weekly, every_6_hours, every_12_hours.

        Returns:
            EventBridge-compatible rate expression.

        Raises:
            ValueError: If frequency is not supported as a rate expression.
        """
        if frequency.startswith("rate("):
            return frequency

        if frequency in self.RATE_MAP:
            return self.RATE_MAP[frequency]

        raise ValueError(
            f"No rate expression for frequency: {frequency}. "
            f"Valid options: {', '.join(self.RATE_MAP.keys())}"
        )

    def generate_schedule_config(self, config: RAGConfig) -> Dict[str, Any]:
        """Generate a complete schedule configuration from RAGForge config.

        Creates a schedule entry for each data source based on its
        update_frequency setting.

        Args:
            config: RAGConfig instance.

        Returns:
            Dictionary with schedule configuration for each data source.
        """
        schedules: Dict[str, Any] = {
            "project_name": config.project_name,
            "schedules": [],
        }

        for source in config.data_sources:
            frequency = source.update_frequency
            try:
                cron_expr = self.get_cron_expression(frequency)
            except ValueError:
                cron_expr = self.FREQUENCY_MAP["daily"]  # Default to daily

            schedule_entry = {
                "source_name": source.name,
                "source_type": source.type,
                "frequency": frequency,
                "cron_expression": cron_expr,
                "enabled": True,
            }
            schedules["schedules"].append(schedule_entry)

        return schedules
