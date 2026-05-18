"""Tests for scheduled ingestion configuration."""

import pytest

from ragforge.core.config import RAGConfig, DataSourceConfig
from ragforge.deployment.scheduler import IngestionScheduler


class TestIngestionScheduler:
    """Tests for IngestionScheduler class."""

    def test_hourly_cron(self):
        """Test hourly cron expression."""
        scheduler = IngestionScheduler()
        expr = scheduler.get_cron_expression("hourly")
        assert expr == "cron(0 * * * ? *)"

    def test_daily_cron(self):
        """Test daily cron expression."""
        scheduler = IngestionScheduler()
        expr = scheduler.get_cron_expression("daily")
        assert expr == "cron(0 0 * * ? *)"

    def test_weekly_cron(self):
        """Test weekly cron expression."""
        scheduler = IngestionScheduler()
        expr = scheduler.get_cron_expression("weekly")
        assert expr == "cron(0 0 ? * MON *)"

    def test_every_6_hours_cron(self):
        """Test every 6 hours cron expression."""
        scheduler = IngestionScheduler()
        expr = scheduler.get_cron_expression("every_6_hours")
        assert expr == "cron(0 */6 * * ? *)"

    def test_every_12_hours_cron(self):
        """Test every 12 hours cron expression."""
        scheduler = IngestionScheduler()
        expr = scheduler.get_cron_expression("every_12_hours")
        assert expr == "cron(0 */12 * * ? *)"

    def test_twice_daily_cron(self):
        """Test twice daily cron expression."""
        scheduler = IngestionScheduler()
        expr = scheduler.get_cron_expression("twice_daily")
        assert expr == "cron(0 0,12 * * ? *)"

    def test_weekdays_cron(self):
        """Test weekdays cron expression."""
        scheduler = IngestionScheduler()
        expr = scheduler.get_cron_expression("weekdays")
        assert expr == "cron(0 0 ? * MON-FRI *)"

    def test_custom_cron_passthrough(self):
        """Test that custom cron expressions are passed through."""
        scheduler = IngestionScheduler()
        custom = "cron(15 10 * * ? *)"
        expr = scheduler.get_cron_expression(custom)
        assert expr == custom

    def test_custom_rate_passthrough(self):
        """Test that custom rate expressions are passed through."""
        scheduler = IngestionScheduler()
        custom = "rate(2 hours)"
        expr = scheduler.get_cron_expression(custom)
        assert expr == custom

    def test_invalid_frequency_raises(self):
        """Test that invalid frequency raises ValueError."""
        scheduler = IngestionScheduler()
        with pytest.raises(ValueError, match="Unknown frequency"):
            scheduler.get_cron_expression("every_minute")

    def test_get_rate_expression_hourly(self):
        """Test hourly rate expression."""
        scheduler = IngestionScheduler()
        expr = scheduler.get_rate_expression("hourly")
        assert expr == "rate(1 hour)"

    def test_get_rate_expression_daily(self):
        """Test daily rate expression."""
        scheduler = IngestionScheduler()
        expr = scheduler.get_rate_expression("daily")
        assert expr == "rate(1 day)"

    def test_get_rate_expression_weekly(self):
        """Test weekly rate expression."""
        scheduler = IngestionScheduler()
        expr = scheduler.get_rate_expression("weekly")
        assert expr == "rate(7 days)"

    def test_get_rate_expression_passthrough(self):
        """Test rate expression passthrough."""
        scheduler = IngestionScheduler()
        expr = scheduler.get_rate_expression("rate(3 hours)")
        assert expr == "rate(3 hours)"

    def test_get_rate_expression_invalid(self):
        """Test invalid rate expression raises ValueError."""
        scheduler = IngestionScheduler()
        with pytest.raises(ValueError, match="No rate expression"):
            scheduler.get_rate_expression("twice_daily")

    def test_generate_schedule_config(self):
        """Test generating schedule config from RAGConfig."""
        scheduler = IngestionScheduler()
        config = RAGConfig(
            project_name="test-project",
            data_sources=[
                DataSourceConfig(name="docs", type="s3", update_frequency="daily"),
                DataSourceConfig(name="api-data", type="api", update_frequency="hourly"),
            ],
        )

        schedule_config = scheduler.generate_schedule_config(config)

        assert schedule_config["project_name"] == "test-project"
        assert len(schedule_config["schedules"]) == 2

        assert schedule_config["schedules"][0]["source_name"] == "docs"
        assert schedule_config["schedules"][0]["frequency"] == "daily"
        assert schedule_config["schedules"][0]["cron_expression"] == "cron(0 0 * * ? *)"
        assert schedule_config["schedules"][0]["enabled"] is True

        assert schedule_config["schedules"][1]["source_name"] == "api-data"
        assert schedule_config["schedules"][1]["frequency"] == "hourly"
        assert schedule_config["schedules"][1]["cron_expression"] == "cron(0 * * * ? *)"

    def test_generate_schedule_config_unknown_frequency_defaults(self):
        """Test that unknown frequency defaults to daily."""
        scheduler = IngestionScheduler()
        config = RAGConfig(
            project_name="test-project",
            data_sources=[
                DataSourceConfig(name="docs", type="s3", update_frequency="biweekly"),
            ],
        )

        schedule_config = scheduler.generate_schedule_config(config)

        # Should default to daily
        assert schedule_config["schedules"][0]["cron_expression"] == "cron(0 0 * * ? *)"

    def test_generate_schedule_config_empty_sources(self):
        """Test schedule config with no data sources."""
        scheduler = IngestionScheduler()
        config = RAGConfig(project_name="test-project", data_sources=[])

        schedule_config = scheduler.generate_schedule_config(config)

        assert schedule_config["schedules"] == []
