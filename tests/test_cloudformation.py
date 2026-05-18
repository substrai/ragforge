"""Tests for CloudFormation template generation."""

import yaml
import pytest

from ragforge.core.config import (
    RAGConfig,
    DataSourceConfig,
    ChunkingConfig,
    EmbeddingConfig,
    StorageConfig,
    RetrievalConfig,
    QueryConfig,
)
from ragforge.deployment.cloudformation import CloudFormationGenerator


class TestCloudFormationGenerator:
    """Tests for CloudFormationGenerator class."""

    def _create_config(self, **kwargs) -> RAGConfig:
        """Create a test config."""
        return RAGConfig(
            project_name=kwargs.get("project_name", "test-project"),
            data_sources=kwargs.get("data_sources", [
                DataSourceConfig(name="docs", type="s3", update_frequency="daily")
            ]),
            chunking=kwargs.get("chunking", ChunkingConfig()),
            embedding=kwargs.get("embedding", EmbeddingConfig()),
            storage=kwargs.get("storage", StorageConfig()),
            retrieval=kwargs.get("retrieval", RetrievalConfig()),
            query=kwargs.get("query", QueryConfig()),
        )

    def test_generate_returns_valid_yaml(self):
        """Test that generate() returns valid YAML."""
        config = self._create_config()
        generator = CloudFormationGenerator()
        result = generator.generate(config)

        # Should be valid YAML
        parsed = yaml.safe_load(result)
        assert parsed is not None
        assert isinstance(parsed, dict)

    def test_template_has_required_keys(self):
        """Test that template has required CloudFormation keys."""
        config = self._create_config()
        generator = CloudFormationGenerator()
        result = generator.generate(config)
        parsed = yaml.safe_load(result)

        assert "AWSTemplateFormatVersion" in parsed
        assert "Transform" in parsed
        assert "Resources" in parsed
        assert "Outputs" in parsed

    def test_template_has_sam_transform(self):
        """Test that template uses SAM transform."""
        config = self._create_config()
        generator = CloudFormationGenerator()
        result = generator.generate(config)
        parsed = yaml.safe_load(result)

        assert parsed["Transform"] == "AWS::Serverless-2016-10-31"

    def test_template_has_s3_bucket(self):
        """Test that template includes S3 bucket resource."""
        config = self._create_config()
        generator = CloudFormationGenerator()
        result = generator.generate(config)
        parsed = yaml.safe_load(result)

        resources = parsed["Resources"]
        bucket_resources = [
            r for r in resources.values()
            if r["Type"] == "AWS::S3::Bucket"
        ]
        assert len(bucket_resources) == 1
        assert bucket_resources[0]["Properties"]["BucketName"] == "test-project-documents"

    def test_template_has_dynamodb_table(self):
        """Test that template includes DynamoDB table resource."""
        config = self._create_config()
        generator = CloudFormationGenerator()
        result = generator.generate(config)
        parsed = yaml.safe_load(result)

        resources = parsed["Resources"]
        table_resources = [
            r for r in resources.values()
            if r["Type"] == "AWS::DynamoDB::Table"
        ]
        assert len(table_resources) == 1
        assert table_resources[0]["Properties"]["BillingMode"] == "PAY_PER_REQUEST"

    def test_template_has_lambda_functions(self):
        """Test that template includes Lambda function resources."""
        config = self._create_config()
        generator = CloudFormationGenerator()
        result = generator.generate(config)
        parsed = yaml.safe_load(result)

        resources = parsed["Resources"]
        lambda_resources = [
            r for r in resources.values()
            if r["Type"] == "AWS::Serverless::Function"
        ]
        # Should have query function and ingestion function
        assert len(lambda_resources) == 2

    def test_template_has_api_gateway_event(self):
        """Test that query function has API Gateway event."""
        config = self._create_config()
        generator = CloudFormationGenerator()
        result = generator.generate(config)
        parsed = yaml.safe_load(result)

        resources = parsed["Resources"]
        query_functions = [
            r for r in resources.values()
            if r["Type"] == "AWS::Serverless::Function"
            and "query" in r["Properties"].get("FunctionName", "")
        ]
        assert len(query_functions) == 1
        events = query_functions[0]["Properties"]["Events"]
        assert "QueryApi" in events
        assert events["QueryApi"]["Type"] == "Api"

    def test_template_has_schedule_event(self):
        """Test that ingestion function has schedule event."""
        config = self._create_config()
        generator = CloudFormationGenerator()
        result = generator.generate(config)
        parsed = yaml.safe_load(result)

        resources = parsed["Resources"]
        ingest_functions = [
            r for r in resources.values()
            if r["Type"] == "AWS::Serverless::Function"
            and "ingest" in r["Properties"].get("FunctionName", "")
        ]
        assert len(ingest_functions) == 1
        events = ingest_functions[0]["Properties"]["Events"]
        assert "ScheduledIngestion" in events
        assert events["ScheduledIngestion"]["Type"] == "Schedule"

    def test_template_uses_config_values(self):
        """Test that template uses values from config."""
        config = self._create_config(
            embedding=EmbeddingConfig(model="bedrock/cohere-embed", dimensions=768),
            retrieval=RetrievalConfig(method="semantic", top_k=10),
        )
        generator = CloudFormationGenerator()
        result = generator.generate(config)
        parsed = yaml.safe_load(result)

        resources = parsed["Resources"]
        query_functions = [
            r for r in resources.values()
            if r["Type"] == "AWS::Serverless::Function"
            and "query" in r["Properties"].get("FunctionName", "")
        ]
        env_vars = query_functions[0]["Properties"]["Environment"]["Variables"]
        assert env_vars["EMBEDDING_MODEL"] == "bedrock/cohere-embed"
        assert env_vars["EMBEDDING_DIMENSIONS"] == "768"
        assert env_vars["RETRIEVAL_TOP_K"] == "10"

    def test_hourly_schedule(self):
        """Test that hourly update frequency generates correct schedule."""
        config = self._create_config(
            data_sources=[DataSourceConfig(name="docs", type="s3", update_frequency="hourly")]
        )
        generator = CloudFormationGenerator()
        result = generator.generate(config)
        parsed = yaml.safe_load(result)

        resources = parsed["Resources"]
        ingest_functions = [
            r for r in resources.values()
            if r["Type"] == "AWS::Serverless::Function"
            and "ingest" in r["Properties"].get("FunctionName", "")
        ]
        schedule = ingest_functions[0]["Properties"]["Events"]["ScheduledIngestion"]
        assert schedule["Properties"]["Schedule"] == "rate(1 hour)"

    def test_write_template(self, tmp_path):
        """Test writing template to file."""
        config = self._create_config()
        generator = CloudFormationGenerator()
        output_path = tmp_path / "template.yaml"

        generator.write_template(config, output_path)

        assert output_path.exists()
        content = output_path.read_text()
        parsed = yaml.safe_load(content)
        assert "Resources" in parsed

    def test_template_outputs(self):
        """Test that template has expected outputs."""
        config = self._create_config()
        generator = CloudFormationGenerator()
        result = generator.generate(config)
        parsed = yaml.safe_load(result)

        outputs = parsed["Outputs"]
        assert "DocumentBucketName" in outputs
        assert "MetadataTableName" in outputs
        assert "QueryFunctionArn" in outputs
        assert "ApiEndpoint" in outputs
