"""CloudFormation/SAM template generator for RAGForge deployments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from ragforge.core.config import RAGConfig


class CloudFormationGenerator:
    """Generates SAM/CloudFormation templates for deploying RAG pipelines.

    Generates resources:
    - Lambda function (query endpoint)
    - S3 bucket (document storage)
    - DynamoDB table (metadata)
    - EventBridge rule (scheduled ingestion)
    - API Gateway (REST endpoint)
    """

    def generate(self, config: RAGConfig) -> str:
        """Generate a SAM/CloudFormation template from RAGForge config.

        Args:
            config: RAGConfig instance with project settings.

        Returns:
            YAML string of the CloudFormation template.
        """
        template = self._build_template(config)
        return yaml.dump(template, default_flow_style=False, sort_keys=False)

    def write_template(self, config: RAGConfig, output_path: str | Path) -> None:
        """Generate and write a CloudFormation template to a file.

        Args:
            config: RAGConfig instance with project settings.
            output_path: Path to write the template file.
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.generate(config))

    def _build_template(self, config: RAGConfig) -> Dict[str, Any]:
        """Build the CloudFormation template dictionary."""
        project_name = config.project_name.replace(" ", "-").lower()
        safe_name = project_name.replace("-", "").title().replace(" ", "")

        template: Dict[str, Any] = {
            "AWSTemplateFormatVersion": "2010-09-09",
            "Transform": "AWS::Serverless-2016-10-31",
            "Description": f"RAGForge deployment for {config.project_name}",
            "Globals": {
                "Function": {
                    "Timeout": int(config.query.timeout_ms / 1000) or 3,
                    "Runtime": "python3.11",
                    "MemorySize": 512,
                }
            },
            "Resources": {},
            "Outputs": {},
        }

        resources = template["Resources"]
        outputs = template["Outputs"]

        # S3 Bucket for document storage
        bucket_logical_id = f"{safe_name}DocumentBucket"
        resources[bucket_logical_id] = {
            "Type": "AWS::S3::Bucket",
            "Properties": {
                "BucketName": f"{project_name}-documents",
                "VersioningConfiguration": {"Status": "Enabled"},
            },
        }
        outputs["DocumentBucketName"] = {
            "Description": "S3 bucket for document storage",
            "Value": {"Ref": bucket_logical_id},
        }

        # DynamoDB table for metadata
        table_logical_id = f"{safe_name}MetadataTable"
        resources[table_logical_id] = {
            "Type": "AWS::DynamoDB::Table",
            "Properties": {
                "TableName": f"{project_name}-metadata",
                "BillingMode": "PAY_PER_REQUEST",
                "AttributeDefinitions": [
                    {"AttributeName": "pk", "AttributeType": "S"},
                    {"AttributeName": "sk", "AttributeType": "S"},
                ],
                "KeySchema": [
                    {"AttributeName": "pk", "KeyType": "HASH"},
                    {"AttributeName": "sk", "KeyType": "RANGE"},
                ],
            },
        }
        outputs["MetadataTableName"] = {
            "Description": "DynamoDB table for metadata",
            "Value": {"Ref": table_logical_id},
        }

        # Lambda function for query endpoint
        function_logical_id = f"{safe_name}QueryFunction"
        resources[function_logical_id] = {
            "Type": "AWS::Serverless::Function",
            "Properties": {
                "FunctionName": f"{project_name}-query",
                "Handler": "handler.query_handler",
                "CodeUri": "./src/",
                "Description": f"RAGForge query endpoint for {config.project_name}",
                "Environment": {
                    "Variables": {
                        "PROJECT_NAME": config.project_name,
                        "EMBEDDING_MODEL": config.embedding.model,
                        "EMBEDDING_DIMENSIONS": str(config.embedding.dimensions),
                        "RETRIEVAL_METHOD": config.retrieval.method,
                        "RETRIEVAL_TOP_K": str(config.retrieval.top_k),
                        "STORAGE_PROVIDER": config.storage.provider,
                        "INDEX_NAME": config.storage.index_name,
                    }
                },
                "Events": {
                    "QueryApi": {
                        "Type": "Api",
                        "Properties": {
                            "Path": "/query",
                            "Method": "post",
                        },
                    }
                },
                "Policies": [
                    "DynamoDBReadPolicy",
                    {"S3ReadPolicy": {"BucketName": f"{project_name}-documents"}},
                ],
            },
        }
        outputs["QueryFunctionArn"] = {
            "Description": "Query Lambda function ARN",
            "Value": {"Fn::GetAtt": [function_logical_id, "Arn"]},
        }

        # API Gateway output
        outputs["ApiEndpoint"] = {
            "Description": "API Gateway endpoint URL",
            "Value": {
                "Fn::Sub": "https://${ServerlessRestApi}.execute-api.${AWS::Region}.amazonaws.com/Prod/query"
            },
        }

        # EventBridge rule for scheduled ingestion
        schedule_logical_id = f"{safe_name}IngestionSchedule"
        # Determine schedule from first data source update_frequency
        cron_expr = "rate(1 day)"
        if config.data_sources:
            freq = config.data_sources[0].update_frequency
            if freq == "hourly":
                cron_expr = "rate(1 hour)"
            elif freq == "weekly":
                cron_expr = "rate(7 days)"
            elif freq == "daily":
                cron_expr = "rate(1 day)"

        ingestion_function_id = f"{safe_name}IngestionFunction"
        resources[ingestion_function_id] = {
            "Type": "AWS::Serverless::Function",
            "Properties": {
                "FunctionName": f"{project_name}-ingest",
                "Handler": "handler.ingest_handler",
                "CodeUri": "./src/",
                "Description": f"RAGForge scheduled ingestion for {config.project_name}",
                "Timeout": 900,
                "MemorySize": 1024,
                "Environment": {
                    "Variables": {
                        "PROJECT_NAME": config.project_name,
                        "DOCUMENT_BUCKET": f"{project_name}-documents",
                    }
                },
                "Events": {
                    "ScheduledIngestion": {
                        "Type": "Schedule",
                        "Properties": {
                            "Schedule": cron_expr,
                            "Description": "Scheduled document ingestion",
                            "Enabled": True,
                        },
                    }
                },
                "Policies": [
                    "DynamoDBCrudPolicy",
                    {"S3ReadPolicy": {"BucketName": f"{project_name}-documents"}},
                ],
            },
        }

        return template
