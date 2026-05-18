"""CI/CD template generator for RAGForge deployments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from ragforge.core.config import RAGConfig


class CICDGenerator:
    """Generates GitHub Actions workflow YAML for CI/CD pipelines.

    Workflow: on push to main → install deps → run tests → build → deploy
    """

    def generate_github_actions(self, config: RAGConfig) -> str:
        """Generate a GitHub Actions workflow YAML.

        Args:
            config: RAGConfig instance with project settings.

        Returns:
            YAML string of the GitHub Actions workflow.
        """
        workflow = self._build_workflow(config)
        return yaml.dump(workflow, default_flow_style=False, sort_keys=False)

    def write_workflow(self, config: RAGConfig, output_path: str | Path) -> None:
        """Generate and write a GitHub Actions workflow to a file.

        Args:
            config: RAGConfig instance with project settings.
            output_path: Path to write the workflow file.
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.generate_github_actions(config))

    def _build_workflow(self, config: RAGConfig) -> Dict[str, Any]:
        """Build the GitHub Actions workflow dictionary."""
        project_name = config.project_name

        workflow: Dict[str, Any] = {
            "name": f"Deploy {project_name}",
            True: {  # 'on' key in YAML
                "push": {
                    "branches": ["main"],
                },
                "pull_request": {
                    "branches": ["main"],
                },
            },
            "env": {
                "PYTHON_VERSION": "3.11",
                "AWS_REGION": "us-east-1",
            },
            "jobs": {
                "test": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {
                            "name": "Checkout code",
                            "uses": "actions/checkout@v4",
                        },
                        {
                            "name": "Set up Python",
                            "uses": "actions/setup-python@v5",
                            "with": {"python-version": "${{ env.PYTHON_VERSION }}"},
                        },
                        {
                            "name": "Install dependencies",
                            "run": "pip install -e '.[dev]'",
                        },
                        {
                            "name": "Run tests",
                            "run": "pytest tests/ --tb=short -q",
                        },
                    ],
                },
                "build-and-deploy": {
                    "needs": "test",
                    "runs-on": "ubuntu-latest",
                    "if": "github.ref == 'refs/heads/main' && github.event_name == 'push'",
                    "steps": [
                        {
                            "name": "Checkout code",
                            "uses": "actions/checkout@v4",
                        },
                        {
                            "name": "Set up Python",
                            "uses": "actions/setup-python@v5",
                            "with": {"python-version": "${{ env.PYTHON_VERSION }}"},
                        },
                        {
                            "name": "Install dependencies",
                            "run": "pip install -e '.[all]'",
                        },
                        {
                            "name": "Configure AWS credentials",
                            "uses": "aws-actions/configure-aws-credentials@v4",
                            "with": {
                                "aws-access-key-id": "${{ secrets.AWS_ACCESS_KEY_ID }}",
                                "aws-secret-access-key": "${{ secrets.AWS_SECRET_ACCESS_KEY }}",
                                "aws-region": "${{ env.AWS_REGION }}",
                            },
                        },
                        {
                            "name": "Install SAM CLI",
                            "uses": "aws-actions/setup-sam@v2",
                        },
                        {
                            "name": "Build SAM application",
                            "run": "sam build",
                        },
                        {
                            "name": "Deploy to AWS",
                            "run": (
                                f"sam deploy --no-confirm-changeset --no-fail-on-empty-changeset "
                                f"--stack-name {project_name.replace(' ', '-').lower()} "
                                f"--capabilities CAPABILITY_IAM"
                            ),
                        },
                    ],
                },
            },
        }

        return workflow
