"""Tests for CI/CD template generation."""

import yaml
import pytest

from ragforge.core.config import RAGConfig, DataSourceConfig
from ragforge.deployment.cicd import CICDGenerator


class TestCICDGenerator:
    """Tests for CICDGenerator class."""

    def _create_config(self, **kwargs) -> RAGConfig:
        """Create a test config."""
        return RAGConfig(
            project_name=kwargs.get("project_name", "test-project"),
            data_sources=kwargs.get("data_sources", [
                DataSourceConfig(name="docs", type="s3")
            ]),
        )

    def test_generate_returns_valid_yaml(self):
        """Test that generate returns valid YAML."""
        config = self._create_config()
        generator = CICDGenerator()
        result = generator.generate_github_actions(config)

        parsed = yaml.safe_load(result)
        assert parsed is not None
        assert isinstance(parsed, dict)

    def test_workflow_has_name(self):
        """Test that workflow has a name."""
        config = self._create_config()
        generator = CICDGenerator()
        result = generator.generate_github_actions(config)
        parsed = yaml.safe_load(result)

        assert "name" in parsed
        assert "test-project" in parsed["name"]

    def test_workflow_has_trigger(self):
        """Test that workflow has push trigger on main."""
        config = self._create_config()
        generator = CICDGenerator()
        result = generator.generate_github_actions(config)
        parsed = yaml.safe_load(result)

        # YAML 'on' key is parsed as True boolean
        assert True in parsed
        triggers = parsed[True]
        assert "push" in triggers
        assert "main" in triggers["push"]["branches"]

    def test_workflow_has_test_job(self):
        """Test that workflow has a test job."""
        config = self._create_config()
        generator = CICDGenerator()
        result = generator.generate_github_actions(config)
        parsed = yaml.safe_load(result)

        assert "jobs" in parsed
        assert "test" in parsed["jobs"]

        test_job = parsed["jobs"]["test"]
        assert test_job["runs-on"] == "ubuntu-latest"

        step_names = [s.get("name", "") for s in test_job["steps"]]
        assert "Checkout code" in step_names
        assert "Set up Python" in step_names
        assert "Install dependencies" in step_names
        assert "Run tests" in step_names

    def test_workflow_has_deploy_job(self):
        """Test that workflow has a build-and-deploy job."""
        config = self._create_config()
        generator = CICDGenerator()
        result = generator.generate_github_actions(config)
        parsed = yaml.safe_load(result)

        assert "build-and-deploy" in parsed["jobs"]

        deploy_job = parsed["jobs"]["build-and-deploy"]
        assert deploy_job["needs"] == "test"
        assert "github.ref == 'refs/heads/main'" in deploy_job["if"]

    def test_deploy_job_has_aws_steps(self):
        """Test that deploy job has AWS-related steps."""
        config = self._create_config()
        generator = CICDGenerator()
        result = generator.generate_github_actions(config)
        parsed = yaml.safe_load(result)

        deploy_job = parsed["jobs"]["build-and-deploy"]
        step_names = [s.get("name", "") for s in deploy_job["steps"]]

        assert "Configure AWS credentials" in step_names
        assert "Install SAM CLI" in step_names
        assert "Build SAM application" in step_names
        assert "Deploy to AWS" in step_names

    def test_deploy_uses_project_name(self):
        """Test that deploy step uses project name as stack name."""
        config = self._create_config(project_name="my-rag-app")
        generator = CICDGenerator()
        result = generator.generate_github_actions(config)
        parsed = yaml.safe_load(result)

        deploy_job = parsed["jobs"]["build-and-deploy"]
        deploy_step = next(
            s for s in deploy_job["steps"] if s.get("name") == "Deploy to AWS"
        )
        assert "my-rag-app" in deploy_step["run"]

    def test_workflow_has_env_vars(self):
        """Test that workflow has environment variables."""
        config = self._create_config()
        generator = CICDGenerator()
        result = generator.generate_github_actions(config)
        parsed = yaml.safe_load(result)

        assert "env" in parsed
        assert "PYTHON_VERSION" in parsed["env"]
        assert "AWS_REGION" in parsed["env"]

    def test_write_workflow(self, tmp_path):
        """Test writing workflow to file."""
        config = self._create_config()
        generator = CICDGenerator()
        output_path = tmp_path / ".github" / "workflows" / "deploy.yml"

        generator.write_workflow(config, output_path)

        assert output_path.exists()
        content = output_path.read_text()
        parsed = yaml.safe_load(content)
        assert "jobs" in parsed

    def test_test_job_runs_pytest(self):
        """Test that test job runs pytest."""
        config = self._create_config()
        generator = CICDGenerator()
        result = generator.generate_github_actions(config)
        parsed = yaml.safe_load(result)

        test_job = parsed["jobs"]["test"]
        run_steps = [s.get("run", "") for s in test_job["steps"]]
        assert any("pytest" in r for r in run_steps)

    def test_pull_request_trigger(self):
        """Test that workflow triggers on pull requests."""
        config = self._create_config()
        generator = CICDGenerator()
        result = generator.generate_github_actions(config)
        parsed = yaml.safe_load(result)

        triggers = parsed[True]
        assert "pull_request" in triggers
        assert "main" in triggers["pull_request"]["branches"]
