"""Access control for per-source, per-tenant document access policies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ragforge.core.models import QueryResult


class AccessController:
    """Manages per-tenant access control policies for document sources.

    Policies are stored as a simple allow-list per tenant in a JSON file.
    Each tenant has a list of source names they are allowed to access.
    """

    def __init__(self, policy_path: str | Path = "ragforge_policies.json"):
        """Initialize the access controller.

        Args:
            policy_path: Path to the JSON file storing access policies.
        """
        self._policy_path = Path(policy_path)
        self._policies: Dict[str, List[str]] = {}
        self._load_policies()

    def _load_policies(self) -> None:
        """Load policies from the JSON file."""
        if self._policy_path.exists():
            try:
                data = json.loads(self._policy_path.read_text(encoding="utf-8"))
                self._policies = data.get("policies", {})
            except (json.JSONDecodeError, IOError):
                self._policies = {}

    def _save_policies(self) -> None:
        """Save policies to the JSON file."""
        self._policy_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"policies": self._policies}
        self._policy_path.write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )

    def add_policy(self, tenant_id: str, sources_allowed: List[str]) -> None:
        """Add or update an access policy for a tenant.

        Args:
            tenant_id: Unique identifier for the tenant.
            sources_allowed: List of source names the tenant can access.
        """
        self._policies[tenant_id] = sources_allowed
        self._save_policies()

    def remove_policy(self, tenant_id: str) -> None:
        """Remove an access policy for a tenant.

        Args:
            tenant_id: Unique identifier for the tenant.
        """
        self._policies.pop(tenant_id, None)
        self._save_policies()

    def check_access(self, tenant_id: str, source_name: str) -> bool:
        """Check if a tenant has access to a specific source.

        Args:
            tenant_id: Unique identifier for the tenant.
            source_name: Name of the data source to check.

        Returns:
            True if the tenant has access, False otherwise.
            Returns True if no policy exists for the tenant (open access).
        """
        if tenant_id not in self._policies:
            # No policy means open access
            return True

        allowed_sources = self._policies[tenant_id]

        # Wildcard access
        if "*" in allowed_sources:
            return True

        return source_name in allowed_sources

    def filter_results(
        self, results: List[QueryResult], tenant_id: str
    ) -> List[QueryResult]:
        """Filter query results based on tenant access policies.

        Args:
            results: List of QueryResult objects to filter.
            tenant_id: Unique identifier for the tenant.

        Returns:
            Filtered list containing only results the tenant can access.
        """
        if tenant_id not in self._policies:
            # No policy means open access
            return results

        return [
            result for result in results
            if self.check_access(tenant_id, result.source)
        ]

    def get_policy(self, tenant_id: str) -> Optional[List[str]]:
        """Get the access policy for a tenant.

        Args:
            tenant_id: Unique identifier for the tenant.

        Returns:
            List of allowed sources, or None if no policy exists.
        """
        return self._policies.get(tenant_id)

    def list_tenants(self) -> List[str]:
        """List all tenants with access policies.

        Returns:
            List of tenant IDs.
        """
        return list(self._policies.keys())
