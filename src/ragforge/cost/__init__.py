"""Cost optimization module for RAGForge.

Provides cost tracking, budget enforcement, model routing,
tiered storage, and embedding quantization.
"""

from ragforge.cost.enforcer import BudgetEnforcer, BudgetExceededError
from ragforge.cost.model_router import EmbeddingModelRouter
from ragforge.cost.quantization import EmbeddingQuantizer
from ragforge.cost.tiered_storage import TieredStorageManager
from ragforge.cost.tracker import CostTracker

__all__ = [
    "CostTracker",
    "BudgetEnforcer",
    "BudgetExceededError",
    "EmbeddingModelRouter",
    "TieredStorageManager",
    "EmbeddingQuantizer",
]
