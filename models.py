from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import List, Optional, Dict, Any


class PropertyAction(Action):
    """Agent's predicted price for the property."""
    estimated_value: float = Field(..., description="Predicted price")


class PropertyObservation(Observation):
    """What the environment returns to the agent."""
    task_id: str = Field(default="task_1_easy")
    pca_features: List[float] = Field(default_factory=list)
    step_count: int = Field(default=0)