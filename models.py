from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class Action(BaseModel):
    estimated_value: float = Field(..., description="The agent's predicted price for the property")

class Observation(BaseModel):
    task_id: str = Field(..., description="The ID of the current task (easy/medium/hard)")
    features: Dict[str, Any] = Field(..., description="Dictionary of property features")
    step_count: int = Field(default=0)

class StepResponse(BaseModel):
    """Aligned with openenv StepResponse — no reward bounds so Pydantic never rejects."""
    observation: Observation
    reward: Optional[float] = Field(default=None, description="Reward signal from the action")
    done: bool = Field(default=False)
    info: Optional[Dict[str, Any]] = Field(default_factory=dict)

class ResetRequest(BaseModel):
    task_id: Optional[str] = Field(default="task_1_easy")