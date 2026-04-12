from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class Action(BaseModel):
    estimated_value: float = Field(..., description="The agent's predicted price for the property")

class Observation(BaseModel):
    task_id: str = Field(..., description="The ID of the current task (easy/medium/hard)")
    features: Dict[str, Any] = Field(..., description="Dictionary of property features")
    step_count: int = Field(default=0)

class StepResponse(BaseModel):
    observation: Observation
    # ge/le instead of gt/lt — clamp_reward in main.py guarantees strictly (0,1)
    # but using ge/le avoids Pydantic throwing a 500 if something slips through
    reward: float = Field(..., ge=0.001, le=0.999, description="Reward score strictly between 0 and 1")
    done: bool = Field(..., description="Whether the episode is finished")
    info: Optional[Dict[str, Any]] = Field(default_factory=dict)

class ResetRequest(BaseModel):
    task_id: Optional[str] = Field(default="task_1_easy")