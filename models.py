from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

# 1. Action: What your Agent (Person B) outputs
class Action(BaseModel):
    estimated_value: float = Field(..., description="The agent's predicted price for the property")

# 2. Observation: What the Environment (Person A) provides
class Observation(BaseModel):
    task_id: str = Field(..., description="The ID of the current task (easy/medium/hard)")
    features: Dict[str, Any] = Field(..., description="Dictionary of property features")
    step_count: int = Field(default=0)

# 3. The response structure for the /step endpoint
class StepResponse(BaseModel):
    observation: Observation
    # gt/lt = strictly greater than / less than (not equal to 0.0 or 1.0)
    reward: float = Field(..., gt=0.0, lt=1.0, description="Reward score strictly between 0 and 1")
    done: bool = Field(..., description="Whether the episode is finished")
    info: Optional[Dict[str, Any]] = Field(default_factory=dict)

# 4. The request structure for the /reset endpoint
#    task_id is OPTIONAL with a default — grader may POST with no body at all
class ResetRequest(BaseModel):
    task_id: Optional[str] = Field(default="task_1_easy")