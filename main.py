# main.py
import uvicorn
import pandas as pd
from fastapi import FastAPI, HTTPException
from typing import Optional
from models import Action, ResetRequest, StepResponse, Observation
from my_env_v4 import MyEnvV4Env, MyEnvV4Action

app = FastAPI()

TASK_ID_MAP = {
    "task_1_easy": "easy",
    "task_2_medium": "medium",
    "task_3_hard": "hard",
}

try:
    env = MyEnvV4Env.from_csv("reduced_dataset.csv")
except Exception as e:
    env = None

@app.post("/reset")
async def reset(req: Optional[ResetRequest] = None):
    if env is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")

    try:
        # Preserve the EXACT string sent by the validator
        ext_task_id = (req.task_id if req and req.task_id else "task_1_easy")
        env.active_external_id = ext_task_id
        
        internal_task = TASK_ID_MAP.get(ext_task_id, "easy")
        env.task_idx = list(MyEnvV4Env.TASKS).index(internal_task)
        env._setup_task()

        raw_v4_obs = await env.reset()
        return {
            "observation": {
                "task_id": ext_task_id, # Must match exactly
                "features": {"pca_features": raw_v4_obs.pca_features},
                "step_count": 0
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step", response_model=StepResponse)
async def step(action: Action):
    v4_action = MyEnvV4Action(predicted_price=action.estimated_value)
    obs, reward, done, info = await env.step(v4_action)

    # env.step already clamps the reward to (0.001, 0.999)
    return StepResponse(
        observation=Observation(
            task_id=env.active_external_id, # Persistent ID
            features={"pca_features": obs.pca_features},
            step_count=env.idx
        ),
        reward=reward, 
        done=done,
        info=info
    )