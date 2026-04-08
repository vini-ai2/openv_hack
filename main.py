import uvicorn
import pandas as pd
from fastapi import FastAPI, HTTPException
from models import Action, ResetRequest, StepResponse, Observation
from my_env_v4 import MyEnvV4Env, MyEnvV4Action

app = FastAPI()

# Map openenv.yaml task IDs → internal task names
TASK_ID_MAP = {
    "task_1_easy": "easy",
    "task_2_medium": "medium",
    "task_3_hard": "hard",
}

# 1. Load the environment
try:
    env = MyEnvV4Env.from_csv("reduced_dataset.csv")
except Exception as e:
    print(f"CRITICAL: Dataset load failed: {e}")
    env = None

@app.get("/")
async def health_check():
    return {"status": "running", "dataset_loaded": env is not None}

@app.post("/reset")
async def reset(req: ResetRequest):
    if env is None:
        raise HTTPException(status_code=500, detail="Environment data not loaded")

    try:
        # Map the grader's task_id (e.g. "task_1_easy") to internal name ("easy")
        internal_task = TASK_ID_MAP.get(req.task_id)
        if internal_task is None:
            raise HTTPException(status_code=400, detail=f"Unknown task_id: {req.task_id}")

        # Set the correct task on the env before resetting
        env.task_idx = list(MyEnvV4Env.TASKS).index(internal_task)
        env._setup_task()

        # Reset and get the first observation
        raw_v4_obs = await env.reset()

        # Return observation WITHOUT true_price (don't leak the answer to the agent)
        formatted_obs = Observation(
            task_id=req.task_id,
            features={
                "pca_features": raw_v4_obs.pca_features,
            },
            step_count=0
        )

        return {"observation": formatted_obs.dict()}

    except HTTPException:
        raise
    except Exception as e:
        print(f"RESET ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step", response_model=StepResponse)
async def step(action: Action):
    if not env:
        raise HTTPException(status_code=500, detail="Environment not initialized")

    v4_action = MyEnvV4Action(predicted_price=action.estimated_value)
    obs, reward, done, info = await env.step(v4_action)

    return StepResponse(
        observation=Observation(
            task_id=MyEnvV4Env.TASKS[env.task_idx],
            features={"pca_features": obs.pca_features},
            step_count=env.idx
        ),
        reward=reward,
        done=done,
        info=info
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)