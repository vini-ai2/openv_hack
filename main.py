import uvicorn
import pandas as pd
from fastapi import FastAPI, HTTPException
from models import Action, ResetRequest, StepResponse, Observation
from my_env_v4 import MyEnvV4Env, MyEnvV4Action 

app = FastAPI()

# 1. Load the environment
try:
    # Ensure this file exists in your Space root!
    env = MyEnvV4Env.from_csv("reduced_dataset.csv") 
except Exception as e:
    print(f"CRITICAL: Dataset load failed: {e}")
    env = None

@app.get("/")
async def health_check():
    return {"status": "running", "dataset_loaded": env is not None}

@app.post("/reset")
async def reset(req: ResetRequest):
    if not env:
        raise HTTPException(status_code=500, detail="Environment not initialized")
    
    # Person 1's reset returns MyEnvV4Observation
    raw_obs = await env.reset() 
    
    # Wrap it in the top-level "observation" key the grader expects
    return {
        "observation": {
            "task_id": raw_obs.task,
            "features": {
                "pca_features": raw_obs.pca_features,
                "true_price": raw_obs.true_price
            },
            "step_count": 0
        }
    }

@app.post("/step", response_model=StepResponse)
async def step(action: Action):
    if not env:
        raise HTTPException(status_code=500, detail="Environment not initialized")
    
    # 2. Map Action (estimated_value) to Person 1's predicted_price
    v4_action = MyEnvV4Action(predicted_price=action.estimated_value)
    obs, reward, done, info = await env.step(v4_action)
    
    # 3. Return the exact StepResponse format
    return StepResponse(
        observation=Observation(
            task_id=obs.task, 
            features={"pca_features": obs.pca_features}, 
            step_count=env.idx
        ),
        reward=reward,
        done=done,
        info=info
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)