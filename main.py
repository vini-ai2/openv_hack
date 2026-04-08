import uvicorn
from fastapi import FastAPI
from models import Action, ResetRequest, StepResponse, Observation

app = FastAPI()

@app.post("/reset")
async def reset(req: ResetRequest):
    # This is where Person 1's data loading logic goes
    return {
        "observation": {
            "task_id": req.task_id,
            "features": {"sqft": 2500, "beds": 3, "year_built": 2010},
            "step_count": 0
        }
    }

@app.post("/step", response_model=StepResponse)
async def step(action: Action):
    # This is where Person 1's grading logic goes
    return StepResponse(
        observation=Observation(task_id="task_1_easy", features={}, step_count=1),
        reward=1.0,
        done=True,
        info={"actual_price": 350000}
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)