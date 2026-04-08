from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ResetRequest(BaseModel):
    task_id: str

@app.post("/reset")
def reset(req: ResetRequest):
    # Match the schema you sent Person 1
    return {
        "observation": {
            "id": "house_123",
            "sqft": 2500,
            "beds": 3,
            "baths": 2,
            "neighborhood": "Ames_Main"
        }
    }

@app.post("/step")
def step(action: dict):
    # Simulated grader logic
    return {
        "observation": {"status": "complete"},
        "reward": 1.0,
        "done": True,
        "info": {}
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)