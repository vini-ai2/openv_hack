# my_env_v4.py
import numpy as np
import pandas as pd
from typing import List, Optional
from pydantic import BaseModel

# -------------------------
# Pydantic Models for OpenEnv
# -------------------------
class MyEnvV4Observation(BaseModel):
    pca_features: List[float]
    true_price: float
    task: str

class MyEnvV4Action(BaseModel):
    predicted_price: float

class MyEnvV4Reward(BaseModel):
    reward: float

# -------------------------
# Hackathon-ready Environment
# -------------------------
class MyEnvV4Env:
    TASKS = ["easy", "medium", "hard"]

    def __init__(self, data: pd.DataFrame):
        """
        data: PCA-reduced dataframe, last column must be 'SalePrice'
        """
        self.data = data.copy()
        self.max_idx = len(self.data) - 1
        self.task_idx = 0  # start with 'easy'
        self.idx = 0
        self._setup_task()

    # -------------------------
    # Setup feature subset per task
    # -------------------------
    def _setup_task(self):
        self.task = MyEnvV4Env.TASKS[self.task_idx]
        if self.task == "easy":
            self.feature_cols = self.data.columns[:3]
        elif self.task == "medium":
            self.feature_cols = self.data.columns[:10]
        else:  # hard
            self.feature_cols = self.data.columns[:-1]

    # -------------------------
    # Switch to next task
    # -------------------------
    def next_task(self):
        self.task_idx = (self.task_idx + 1) % len(MyEnvV4Env.TASKS)
        self.idx = 0
        self._setup_task()

    # -------------------------
    # Reset environment
    # -------------------------
    async def reset(self) -> MyEnvV4Observation:
        self.idx = 0
        row = self.data.iloc[self.idx]
        return MyEnvV4Observation(
            pca_features=row[self.feature_cols].tolist(),
            true_price=float(row["SalePrice"]),
            task=self.task
        )

    # -------------------------
    # Step function
    # -------------------------
    async def step(self, action: MyEnvV4Action):
        if self.idx > self.max_idx:
            # episode done
            done = True
            obs = MyEnvV4Observation(pca_features=[], true_price=0.0, task=self.task)
            reward = 0.001  # strictly > 0
            info = {"error": "Episode finished"}
            return obs, reward, done, info

        row = self.data.iloc[self.idx]
        obs = MyEnvV4Observation(
            pca_features=row[self.feature_cols].tolist(),
            true_price=float(row["SalePrice"]),
            task=self.task
        )

        # Reward: inverse relative error
        raw_reward = 1 - abs(action.predicted_price - obs.true_price) / obs.true_price
        reward = max(0.001, min(0.999, raw_reward))
        done = self.idx == self.max_idx
        info = {}

        self.idx += 1
        return obs, reward, done, info

    # -------------------------
    # Return internal state
    # -------------------------
    def state(self):
        return {"current_index": self.idx, "task": self.task, "task_idx": self.task_idx}

    # -------------------------
    # Grader for any task
    # -------------------------
    @staticmethod
    def grade(predictions: List[float], true_prices: List[float]) -> float:
        rewards = [max(0.001, min(0.999, 1 - abs(p - t) / t)) for p, t in zip(predictions, true_prices)]
        return float(np.mean(rewards))  # normalized 0-1 score

    # -------------------------
    # Load from CSV helper
    # -------------------------
    @classmethod
    def from_csv(cls, csv_path: str):
        df = pd.read_csv(csv_path)
        return cls(df)