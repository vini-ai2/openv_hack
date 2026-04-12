import pandas as pd
from typing import Optional
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import PropertyAction, PropertyObservation
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from models import PropertyAction, PropertyObservation

TASK_ID_MAP = {
    "task_1_easy":   "easy",
    "task_2_medium": "medium",
    "task_3_hard":   "hard",
}
TASKS = ["easy", "medium", "hard"]
TASK_ID_REVERSE = {v: k for k, v in TASK_ID_MAP.items()}

EPSILON = 1e-6  # distance from 0/1 boundary

def _clamp(r: float) -> float:
    """Strictly between 0 and 1 — never exactly 0.0 or 1.0."""
    return float(max(EPSILON, min(1.0 - EPSILON, r)))

def _load_data():
    for path in ["reduced_dataset.csv", "/app/reduced_dataset.csv"]:
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            continue
    raise FileNotFoundError("reduced_dataset.csv not found")


class PropertyValuationEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self):
        super().__init__()
        self.data = _load_data()
        self.max_idx = len(self.data) - 1
        self.task_idx = 0
        self.idx = 0
        self._state = State(episode_id="property-env", step_count=0)
        self._setup_task()

    def _setup_task(self):
        task = TASKS[self.task_idx]
        if task == "easy":
            self.feature_cols = self.data.columns[:3]
        elif task == "medium":
            self.feature_cols = self.data.columns[:10]
        else:
            self.feature_cols = self.data.columns[:-1]

    def reset(self, seed=None, episode_id=None, **kwargs) -> PropertyObservation:
        task_id = kwargs.get("task_id") or episode_id or "task_1_easy"
        internal = TASK_ID_MAP.get(task_id, "easy")
        self.task_idx = TASKS.index(internal)
        self._setup_task()
        self.idx = 0
        self._state = State(episode_id=task_id, step_count=0)

        row = self.data.iloc[self.idx]
        return PropertyObservation(
            task_id=task_id,
            pca_features=row[self.feature_cols].tolist(),
            step_count=0,
            done=False,
            reward=0.5,  # neutral non-boundary value — never None/0.0/1.0
        )

    def step(self, action: PropertyAction, timeout_s=None, **kwargs) -> PropertyObservation:
        if self.idx > self.max_idx:
            return PropertyObservation(
                task_id=TASK_ID_REVERSE.get(TASKS[self.task_idx], "task_1_easy"),
                pca_features=[],
                step_count=self.idx,
                done=True,
                reward=0.5,
            )

        row = self.data.iloc[self.idx]
        true_price = float(row["SalePrice"])
        pred = float(action.estimated_value)

        # Avoid division by zero
        if true_price == 0:
            raw = 0.5
        else:
            raw = 1.0 - abs(pred - true_price) / true_price

        reward = _clamp(raw)

        done = self.idx == self.max_idx
        self.idx += 1
        self._state.step_count = self.idx

        return PropertyObservation(
            task_id=TASK_ID_REVERSE.get(TASKS[self.task_idx], "task_1_easy"),
            pca_features=row[self.feature_cols].tolist(),
            step_count=self.idx,
            done=done,
            reward=reward,
        )

    @property
    def state(self) -> State:
        return self._state