import os
import asyncio
import json
import requests
import re
from openai import OpenAI

# The grader injects SPACE_URL pointing to the live HF Space
# Fall back to localhost only for local testing
ENV_URL = os.getenv("SPACE_URL", "http://localhost:7860").rstrip("/")

MODEL_CANDIDATES = [
    os.getenv("MODEL_NAME", ""),
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistral",
    "gpt-3.5-turbo",
    "openai/gpt-3.5-turbo",
]

def get_client():
    api_base = os.environ.get("API_BASE_URL")
    api_key  = os.environ.get("API_KEY")
    if not api_base:
        raise RuntimeError("API_BASE_URL environment variable is not set!")
    if not api_key:
        raise RuntimeError("API_KEY environment variable is not set!")
    print(f"[CONFIG] API_BASE_URL={api_base} ENV_URL={ENV_URL}", flush=True)
    return OpenAI(base_url=api_base, api_key=api_key)

def call_llm(client, features) -> float:
    prompt = f"Given these house PCA features: {features}, predict the sale price in USD."
    last_error = None
    for model in MODEL_CANDIDATES:
        if not model:
            continue
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a real estate expert. Return ONLY a raw number (e.g. 250000). No text, no units, no symbols."},
                    {"role": "user",   "content": prompt}
                ],
                max_tokens=32,
                temperature=0.0,
            )
            raw = response.choices[0].message.content or ""
            numeric = re.sub(r"[^\d.]", "", raw)
            print(f"[LLM] model={model} raw={raw!r} parsed={numeric!r}", flush=True)
            return float(numeric) if numeric else 150000.0
        except Exception as e:
            print(f"[LLM] model={model} failed: {e}", flush=True)
            last_error = e

    print(f"[LLM] All models failed: {last_error}", flush=True)
    return 150000.0

def extract_features(obs: dict):
    """Handle both old format (obs.features.pca_features) and new (obs.pca_features)."""
    # New openenv format: pca_features directly on observation
    if "pca_features" in obs:
        return obs["pca_features"]
    # Old format: nested under features
    if "features" in obs and obs["features"]:
        return obs["features"].get("pca_features")
    return []

async def run_inference(task_id: str):
    print(f"[START] task={task_id} env={ENV_URL}", flush=True)
    client = get_client()

    try:
        res = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
        res.raise_for_status()
        body = res.json()
        # create_app wraps in {observation: {...}, reward: ..., done: ...}
        obs = body.get("observation", body)
    except Exception as e:
        print(f"[ERROR] Reset failed for {task_id}: {e}", flush=True)
        return

    total_reward = 0.0
    steps_taken  = 0

    for i in range(1, 6):
        steps_taken = i
        try:
            features = extract_features(obs)
            pred_val = call_llm(client, features)

            action   = {"estimated_value": pred_val}
            step_res = requests.post(f"{ENV_URL}/step",
                                     json={"action": action},   # create_app wraps action in {"action": ...}
                                     timeout=30).json()

            reward = step_res.get("reward") or 0.001
            done   = step_res.get("done", True)
            obs    = step_res.get("observation", obs)

            total_reward += float(reward)
            print(f"[STEP] step={i} action={json.dumps(action)} reward={reward} done={done}", flush=True)

            if done:
                break

        except Exception as e:
            print(f"[ERROR] Step {i} failed: {e}", flush=True)
            break

    print(f"[END] task={task_id} success={total_reward >= 0.8} steps={steps_taken} score={total_reward}", flush=True)

def main():
    for t in ["task_1_easy", "task_2_medium", "task_3_hard"]:
        asyncio.run(run_inference(t))

if __name__ == "__main__":
    main()