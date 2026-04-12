import os
import asyncio
import json
import requests
import re
from openai import OpenAI

ENV_URL = os.getenv("SPACE_URL", "http://localhost:7860")

# LiteLLM proxies commonly alias models — try common names in order
MODEL_CANDIDATES = [
    os.getenv("MODEL_NAME", ""),          # honour explicit override if set
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
    print(f"[CONFIG] API_BASE_URL={api_base}", flush=True)
    return OpenAI(base_url=api_base, api_key=api_key)

def call_llm(client, prompt: str) -> float:
    """Try each model candidate until one works; return predicted price."""
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

    # All candidates failed — use a reasonable fallback so the episode still runs
    print(f"[LLM] All models failed, using fallback price. Last error: {last_error}", flush=True)
    return 150000.0

def log_step(step, action, reward, done, error=None):
    print(f"[STEP] step={step} action={json.dumps(action)} reward={reward} done={done} error={error}", flush=True)

async def run_inference(task_id: str):
    print(f"[START] task={task_id}", flush=True)
    client = get_client()

    try:
        # 1. Reset
        res = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
        res.raise_for_status()
        obs = res.json()["observation"]
    except Exception as e:
        print(f"[ERROR] Reset failed for {task_id}: {e}", flush=True)
        return  # skip this task, don't crash the whole run

    total_reward = 0.0
    steps_taken  = 0

    for i in range(1, 6):
        steps_taken = i
        try:
            prompt = f"Given these house PCA features: {obs['features'].get('pca_features')}, what is the price?"
            pred_val = call_llm(client, prompt)

            action   = {"estimated_value": pred_val}
            step_res = requests.post(f"{ENV_URL}/step", json=action, timeout=30).json()
            reward   = step_res.get("reward", 0.001)
            done     = step_res.get("done", True)
            obs      = step_res.get("observation", obs)

            total_reward += reward
            log_step(step=i, action=action, reward=reward, done=done)

            if done:
                break

        except Exception as e:
            print(f"[ERROR] Step {i} failed: {e}", flush=True)
            break  # stop this episode but don't crash

    print(f"[END] task={task_id} success={total_reward >= 0.8} steps={steps_taken} score={total_reward}", flush=True)

def main():
    for t in ["task_1_easy", "task_2_medium", "task_3_hard"]:
        asyncio.run(run_inference(t))

if __name__ == "__main__":
    main()