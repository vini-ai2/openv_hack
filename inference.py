import os
import asyncio
import json
import requests
import re
from openai import OpenAI

# 1. Configuration - Updated with the new HF Router URL
API_BASE_URL = "https://router.huggingface.co/openai/v1"
API_KEY = os.getenv("HF_TOKEN") 
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3" 
ENV_URL = os.getenv("SPACE_URL", "http://localhost:8000")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

def log_step(step, action, reward, done, error=None):
    print(f"[STEP] step={step} action={json.dumps(action)} reward={reward} done={done} error={error}", flush=True)

async def run_inference(task_id):
    print(f"[START] task={task_id} env=real-estate-valuation model={MODEL_NAME}", flush=True)
    
    try:
        # 1. Reset Environment
        res = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id})
        res.raise_for_status()
        obs = res.json()["observation"]
        
        total_reward = 0
        steps_taken = 0
        
        for i in range(1, 6): # Max 5 steps
            steps_taken = i
            
            # 2. LLM Call with cleaned prompt
            prompt = f"House Data: {obs['features']}. Task: {task_id}. Provide a numeric price estimate."
            
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a real estate expert. Return ONLY a raw number (e.g. 250000). No text."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            raw_content = response.choices[0].message.content
            # Extract only digits and decimals
            numeric_string = re.sub(r"[^\d.]", "", raw_content)
            
            try:
                pred_val = float(numeric_string) if numeric_string else 0.0
            except ValueError:
                pred_val = 0.0

            # 3. Formulate Action
            action = {"estimated_value": pred_val} 
            
            # 4. Step
            step_res = requests.post(f"{ENV_URL}/step", json=action).json()
            reward = step_res.get("reward", 0.0)
            done = step_res.get("done", True)
            
            total_reward += reward
            log_step(step=i, action=action, reward=reward, done=done)
            
            if done:
                break

        success = total_reward >= 0.8
        print(f"[END] success={success} steps={steps_taken} score={total_reward}", flush=True)

    except Exception as e:
        print(f"[DEBUG] Error during inference: {e}")

if __name__ == "__main__":
    tasks = ["task_1_easy", "task_2_medium", "task_3_hard"]
    for t in tasks:
        asyncio.run(run_inference(t))