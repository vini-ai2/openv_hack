# openv_hack
Building a complete, real-world OpenEnv environment that an AI agent can learn from through the standard  step() / reset() / state()  API.
# Real Estate Valuation Agent (OpenEnv)

## Environment Description
This environment simulates a real-world property appraisal task using the Ames Housing Dataset. Agents must analyze property features to provide accurate valuations.

## Action Space
- `estimated_value` (float): The agent's predicted market price in USD.

## Observation Space
- `task_id` (string): Current difficulty level.
- `features` (dict): Property details including:
    - `sqft`: Living area square footage.
    - `beds`: Number of bedrooms.
    - `year_built`: Original construction date.

## Tasks
1. **Task 1 (Easy)**: Price Bracket Classification.
2. **Task 2 (Medium)**: Direct Point Estimate Valuation.
3. **Task 3 (Hard)**: Comparative Market Analysis reconciliation.