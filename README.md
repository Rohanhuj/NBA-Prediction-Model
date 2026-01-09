# NBA Prediction Studio

This project is a full NBA prediction experience that I built to combine schedule browsing, model picks, and LLM-backed explanations in one place. The core idea is simple: the schedule stays fast and lightweight, and richer explanations are only generated when I actually open a matchup. That keeps the UI responsive while still letting me dive deep into a single game.


### Matchups Screen
<img width="2546" height="1178" alt="image" src="https://github.com/user-attachments/assets/8afe684b-7eae-40a1-aa83-7b3282358600" />
Browse through matchups throughout the day with a calendar button to flip through the years. Retrieves predictions for 2015-2024 seasons. Matchups per day are printed on screen which are clickable to see model prediction + explanation. 

## Prediction Screen
See the model predictions with LLM explanations and supporting news articles per matchup, with ranked news articles based on info supplied to model and how they back up model prediction
<img width="2544" height="874" alt="image" src="https://github.com/user-attachments/assets/a38824e5-a351-4ffa-9a42-578ce003b739" />
<img width="2559" height="1091" alt="image" src="https://github.com/user-attachments/assets/4e0df257-e189-4807-b691-7399b8d32e57" />
<img width="2549" height="1177" alt="image" src="https://github.com/user-attachments/assets/14f34f7c-a218-4b9f-ac03-af6e59365c95" />
<img width="2543" height="742" alt="image" src="https://github.com/user-attachments/assets/85d7b15f-21ae-44b9-983d-1cf0ea9c4755" />





## Why it is structured this way

- **Fast schedule, deep detail**: The schedule endpoint only returns the day’s matchups, while the prediction endpoint does the heavier work (model + LLM). This avoids slow list pages and makes the detail page the place for richer context.
- **Single source of truth for games**: Every prediction is tied to a `game_id` that comes directly from the daily schedule. This prevents mismatched teams or dates and keeps the UI from guessing payloads.
- **Schema-safe predictions**: The model pipeline normalizes input features so that newer season data (extra columns like `GmSc` or `Unnamed`) does not break the model feature shape.
- **On-demand caching**: I cache a sliding date window on the backend so flipping dates is quick but still avoids hitting the upstream API on every click.

## Components and how they work together

### Backend (FastAPI in `api/`)

- **`/api/matchups`** returns the schedule for a given date with clean, snake_case fields and a normalized tipoff timestamp. This is intentionally lightweight and fast.
- **`/api/predict`** accepts `game_id` + `date`, validates the ID against the cached schedule, runs the model, and only then calls the LangGraph agent to produce an explanation. If the model can’t predict, the LLM call is skipped and the response is clean and UI‑safe.
- **Caching**: the backend prefetches a small window of dates (configurable) and stores them in memory to reduce upstream calls and keep the UI snappy.

### Model + feature pipeline (`model/` + `simple_agent/`)

- The trained model lives in `model/model.pkl` and is reused by both the API and the agent.
- Predictions use the same feature construction logic as the training pipeline, with added schema normalization to strip unexpected columns and enforce a stable feature shape.
- The `simple_agent/` layer wraps predictions with a LangGraph agent that fetches and ranks external news, then synthesizes a unified explanation.

### Frontend (React/Vite in `nba-frontend/`)

- **Home screen**: a calendar‑style schedule view that lets me pick a date and see all matchups without loading heavy prediction data.
- **Matchup detail screen**: a dedicated page for the prediction output, showing the model pick, confidence, the unified explanation, and ranked article summaries.
- **UI clarity**: the explanation is broken into sections (why the pick is supported, evidence summary, combined context, and top articles). Each article is displayed with its own breakdown (player availability, odds, momentum, injuries, sentiment, etc.) to keep it readable.

## Main functionality

- Browse the NBA schedule by date.
- Click a matchup to generate a model pick and an LLM‑based explanation.
- See the explanation split into easy‑to‑scan sections instead of a single wall of text.
- Review the top supporting articles and their summaries in a clean card layout.
- Quickly flip between dates without overwhelming the backend or upstream APIs.

## Why this is useful

This project turns a raw prediction model into something interpretable and usable. Instead of just returning a team label, it links predictions to real‑world context (injuries, betting odds, team form, and analyst sentiment). It’s especially useful if I want to explore **why** a model pick makes sense and compare that against news or market signals.

## What makes it feel like a real product

- A schedule‑first UI that stays fast.
- A detailed, structured explanation experience for each game.
- A layout and visual system that makes the data easy to consume.
- Controlled backend behavior so that errors are readable and never break the UI.

This is the foundation of a full NBA prediction and explanation app, and it is designed so I can keep expanding it (better model inputs, richer UI, deeper news ranking, and more advanced filtering) without rewriting the core flow.
