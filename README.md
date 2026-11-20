# Ghanex Tech GenAI Internship Assessment - Arnav Mishra

My solutions to the three tasks in the Ghanex Tech GenAI & Open-Source Developer Intern assessment. The original brief is in [ASSESSMENT.md](ASSESSMENT.md) and the task files next to it.

| Task | Folder | Live app |
|---|---|---|
| 1. Document assistant (RAG) | [task1-rag](task1-rag) | LIVE_URL_TASK1 |
| 2. AI agent with tool calling | [task2-agent](task2-agent) | LIVE_URL_TASK2 |
| 3. GitHub repository analyzer | [task3-github-analyzer](task3-github-analyzer) | LIVE_URL_TASK3 |

Open-source contribution: CONTRIBUTION_URL

Each folder is a separate Streamlit app with its own README (architecture, design choices, setup, tests, limits), `requirements.txt` and `.env.example`.

## Stack

- Python 3.11, Streamlit for all three UIs
- LLM: Groq free tier, model `openai/gpt-oss-20b`, called through the OpenAI-compatible API, so any compatible provider (Gemini, OpenRouter, a local Ollama server) can be used by changing two environment variables
- Task 1: pypdf, all-MiniLM-L6-v2 embeddings through fastembed (ONNX, no PyTorch), FAISS
- Task 2: native tool calling, Open-Meteo, Wikipedia and Frankfurter APIs (all free, no keys)
- Task 3: GitHub REST API with `requests`, pandas for the charts

No LangChain or LlamaIndex: the pipelines are short enough to write directly, which keeps every step visible and easy to test.

## Running any app locally

```bash
cd task1-rag                     # or task2-agent, task3-github-analyzer
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env             # add a free Groq key from https://console.groq.com/keys
streamlit run app.py
pytest
```

## Tests

Every project has offline tests (no API key or network needed): the LLM and HTTP calls are replaced with fakes. GitHub Actions runs all three suites on every push (`.github/workflows/tests.yml`).

## Keys and cost

No secrets are in this repository. Keys go in a local `.env` file (git-ignored) or in the hosting platform's secrets. The deployed apps use a Groq free-tier key, which has no billing attached, so heavy traffic can only hit rate limits, not create costs. Each app also limits how many LLM calls one browser session can make and lets visitors paste their own key.
