# Task 3: GitHub Repo Analyzer

Paste any public GitHub repository and get a dashboard: key numbers, a language chart, health checks, and an AI review with a summary, architecture notes, strengths, at least three recommendations and ideas for new contributors. The review can be downloaded as Markdown.

Live app: LIVE_URL_TASK3

## How it works

```
"pallets/flask" or a GitHub URL
   |
   v
parse_repo()          accepts owner/repo, https URLs (any page in the repo), SSH URLs
   |
   v
GitHubClient.fetch()  6 to 9 REST API calls:
   |                    /repos/{o}/{r}               metadata
   |                    /repos/{o}/{r}/languages     bytes per language
   |                    /repos/{o}/{r}/readme        README
   |                    /repos/{o}/{r}/git/trees/... full file list in one call
   |                    /repos/{o}/{r}/contents/...  up to 3 package files
   |                    /repos/{o}/{r}/issues        open "good first issue" issues
   v
compute_metrics()     plain code, no AI: stars, forks, days since last push,
   |                  language %, and six health checks (README, license,
   |                  contributing guide, CI config, tests, pushed in last 180 days)
   v
analyze()             one LLM call in JSON mode. The prompt holds the metrics,
   |                  the file tree, package files, issue titles and the README.
   |                  The reply is checked: valid JSON, all keys present,
   |                  at least 3 recommendations. One retry if not.
   v
Streamlit dashboard + Markdown download
```

Files:

```
app.py                     Streamlit dashboard
analyzer/github_client.py  URL parsing, REST calls, error handling
analyzer/metrics.py        health checks and numbers computed in code
analyzer/llm_analysis.py   prompt, JSON validation, retry
analyzer/llm.py            OpenAI-compatible client (Groq by default)
analyzer/report.py         Markdown export
```

## Design choices

Facts from code, opinions from the LLM. Whether a repo has tests, CI or a license is checked by code from the real file list, then given to the model as facts. The model is told to trust them and to say when something cannot be known. This stops it from inventing things like "add CI" for a repo that already has GitHub Actions.

One recursive tree call. `git/trees/{branch}?recursive=1` returns every file path in one request, instead of walking folders one call at a time. That matters because unauthenticated users only get 60 calls per hour. For microsoft/vscode (21,575 files) the whole fetch took 3.3 seconds in my test.

Bounded prompt. The README is cut to 5,000 characters, the tree to the 150 shallowest paths (so the model sees the overall layout first), and package files to 1,200 characters each. The largest prompt I measured (microsoft/vscode) was about 14,800 characters, roughly 3,700 tokens, which fits the free tier's 8,000 tokens per minute.

JSON mode plus validation. The model is asked for fixed keys and the response is parsed and checked in `parse_analysis()`. Unknown priority values are cleaned to "medium". Broken output gets one retry, then a clear error, and the facts part of the page still shows.

Caching. Snapshots and reviews are cached for one hour and shared between visitors, keyed by repo name (and last push time for the review). Analysing the same repo twice costs no extra GitHub calls or LLM tokens. The token is never part of the cache key.

## Error handling

| Situation | What the user sees |
|---|---|
| Not a GitHub repo string | "That does not look like a GitHub repository..." |
| Repo does not exist or is private | "Repository not found. Check the name, or it may be private." |
| Rate limit hit | Reset time in UTC, plus a tip to add a token |
| Bad token | "GitHub rejected the access token..." |
| Empty repo, no README, no package files | Handled; those parts are shown as empty |
| GitHub down or slow | Network or server error message |
| LLM fails or gives bad JSON | Error box; metrics and charts still shown |

## Run locally

Needs Python 3.10 or newer.

```bash
cd task3-github-analyzer
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env             # put your free Groq key in .env (GitHub token optional)
streamlit run app.py
```

Without a GitHub token you get 60 API calls per hour (about 7 to 10 new repos). Any token, for example a fine-grained token with read-only access to public repositories, raises this to 5,000.

## Tests

```bash
cd task3-github-analyzer
pytest
```

All tests are offline. A fake HTTP session plays the GitHub API and a fake LLM returns fixed JSON. They cover URL parsing (9 valid and 8 invalid formats), fetching, empty repos, every API error type, the health checks, prompt contents and size limits, JSON validation and retry, the Markdown report, and the dashboard itself.

## Deploying (Streamlit Community Cloud)

New app, pick this repo, main file `task3-github-analyzer/app.py`, Python 3.11. In Advanced settings, Secrets:

```toml
GROQ_API_KEY = "your-key"
GITHUB_TOKEN = "optional-read-only-token"
```

All visitors of a hosted app share one IP address, so a server-side GitHub token is strongly recommended there.

## Limits and next steps

- The analysis reads metadata, structure and the README, not the source code itself, so it cannot judge code quality line by line. A next step would be to sample a few core source files into the prompt.
- Health checks are based on file names. A `tests/` folder counts as tests even if it is nearly empty.
- `open_issues_count` from GitHub includes open pull requests; the dashboard labels it that way.
- Groq free tier: 30 requests per minute, 1,000 per day, 8,000 tokens per minute for openai/gpt-oss-20b. Each session gets 10 AI reviews on the shared key.
