# Task 2: Tool-calling agent

A chat assistant that decides by itself when to use a tool. It can do exact maths, get live weather, look things up on Wikipedia and convert currencies. For greetings and general questions it just answers.

Live app: https://arnav-ghanex-agent.streamlit.app

## Architecture

```
app.py             Streamlit chat UI, history, example prompts, shows each tool call
agent/agent.py     the agent loop
agent/tools.py     the four tools, their JSON schemas, and run_tool()
agent/llm.py       OpenAI-compatible client (Groq by default) and error handling
```

The loop in `agent/agent.py`:

```
messages = system prompt + last 20 chat messages + new user message
repeat at most 5 times:
    send messages + tool schemas to the model (tool_choice = "auto")
    if the reply has no tool calls: return it            <- direct answer
    else for each tool call:
        result = run_tool(name, arguments)               <- never raises
        append the result as a "tool" message
stop with a "too many steps" message
```

This uses the model's native tool calling (no LangChain). The model sees each tool's name, description and JSON schema and returns structured calls. The code does not guess intent with keywords; the model decides, guided by the system prompt, which tells it to answer greetings and general questions directly and use tools only when needed. Because the loop repeats, the model can call several tools one after another, for example weather in two cities and then a comparison.

Error handling happens in layers:

1. Each tool raises `ToolError` with a plain message for things like an unknown city, an unsupported currency, a timeout or HTTP errors.
2. `run_tool()` checks the tool name, that the arguments are valid JSON, that required arguments are present and have the right types, then catches `ToolError` and any unexpected exception. It always returns a dict, with an `error` key on failure.
3. The error goes back to the model as the tool result, so the model can explain it to the user ("I couldn't find a place called ..."). The chat never crashes because of a tool.
4. Model-level problems (rate limit, bad key, network) become an `LLMError` with a friendly message shown in the chat. A malformed tool call from the model is retried once.

## Tools

| Tool | Parameters | Source |
|---|---|---|
| `calculator` | `expression` (string) | Local. Parses the expression with Python's `ast` module and only allows numbers, `+ - * / // % **`, brackets, a few math functions (`sqrt`, `log`, `sin`, ...) and `pi`, `e`. No `eval`, so input like `__import__('os')` is rejected. Exponents above 1000 are refused so `9**9**9` cannot freeze the app. |
| `get_weather` | `location` (string) | Open-Meteo geocoding, then the Open-Meteo forecast API. Returns current temperature, feels-like, humidity, wind, condition, today's high and low and rain chance. Free, no key. |
| `search_wikipedia` | `query` (string) | MediaWiki search API for the best match, then the REST summary endpoint. Returns title, summary, link and other matches. Free, no key. |
| `convert_currency` | `amount` (number), `from_currency` (string), `to_currency` (string) | Frankfurter API (European Central Bank reference rates, updated once per working day). Free, no key. Covers about 30 major currencies including USD, EUR, GBP, INR. |

## Example prompts

Direct answers, no tool:
- "Hi! What can you do?"
- "Explain what an API is in one sentence."

One tool:
- "What's the weather in Pune right now?" (get_weather)
- "What is 18% GST on 1249.50?" (calculator)
- "Convert 250 USD to INR" (convert_currency)
- "Who is Satya Nadella?" (search_wikipedia)

Several tools in a row:
- "Is it warmer in Delhi or Mumbai right now?" (get_weather twice)
- "Convert 100 EUR to INR and then add 18% tax" (convert_currency, then calculator)

Error handling:
- "What's the weather in Xyzzyville?" (the agent says it could not find the place)
- "Convert 10 USD to XYZ" (the agent says the currency is not supported)

The UI shows every tool call in a small box with its arguments and raw result, so you can see what the agent did.

## Run locally

Needs Python 3.10 or newer.

```bash
cd task2-agent
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env             # then put your free Groq key in .env
streamlit run app.py
```

Get a free key at https://console.groq.com/keys. The tools themselves need no keys.

## Tests

```bash
cd task2-agent
pytest
```

All tests run offline: HTTP calls are replaced with canned responses and the model is replaced with scripted replies. They cover each tool (success, bad input, not found, timeout, HTTP errors), the calculator's safety rules, `run_tool` argument checks, the agent loop (direct answer, one tool, several tools, tool errors, step limit, history trimming) and the Streamlit page.

## Deploying (Streamlit Community Cloud)

New app, pick this repo, main file `task2-agent/app.py`, Python 3.11. In Advanced settings, Secrets, add `GROQ_API_KEY = "your-key"`.

## Limits

- Groq free tier for openai/gpt-oss-20b: 30 requests per minute, 1,000 per day, 8,000 tokens per minute (Groq rate limit page, November 2026). One chat message can use several model calls. Each browser session gets 20 messages on the shared key, and visitors can paste their own free key in the sidebar.
- Only user and assistant text is kept in history. Tool results from earlier turns are not resent, which keeps requests small, but the agent may call a tool again if you ask about an earlier result in detail.
- Exchange rates are daily reference rates, not live market prices.
- Open-Meteo and Wikipedia are free services with fair-use limits; heavy use could be throttled, which the tools report as an error.
