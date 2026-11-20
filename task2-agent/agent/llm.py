"""Connection to an OpenAI-compatible chat API that supports tool calling.

Groq is the default (free tier, no card). Gemini, OpenRouter or a local
Ollama server can be used by changing LLM_BASE_URL and LLM_MODEL.
"""

import os

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-20b"


class LLMError(Exception):
    """An LLM failure, with a message that is safe to show to the user."""


def get_api_key() -> str:
    return os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY") or ""


def model_name() -> str:
    return os.getenv("LLM_MODEL", DEFAULT_MODEL)


def make_client(api_key: str | None = None) -> OpenAI:
    key = api_key or get_api_key()
    if not key:
        raise LLMError("No API key set. Add GROQ_API_KEY to your .env file or app secrets.")
    # On a 429 the SDK waits for Groq's retry-after header before retrying, so a short
    # burst over the free 8,000 tokens per minute limit slows down instead of failing.
    return OpenAI(api_key=key, base_url=os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL), timeout=60, max_retries=3)


def complete(client: OpenAI, messages: list[dict], tools: list[dict]):
    """One model call. Returns the assistant message (which may contain tool_calls)."""
    model = model_name()
    extra = {"reasoning_effort": "low"} if "gpt-oss" in model else {}
    attempts = 2  # retry once if the model writes a malformed tool call
    for attempt in range(attempts):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0,
                max_completion_tokens=2048,
                **extra,
            )
            return response.choices[0].message
        except BadRequestError as exc:
            if "tool" in str(exc).lower() and attempt < attempts - 1:
                continue
            raise LLMError("The model made an invalid request. Please rephrase and try again.") from exc
        except RateLimitError as exc:
            raise LLMError("The free model's rate limit was reached. Please wait a minute and try again.") from exc
        except AuthenticationError as exc:
            raise LLMError("The API key was rejected. Check that it is correct.") from exc
        except (APITimeoutError, APIConnectionError) as exc:
            raise LLMError("Could not reach the model provider. Please try again.") from exc
        except APIError as exc:
            raise LLMError(f"The model provider returned an error: {exc.message}") from exc
