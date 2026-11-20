"""Small wrapper around any OpenAI-compatible chat API.

Groq is the default because its free tier needs no card, so a public demo
cannot run up a bill. Gemini, OpenRouter or a local Ollama server also speak
this API, so switching provider is only a change of LLM_BASE_URL and LLM_MODEL.
"""

import os

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-20b"


class LLMError(Exception):
    """An LLM failure, with a message that is safe to show to the user."""


def get_api_key() -> str:
    return os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY") or ""


def make_client(api_key: str | None = None) -> OpenAI:
    key = api_key or get_api_key()
    if not key:
        raise LLMError("No API key set. Add GROQ_API_KEY to your .env file or app secrets.")
    base_url = os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL)
    return OpenAI(api_key=key, base_url=base_url, timeout=60, max_retries=1)


def model_name() -> str:
    return os.getenv("LLM_MODEL", DEFAULT_MODEL)


def chat(client: OpenAI, messages: list[dict], **kwargs) -> str:
    """Send messages and return the reply text, or raise LLMError."""
    model = model_name()
    if "gpt-oss" in model:
        # gpt-oss thinks before it answers. "low" keeps that short, which
        # saves tokens on the free tier (8K tokens per minute).
        kwargs.setdefault("reasoning_effort", "low")
    try:
        response = client.chat.completions.create(
            model=model, messages=messages, temperature=0, max_completion_tokens=2048, **kwargs
        )
    except RateLimitError as exc:
        raise LLMError(
            "The free model's rate limit was reached. Please wait a minute and try again."
        ) from exc
    except AuthenticationError as exc:
        raise LLMError("The API key was rejected. Check that it is correct.") from exc
    except (APITimeoutError, APIConnectionError) as exc:
        raise LLMError("Could not reach the model provider. Please try again.") from exc
    except APIError as exc:
        raise LLMError(f"The model provider returned an error: {exc.message}") from exc
    return (response.choices[0].message.content or "").strip()
