import httpx
import pytest
from openai import APIConnectionError, AuthenticationError, RateLimitError

from rag import llm

REQUEST = httpx.Request("POST", "https://example.test/v1/chat/completions")


class FakeCompletions:
    def __init__(self, error=None, reply="ok"):
        self.error = error
        self.reply = reply
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error

        class Msg:
            content = self.reply

        class Choice:
            message = Msg()

        class Resp:
            choices = [Choice()]

        return Resp()


class FakeClient:
    def __init__(self, completions):
        self.chat = type("Chat", (), {"completions": completions})()


def test_returns_reply_and_uses_low_reasoning_for_gpt_oss(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-oss-20b")
    completions = FakeCompletions(reply="  hello  ")
    assert llm.chat(FakeClient(completions), [{"role": "user", "content": "hi"}]) == "hello"
    assert completions.kwargs["reasoning_effort"] == "low"
    assert completions.kwargs["temperature"] == 0


def test_other_models_do_not_get_reasoning_effort(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "some-other-model")
    completions = FakeCompletions()
    llm.chat(FakeClient(completions), [])
    assert "reasoning_effort" not in completions.kwargs


@pytest.mark.parametrize(
    "error, expected",
    [
        (RateLimitError("slow down", response=httpx.Response(429, request=REQUEST), body=None), "rate limit"),
        (AuthenticationError("bad key", response=httpx.Response(401, request=REQUEST), body=None), "rejected"),
        (APIConnectionError(request=REQUEST), "Could not reach"),
    ],
)
def test_api_errors_become_friendly_messages(error, expected):
    with pytest.raises(llm.LLMError, match=expected):
        llm.chat(FakeClient(FakeCompletions(error=error)), [])


def test_missing_key_is_reported(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(llm.LLMError, match="No API key"):
        llm.make_client()
