"""Smoke tests of the Streamlit page using Streamlit's built-in test runner."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

import agent.agent
import agent.llm
from agent.agent import AgentReply, ToolCallRecord

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_page_renders_with_tools_and_examples():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value == "Tool-calling agent"
    assert any("Pune" in b.label for b in at.sidebar.button)


def test_missing_key_gives_message_not_crash(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("hello").run()
    assert not at.exception
    assert "No API key" in at.chat_message[1].markdown[0].value


def test_chat_shows_reply_and_tool_call(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    class FakeAgent:
        def __init__(self, client):
            pass

        def run(self, history, message):
            return AgentReply("It is 4.", [ToolCallRecord("calculator", '{"expression": "2+2"}', {"result": 4})])

    monkeypatch.setattr(agent.agent, "Agent", FakeAgent)
    monkeypatch.setattr(agent.llm, "make_client", lambda key=None: object())
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("what is 2+2").run()
    assert not at.exception
    assistant = at.chat_message[1]
    assert assistant.markdown[-1].value == "It is 4."
    assert "Used tool: calculator" in assistant.expander[0].label
