"""Smoke tests of the Streamlit dashboard with GitHub and the LLM replaced by fakes."""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import analyzer.github_client
import analyzer.llm
from conftest import FakeSession, widget_routes
from test_llm_analysis import GOOD

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture(autouse=True)
def fake_github(monkeypatch):
    original = analyzer.github_client.GitHubClient.__init__

    def init(self, token=None, session=None):
        original(self, token=token, session=FakeSession(widget_routes()))

    monkeypatch.setattr(analyzer.github_client.GitHubClient, "__init__", init)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)


def submit(repo_text):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_input(key="repo").set_value(repo_text)
    at.button[0].click().run()
    return at


def test_first_page_shows_examples():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert "pallets/flask" in at.info[0].value


def test_invalid_input_shows_error():
    at = submit("not a repo")
    assert not at.exception
    assert "does not look like a GitHub repository" in at.error[0].value


def test_full_analysis_with_fake_llm(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(analyzer.llm, "make_client", lambda key=None: object())
    monkeypatch.setattr(analyzer.llm, "chat", lambda client, messages, **kw: json.dumps(GOOD))
    at = submit("https://github.com/acme/widget")
    assert not at.exception
    assert at.subheader[0].value == "acme/widget"
    assert [m.value for m in at.metric][:2] == ["1,200", "80"]
    text = " ".join(m.value for m in at.markdown)
    assert "Add a contributing guide" in text
    assert "Add type hints to core.py" in text


def test_without_llm_key_facts_still_show(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    at = submit("acme/widget")
    assert not at.exception
    assert at.metric[0].value == "1,200"
    assert "No LLM API key" in at.warning[0].value
