"""Smoke test of the Streamlit page using Streamlit's built-in test runner."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_page_renders_and_waits_for_upload():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value == "Ask your PDFs"
    assert "Upload a PDF" in at.info[0].value
