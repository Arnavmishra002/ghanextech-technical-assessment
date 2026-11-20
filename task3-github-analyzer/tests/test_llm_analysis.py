import json

import pytest

from analyzer.github_client import GitHubClient
from analyzer.llm_analysis import MAX_README_IN_PROMPT, AnalysisError, analyze, build_messages, parse_analysis
from analyzer.metrics import compute_metrics
from analyzer.report import to_markdown

GOOD = {
    "summary": "A small Python widget library.",
    "tech_stack": ["Python"],
    "architecture": "src layout with one core module.",
    "strengths": ["Has CI", "Has tests"],
    "recommendations": [
        {"title": "Add a contributing guide", "detail": "None was found.", "priority": "High"},
        {"title": "Add type hints", "detail": "Helps users.", "priority": "medium"},
        {"title": "Publish docs", "detail": "README is short.", "priority": "urgent"},
    ],
    "contribution_opportunities": ["Type hints issue #7"],
}


@pytest.fixture
def snap_and_metrics(widget_session):
    snap = GitHubClient(session=widget_session).fetch("acme", "widget")
    return snap, compute_metrics(snap)


def test_prompt_contains_facts_tree_manifest_issues_and_readme(snap_and_metrics):
    snap, metrics = snap_and_metrics
    system, user = build_messages(snap, metrics)
    assert system["role"] == "system" and "JSON" in system["content"]
    text = user["content"]
    assert '"Contributing guide": false' in text
    assert "src/widget/core.py" in text
    assert "--- pyproject.toml ---" in text
    assert "Add type hints to core.py" in text
    assert "Makes widgets" in text


def test_prompt_size_is_bounded(snap_and_metrics):
    snap, metrics = snap_and_metrics
    snap.readme = "word " * 20_000
    snap.paths = [f"pkg/module_{i}.py" for i in range(5_000)]
    user = build_messages(snap, metrics)[1]["content"]
    assert len(user) < MAX_README_IN_PROMPT + 12_000


def test_parse_cleans_priorities_and_accepts_code_fences():
    data = parse_analysis("```json\n" + json.dumps(GOOD) + "\n```")
    assert [r["priority"] for r in data["recommendations"]] == ["high", "medium", "medium"]


@pytest.mark.parametrize(
    "bad",
    [
        "not json",
        "[1, 2]",
        json.dumps({k: v for k, v in GOOD.items() if k != "summary"}),
        json.dumps({**GOOD, "recommendations": GOOD["recommendations"][:2]}),
    ],
)
def test_parse_rejects_unusable_output(bad):
    with pytest.raises(AnalysisError):
        parse_analysis(bad)


def test_analyze_retries_once_then_succeeds(snap_and_metrics):
    replies = iter(["oops", json.dumps(GOOD)])
    result = analyze(*snap_and_metrics, llm=lambda m: next(replies))
    assert result["summary"] == GOOD["summary"]


def test_analyze_gives_up_after_retries(snap_and_metrics):
    with pytest.raises(AnalysisError, match="try again"):
        analyze(*snap_and_metrics, llm=lambda m: "still not json")


def test_markdown_report(snap_and_metrics):
    snap, metrics = snap_and_metrics
    md = to_markdown(metrics, parse_analysis(json.dumps(GOOD)), snap.good_first_issues)
    assert md.startswith("# acme/widget")
    assert "- [high] Add a contributing guide: None was found." in md
    assert "[Add type hints to core.py](https://github.com/acme/widget/issues/7)" in md
