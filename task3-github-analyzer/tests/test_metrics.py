from datetime import datetime, timezone

from analyzer.github_client import GitHubClient, RepoSnapshot
from analyzer.metrics import compute_metrics, language_shares

NOW = datetime(2026, 11, 20, tzinfo=timezone.utc)


def test_metrics_for_healthy_repo(widget_session):
    snap = GitHubClient(session=widget_session).fetch("acme", "widget")
    m = compute_metrics(snap, now=NOW)
    assert m["stars"] == 1200 and m["license"] == "MIT"
    assert m["days_since_push"] == 18  # 1 Nov 10:00 to 20 Nov 00:00 is 18 full days
    assert m["languages_percent"] == {"Python": 90.0, "Shell": 10.0}
    assert m["checks"] == {
        "README": True,
        "License": True,
        "Contributing guide": False,
        "CI config": True,
        "Tests": True,
        "Pushed in last 180 days": True,
    }
    assert m["health_score"] == "5/6"
    assert m["top_level"] == [".github/", "LICENSE", "README.md", "pyproject.toml", "src/", "tests/"]
    assert m["package_files"] == ["pyproject.toml"]


def bare_snapshot(paths, **meta):
    base = {"full_name": "x/y", "pushed_at": "2024-01-01T00:00:00Z", "license": None}
    base.update(meta)
    return RepoSnapshot("x", "y", base, {}, "", paths, False, {})


def test_metrics_for_neglected_repo():
    m = compute_metrics(bare_snapshot(["main.py"]), now=NOW)
    assert m["health_score"] == "0/6"
    assert m["license"] == "None" and m["languages_percent"] == {}


def test_test_detection_patterns():
    for path in ["pkg/test_utils.py", "server_test.go", "src/app.spec.ts", "web/Button.test.jsx", "spec/model_spec.rb", "src/__tests__/a.js"]:
        assert compute_metrics(bare_snapshot([path]), now=NOW)["checks"]["Tests"], path
    for path in ["src/contest.py", "latest/notes.md", "testimonials.html"]:
        assert not compute_metrics(bare_snapshot([path]), now=NOW)["checks"]["Tests"], path


def test_contributing_found_in_github_or_docs_folder():
    for path in [".github/CONTRIBUTING.md", "docs/contributing.rst", "CONTRIBUTING"]:
        assert compute_metrics(bare_snapshot([path]), now=NOW)["checks"]["Contributing guide"], path
    assert not compute_metrics(bare_snapshot(["src/contributing/helpers.py"]), now=NOW)["checks"]["Contributing guide"]


def test_language_shares_sorted_and_rounded():
    assert language_shares({"C": 1, "Python": 2}) == {"Python": 66.7, "C": 33.3}
    assert language_shares({}) == {}
