import pytest
import requests

from analyzer.github_client import GitHubClient, GitHubError, parse_repo
from conftest import FakeResponse, FakeSession, widget_routes


@pytest.mark.parametrize(
    "text, expected",
    [
        ("pallets/flask", ("pallets", "flask")),
        ("  pallets/flask  ", ("pallets", "flask")),
        ("https://github.com/pallets/flask", ("pallets", "flask")),
        ("http://www.github.com/pallets/flask/", ("pallets", "flask")),
        ("https://github.com/pallets/flask.git", ("pallets", "flask")),
        ("github.com/pallets/flask/tree/main/src", ("pallets", "flask")),
        ("https://github.com/pallets/flask/issues?q=bug", ("pallets", "flask")),
        ("git@github.com:pallets/flask.git", ("pallets", "flask")),
        ("my-org/my.repo_name", ("my-org", "my.repo_name")),
    ],
)
def test_parse_repo_accepts_common_formats(text, expected):
    assert parse_repo(text) == expected


@pytest.mark.parametrize(
    "text",
    ["", "flask", "https://gitlab.com/a/b", "https://github.com/pallets", "a/b/c", "-bad/name", "owner/..", "not a url at all"],
)
def test_parse_repo_rejects_invalid_input(text):
    with pytest.raises(GitHubError) as info:
        parse_repo(text)
    assert info.value.kind == "invalid"


def test_fetch_collects_everything(widget_session):
    snap = GitHubClient(session=widget_session).fetch("acme", "widget")
    assert snap.metadata["stargazers_count"] == 1200
    assert snap.languages == {"Python": 9000, "Shell": 1000}
    assert snap.readme.startswith("# Widget")
    assert "src/widget/core.py" in snap.paths and "src" not in snap.paths  # folders are not files
    assert snap.manifests == {"pyproject.toml": '[project]\nname = "widget"\n'}
    assert snap.good_first_issues == [
        {"title": "Add type hints to core.py", "url": "https://github.com/acme/widget/issues/7", "comments": 2}
    ]


def test_token_is_sent_only_when_given(widget_session):
    GitHubClient(session=widget_session).fetch("acme", "widget")
    assert all("Authorization" not in h for _, _, h in widget_session.calls)

    session = FakeSession(widget_routes())
    GitHubClient(token="abc123", session=session).fetch("acme", "widget")
    assert all(h["Authorization"] == "Bearer abc123" for _, _, h in session.calls)


def test_missing_readme_and_empty_repo_are_fine():
    session = FakeSession(
        widget_routes(
            **{
                "/repos/acme/widget/readme": FakeResponse(404, {}),
                "/repos/acme/widget/git/trees/main": FakeResponse(409, {"message": "Git Repository is empty."}),
                "/repos/acme/widget/issues": FakeResponse(200, []),
            }
        )
    )
    snap = GitHubClient(session=session).fetch("acme", "widget")
    assert snap.readme == "" and snap.paths == [] and snap.manifests == {} and snap.good_first_issues == []


def test_not_found_or_private():
    with pytest.raises(GitHubError) as info:
        GitHubClient(session=FakeSession({})).fetch("acme", "secret")
    assert info.value.kind == "not_found" and "private" in str(info.value)


def test_rate_limit_message_has_reset_time_and_token_hint():
    limited = FakeResponse(403, {"message": "API rate limit exceeded"}, {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1795181414"})
    with pytest.raises(GitHubError) as info:
        GitHubClient(session=FakeSession({"/repos/acme/widget": limited})).fetch("acme", "widget")
    assert info.value.kind == "rate_limit"
    assert "UTC" in str(info.value) and "token" in str(info.value)


def test_bad_token_and_forbidden_and_server_errors():
    cases = [
        (FakeResponse(401, {}), "auth"),
        (FakeResponse(403, {}, {"X-RateLimit-Remaining": "40"}), "forbidden"),
        (FakeResponse(502, {}), "server"),
        (requests.Timeout(), "network"),
        (requests.ConnectionError(), "network"),
    ]
    for response, kind in cases:
        with pytest.raises(GitHubError) as info:
            GitHubClient(token="t", session=FakeSession({"/repos/acme/widget": response})).fetch("acme", "widget")
        assert info.value.kind == kind


def test_remaining_rate_limit_is_tracked(widget_session):
    client = GitHubClient(session=widget_session)
    client.fetch("acme", "widget")
    assert client.rate_limit_remaining == 55
