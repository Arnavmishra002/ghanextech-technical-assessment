import base64
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def b64(text: str) -> dict:
    return {"encoding": "base64", "content": base64.b64encode(text.encode()).decode()}


class FakeResponse:
    def __init__(self, status_code=200, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {"X-RateLimit-Remaining": "55"}

    def json(self):
        return self._payload


class FakeSession:
    """Answers GET requests from a dict of {path: FakeResponse}. Unknown paths give 404."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        path = url.replace("https://api.github.com", "")
        self.calls.append((path, params, headers))
        response = self.routes.get(path, FakeResponse(404, {"message": "Not Found"}))
        if isinstance(response, Exception):
            raise response
        return response


META = {
    "full_name": "acme/widget",
    "name": "widget",
    "owner": {"login": "acme"},
    "description": "A tiny widget library.",
    "stargazers_count": 1200,
    "forks_count": 80,
    "open_issues_count": 14,
    "language": "Python",
    "license": {"spdx_id": "MIT"},
    "topics": ["widgets", "python"],
    "archived": False,
    "default_branch": "main",
    "created_at": "2023-01-05T10:00:00Z",
    "pushed_at": "2026-11-01T10:00:00Z",
    "homepage": "",
}

TREE = {
    "truncated": False,
    "tree": [
        {"path": "README.md", "type": "blob"},
        {"path": "LICENSE", "type": "blob"},
        {"path": "pyproject.toml", "type": "blob"},
        {"path": "src", "type": "tree"},
        {"path": "src/widget/__init__.py", "type": "blob"},
        {"path": "src/widget/core.py", "type": "blob"},
        {"path": "tests/test_core.py", "type": "blob"},
        {"path": ".github/workflows/ci.yml", "type": "blob"},
    ],
}


def widget_routes(**overrides):
    routes = {
        "/repos/acme/widget": FakeResponse(200, META),
        "/repos/acme/widget/languages": FakeResponse(200, {"Python": 9000, "Shell": 1000}),
        "/repos/acme/widget/readme": FakeResponse(200, b64("# Widget\nMakes widgets. Install with pip.")),
        "/repos/acme/widget/git/trees/main": FakeResponse(200, TREE),
        "/repos/acme/widget/contents/pyproject.toml": FakeResponse(200, b64('[project]\nname = "widget"\n')),
        "/repos/acme/widget/issues": FakeResponse(
            200,
            [
                {"title": "Add type hints to core.py", "html_url": "https://github.com/acme/widget/issues/7", "comments": 2},
                {"title": "A pull request", "html_url": "https://github.com/acme/widget/pull/8", "pull_request": {}},
            ],
        ),
    }
    routes.update(overrides)
    return routes


@pytest.fixture
def widget_session():
    return FakeSession(widget_routes())
