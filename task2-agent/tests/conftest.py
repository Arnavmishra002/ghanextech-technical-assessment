import json
import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=None):
        self.status_code = status_code
        self._payload = payload
        self.text = text if text is not None else (json.dumps(payload) if payload is not None else "")

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


@pytest.fixture
def fake_http(monkeypatch):
    """Route requests.get calls to canned responses.

    Usage: fake_http({"geocoding-api": FakeResponse(...), ...}). The first key
    that appears in the URL wins. Exceptions in the map are raised instead.
    """
    calls = []

    def install(routes):
        def fake_get(url, params=None, headers=None, timeout=None):
            calls.append((url, params))
            for fragment, response in routes.items():
                if fragment in url:
                    if isinstance(response, Exception):
                        raise response
                    return response
            raise AssertionError(f"unexpected URL {url}")

        monkeypatch.setattr(requests, "get", fake_get)
        return calls

    return install


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("tests must not use the network")

    monkeypatch.setattr(requests, "get", blocked)
