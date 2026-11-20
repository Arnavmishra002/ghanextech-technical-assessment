"""Fetch everything the analyzer needs about a public repo from the GitHub REST API.

About 6 to 9 API calls per repository:
  /repos/{o}/{r}                     metadata (stars, forks, license, topics, ...)
  /repos/{o}/{r}/languages           bytes of code per language
  /repos/{o}/{r}/readme              README (base64)
  /repos/{o}/{r}/git/trees/{branch}  full file list in one call (recursive=1)
  /repos/{o}/{r}/contents/{path}     up to 3 package files (requirements.txt, package.json, ...)
  /repos/{o}/{r}/issues              open "good first issue" issues

Without a token GitHub allows 60 calls per hour per IP address; with any
personal access token it allows 5,000. Every failure is turned into a
GitHubError with a message meant for the user.
"""

import base64
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

import requests

API = "https://api.github.com"
TIMEOUT_SECONDS = 15
MAX_README_CHARS = 20_000
MAX_MANIFEST_CHARS = 4_000
MAX_MANIFESTS = 3

MANIFEST_NAMES = [
    "pyproject.toml", "requirements.txt", "setup.py", "setup.cfg", "Pipfile", "environment.yml",
    "package.json", "Cargo.toml", "go.mod", "pom.xml", "build.gradle", "build.gradle.kts",
    "Gemfile", "composer.json", "pubspec.yaml", "CMakeLists.txt",
]

_NAME = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})"
_REPO = r"[A-Za-z0-9._-]{1,100}"
_PATTERNS = [
    re.compile(rf"^(?:https?://)?(?:www\.)?github\.com/({_NAME})/({_REPO}?)(?:\.git)?(?:[/?#].*)?$"),
    re.compile(rf"^git@github\.com:({_NAME})/({_REPO}?)(?:\.git)?$"),
    re.compile(rf"^({_NAME})/({_REPO}?)(?:\.git)?$"),
]


class GitHubError(Exception):
    """kind is one of: invalid, not_found, rate_limit, auth, forbidden, network, server."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def parse_repo(text: str) -> tuple[str, str]:
    """Accept 'owner/repo', a github.com URL (any page inside the repo) or an SSH URL."""
    text = (text or "").strip().rstrip("/")
    for pattern in _PATTERNS:
        match = pattern.match(text)
        if match and match.group(2) not in ("", ".", ".."):
            return match.group(1), match.group(2)
    raise GitHubError(
        "invalid",
        "That does not look like a GitHub repository. Use 'owner/repo' or a link like https://github.com/owner/repo.",
    )


@dataclass
class RepoSnapshot:
    owner: str
    name: str
    metadata: dict
    languages: dict[str, int]
    readme: str
    paths: list[str]  # every file path in the default branch
    tree_truncated: bool
    manifests: dict[str, str]  # file name -> content
    good_first_issues: list[dict] = field(default_factory=list)


class GitHubClient:
    def __init__(self, token: str | None = None, session: requests.Session | None = None):
        self.session = session or requests.Session()
        self.headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "repo-analyzer-demo",
        }
        if token:
            self.headers["Authorization"] = f"Bearer {token.strip()}"
        self.rate_limit_remaining: int | None = None

    def get(self, path: str, params: dict | None = None, missing_ok: bool = False):
        try:
            r = self.session.get(API + path, params=params, headers=self.headers, timeout=TIMEOUT_SECONDS)
        except requests.Timeout as exc:
            raise GitHubError("network", "GitHub took too long to answer. Please try again.") from exc
        except requests.RequestException as exc:
            raise GitHubError("network", "Could not connect to GitHub. Check your internet connection.") from exc

        remaining = r.headers.get("X-RateLimit-Remaining")
        if remaining is not None and remaining.isdigit():
            self.rate_limit_remaining = int(remaining)

        if r.status_code in (404, 409) and missing_ok:
            return None  # 409 means the repository is empty
        if r.status_code == 404:
            raise GitHubError("not_found", "Repository not found. Check the name, or it may be private.")
        if r.status_code == 401:
            raise GitHubError("auth", "GitHub rejected the access token. Check it or leave the token field empty.")
        if r.status_code == 429 or (r.status_code == 403 and remaining == "0"):
            raise GitHubError("rate_limit", self._rate_limit_message(r))
        if r.status_code == 403:
            raise GitHubError("forbidden", "GitHub refused access to this repository.")
        if r.status_code >= 500:
            raise GitHubError("server", f"GitHub had a server problem (HTTP {r.status_code}). Please try again.")
        if r.status_code >= 400:
            raise GitHubError("server", f"GitHub returned HTTP {r.status_code}.")
        return r.json()

    def _rate_limit_message(self, r) -> str:
        reset = r.headers.get("X-RateLimit-Reset", "")
        when = ""
        if reset.isdigit():
            when = " It resets at " + datetime.fromtimestamp(int(reset), tz=timezone.utc).strftime("%H:%M UTC") + "."
        hint = "" if "Authorization" in self.headers else " Add a GitHub token in the sidebar to get 5,000 requests per hour."
        return "GitHub API rate limit reached." + when + hint

    def _file_text(self, owner: str, repo: str, path: str, limit: int) -> str:
        data = self.get(f"/repos/{owner}/{repo}/contents/{path}", missing_ok=True)
        if not data or data.get("encoding") != "base64":
            return ""
        return base64.b64decode(data["content"]).decode("utf-8", errors="replace")[:limit]

    def fetch(self, owner: str, repo: str) -> RepoSnapshot:
        meta = self.get(f"/repos/{owner}/{repo}")
        owner, repo = meta["owner"]["login"], meta["name"]  # follow renames, fix letter case
        languages = self.get(f"/repos/{owner}/{repo}/languages") or {}

        readme = ""
        readme_data = self.get(f"/repos/{owner}/{repo}/readme", missing_ok=True)
        if readme_data and readme_data.get("encoding") == "base64":
            readme = base64.b64decode(readme_data["content"]).decode("utf-8", errors="replace")[:MAX_README_CHARS]

        tree = self.get(
            f"/repos/{owner}/{repo}/git/trees/{meta['default_branch']}", {"recursive": "1"}, missing_ok=True
        ) or {"tree": [], "truncated": False}
        paths = [item["path"] for item in tree.get("tree", []) if item.get("type") == "blob"]

        root_files = {p for p in paths if "/" not in p}
        manifests = {}
        for name in MANIFEST_NAMES:
            if name in root_files and len(manifests) < MAX_MANIFESTS:
                text = self._file_text(owner, repo, name, MAX_MANIFEST_CHARS)
                if text:
                    manifests[name] = text

        issues = self.get(
            f"/repos/{owner}/{repo}/issues",
            {"labels": "good first issue", "state": "open", "per_page": 10},
            missing_ok=True,
        ) or []
        good_first = [
            {"title": i["title"], "url": i["html_url"], "comments": i.get("comments", 0)}
            for i in issues
            if "pull_request" not in i
        ][:5]

        return RepoSnapshot(
            owner=owner,
            name=repo,
            metadata=meta,
            languages=languages,
            readme=readme,
            paths=paths,
            tree_truncated=bool(tree.get("truncated")),
            manifests=manifests,
            good_first_issues=good_first,
        )
