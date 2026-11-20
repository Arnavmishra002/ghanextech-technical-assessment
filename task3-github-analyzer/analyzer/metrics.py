"""Facts about a repository that can be computed without an LLM.

These are shown in the dashboard and also given to the LLM, so its
recommendations are based on checked facts (for example "no CI config
found") instead of guesses.
"""

import re
from datetime import datetime, timezone

from .github_client import RepoSnapshot

STALE_AFTER_DAYS = 180

_TEST_FILE = re.compile(r"(^|/)(test_[^/]+\.py|[^/]+_test\.(py|go)|[^/]+\.(test|spec)\.[jt]sx?)$")
_TEST_DIRS = {"test", "tests", "__tests__", "spec", "testing"}
_CI_PREFIXES = (".github/workflows/", ".circleci/", ".gitlab-ci.yml", ".travis.yml", "azure-pipelines.yml", "Jenkinsfile")


# Config files that show which tooling a project already uses. Passing this
# list to the LLM stops it from recommending things the repo already has.
TOOLING_FILES = {
    ".pre-commit-config.yaml": "pre-commit hooks",
    "ruff.toml": "ruff",
    ".ruff.toml": "ruff",
    ".flake8": "flake8",
    "tox.ini": "tox",
    "noxfile.py": "nox",
    "mypy.ini": "mypy",
    ".eslintrc.json": "eslint",
    "eslint.config.js": "eslint",
    ".prettierrc": "prettier",
    ".editorconfig": "editorconfig",
    "Makefile": "Makefile",
    "Dockerfile": "Dockerfile",
    "docker-compose.yml": "docker compose",
    "compose.yaml": "docker compose",
    "mkdocs.yml": "mkdocs docs site",
    ".readthedocs.yaml": "Read the Docs",
    "codecov.yml": "codecov",
    ".github/dependabot.yml": "dependabot",
    ".github/CODEOWNERS": "CODEOWNERS",
    ".github/PULL_REQUEST_TEMPLATE.md": "pull request template",
}


def tooling_found(paths: list[str]) -> list[str]:
    present = set(paths)
    found = {label for path, label in TOOLING_FILES.items() if path in present}
    if any(p.startswith(".github/ISSUE_TEMPLATE/") for p in paths):
        found.add("issue templates")
    return sorted(found)


def _has_tests(paths: list[str]) -> bool:
    for p in paths:
        if _TEST_FILE.search(p):
            return True
        if any(part.lower() in _TEST_DIRS for part in p.split("/")[:-1]):
            return True
    return False


def _has_file(paths: list[str], *names: str) -> bool:
    """True if a file with one of these names (any extension, any case) is at root, docs/ or .github/."""
    wanted = {n.lower() for n in names}
    for p in paths:
        folder, _, filename = p.rpartition("/")
        if folder.lower() in ("", "docs", ".github") and filename.lower().split(".")[0] in wanted:
            return True
    return False


def language_shares(languages: dict[str, int]) -> dict[str, float]:
    total = sum(languages.values())
    if not total:
        return {}
    shares = {lang: round(100 * n / total, 1) for lang, n in languages.items()}
    return dict(sorted(shares.items(), key=lambda kv: kv[1], reverse=True))


def compute_metrics(snap: RepoSnapshot, now: datetime | None = None) -> dict:
    meta = snap.metadata
    now = now or datetime.now(timezone.utc)
    pushed = datetime.fromisoformat(meta["pushed_at"].replace("Z", "+00:00")) if meta.get("pushed_at") else None
    days_since_push = (now - pushed).days if pushed else None
    top_level = sorted({p.split("/")[0] + ("/" if "/" in p else "") for p in snap.paths})

    checks = {
        "README": bool(snap.readme.strip()),
        "License": bool(meta.get("license")),
        "Contributing guide": _has_file(snap.paths, "contributing"),
        "CI config": any(p.startswith(_CI_PREFIXES) for p in snap.paths),
        "Tests": _has_tests(snap.paths),
        f"Pushed in last {STALE_AFTER_DAYS} days": days_since_push is not None and days_since_push <= STALE_AFTER_DAYS,
    }

    return {
        "full_name": meta["full_name"],
        "description": meta.get("description") or "",
        "homepage": meta.get("homepage") or "",
        "stars": meta.get("stargazers_count", 0),
        "forks": meta.get("forks_count", 0),
        # GitHub counts open pull requests in this number too.
        "open_issues_and_prs": meta.get("open_issues_count", 0),
        "primary_language": meta.get("language") or "Unknown",
        "license": (meta.get("license") or {}).get("spdx_id") or "None",
        "topics": meta.get("topics", []),
        "archived": bool(meta.get("archived")),
        "created_at": (meta.get("created_at") or "")[:10],
        "last_push": (meta.get("pushed_at") or "")[:10],
        "days_since_push": days_since_push,
        "file_count": len(snap.paths),
        "file_list_truncated": snap.tree_truncated,
        "top_level": top_level,
        "languages_percent": language_shares(snap.languages),
        "package_files": list(snap.manifests),
        "readme_words": len(snap.readme.split()),
        "has_docs_folder": any(p.lower().startswith(("docs/", "doc/")) for p in snap.paths),
        "has_dockerfile": any(p.split("/")[-1].lower().startswith("dockerfile") for p in snap.paths),
        "has_code_of_conduct": _has_file(snap.paths, "code_of_conduct"),
        "has_changelog": _has_file(snap.paths, "changelog", "changes", "history"),
        "tooling_found": tooling_found(snap.paths),
        "checks": checks,
        "health_score": f"{sum(checks.values())}/{len(checks)}",
        "good_first_issue_count": len(snap.good_first_issues),
    }
