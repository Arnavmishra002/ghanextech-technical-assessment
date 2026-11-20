"""Turn a repository snapshot into a structured review using an LLM.

The prompt contains only data fetched from GitHub plus the computed health
checks, and the model must answer in JSON with fixed keys. The JSON is
validated; if it is broken or has fewer than three recommendations, the
model is asked once more before giving up.
"""

import json

from .github_client import RepoSnapshot

MAX_README_IN_PROMPT = 5_000
MAX_PATHS_IN_PROMPT = 150
MAX_MANIFEST_IN_PROMPT = 1_200

SYSTEM_PROMPT = """You are a senior software engineer reviewing a public GitHub repository for a developer who wants to understand it or contribute to it.

Base every statement only on the data provided. If something cannot be known from the data, say so instead of guessing. The health checks and "tooling_found" were computed by code from the real file list, so trust them. Never recommend adding something the repository already has (a check marked true, an item in tooling_found, or a file visible in the tree).

Reply with one JSON object and nothing else, using exactly these keys:
{
  "summary": "2-4 sentences: what the project does, who it is for, and its main tech stack",
  "tech_stack": ["main languages, frameworks and tools"],
  "architecture": "2-4 sentences on how the code is organised and any patterns visible from the file tree",
  "strengths": ["2-5 specific strengths"],
  "recommendations": [
    {"title": "short action", "detail": "1-2 sentences: what to change and why", "priority": "high | medium | low"}
  ],
  "contribution_opportunities": ["2-4 concrete ways a new contributor could help"]
}
Give 3 to 5 recommendations. Make them specific to this repository (testing, docs, CI/CD, structure, dependencies), not generic advice."""

REQUIRED_KEYS = {"summary", "tech_stack", "architecture", "strengths", "recommendations", "contribution_opportunities"}


class AnalysisError(Exception):
    """The model's output could not be used."""


def _shorten_tree(paths: list[str], limit: int) -> list[str]:
    """Prefer shallow paths so the model sees the overall structure first."""
    ranked = sorted(paths, key=lambda p: (p.count("/"), p))
    return sorted(ranked[:limit])


def build_messages(snap: RepoSnapshot, metrics: dict) -> list[dict]:
    facts = {k: v for k, v in metrics.items() if k not in ("top_level",)}
    tree = _shorten_tree(snap.paths, MAX_PATHS_IN_PROMPT)
    manifest_text = "\n\n".join(
        f"--- {name} ---\n{text[:MAX_MANIFEST_IN_PROMPT]}" for name, text in snap.manifests.items()
    ) or "(none found at the repository root)"
    issues = "\n".join(f"- {i['title']}" for i in snap.good_first_issues) or "(none labelled 'good first issue')"
    readme = snap.readme[:MAX_README_IN_PROMPT] or "(no README)"

    user = (
        f"REPOSITORY FACTS AND HEALTH CHECKS (JSON)\n{json.dumps(facts, indent=1)}\n\n"
        f"FILE TREE ({len(tree)} of {len(snap.paths)} files, shallowest first)\n" + "\n".join(tree) + "\n\n"
        f"PACKAGE FILES\n{manifest_text}\n\n"
        f"OPEN GOOD FIRST ISSUES\n{issues}\n\n"
        f"README (first {MAX_README_IN_PROMPT} characters)\n{readme}"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def parse_analysis(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{") :]
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AnalysisError("The model did not return valid JSON.") from exc
    if not isinstance(data, dict):
        raise AnalysisError("The model returned JSON that is not an object.")
    missing = REQUIRED_KEYS - data.keys()
    if missing:
        raise AnalysisError(f"The model's answer is missing: {', '.join(sorted(missing))}.")

    recs = [r for r in data["recommendations"] if isinstance(r, dict) and r.get("title")]
    if len(recs) < 3:
        raise AnalysisError("The model gave fewer than three recommendations.")
    for r in recs:
        r["priority"] = str(r.get("priority", "medium")).lower()
        if r["priority"] not in ("high", "medium", "low"):
            r["priority"] = "medium"
        r["detail"] = str(r.get("detail", ""))
    data["recommendations"] = recs
    for key in ("tech_stack", "strengths", "contribution_opportunities"):
        if not isinstance(data[key], list):
            data[key] = [str(data[key])]
    return data


def analyze(snap: RepoSnapshot, metrics: dict, llm, attempts: int = 2) -> dict:
    """llm is a function: messages -> reply text. Raises AnalysisError after the retries."""
    messages = build_messages(snap, metrics)
    error = None
    for _ in range(attempts):
        try:
            return parse_analysis(llm(messages))
        except AnalysisError as exc:
            error = exc
    raise AnalysisError(f"{error} Please try again.")
