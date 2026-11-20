"""Build a Markdown report so users can download and share an analysis."""


def to_markdown(metrics: dict, analysis: dict, good_first_issues: list[dict]) -> str:
    lines = [
        f"# {metrics['full_name']}",
        "",
        metrics["description"],
        "",
        f"Stars: {metrics['stars']} | Forks: {metrics['forks']} | Open issues and PRs: {metrics['open_issues_and_prs']} "
        f"| License: {metrics['license']} | Last push: {metrics['last_push']} | Health checks: {metrics['health_score']}",
        "",
        "## Summary",
        analysis["summary"],
        "",
        "## Tech stack",
        ", ".join(analysis["tech_stack"]),
        "",
        "## Architecture",
        analysis["architecture"],
        "",
        "## Strengths",
        *[f"- {s}" for s in analysis["strengths"]],
        "",
        "## Recommendations",
        *[f"- [{r['priority']}] {r['title']}: {r['detail']}" for r in analysis["recommendations"]],
        "",
        "## Contribution opportunities",
        *[f"- {c}" for c in analysis["contribution_opportunities"]],
    ]
    if good_first_issues:
        lines += ["", "## Open good first issues", *[f"- [{i['title']}]({i['url']})" for i in good_first_issues]]
    return "\n".join(lines) + "\n"
