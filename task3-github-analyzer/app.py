"""Streamlit dashboard for the GitHub repository analyzer."""

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from analyzer.github_client import GitHubClient, GitHubError, RepoSnapshot, parse_repo
from analyzer.llm import LLMError, chat, get_api_key, make_client, model_name
from analyzer.llm_analysis import AnalysisError, analyze
from analyzer.metrics import compute_metrics
from analyzer.report import to_markdown

load_dotenv()

# Each analysis is one LLM call on a shared free key.
MAX_ANALYSES_PER_SESSION = 10

st.set_page_config(page_title="GitHub Repo Analyzer", layout="wide")


# Results are cached for an hour and shared between visitors, so analysing a
# popular repo twice costs no extra GitHub or LLM quota. Arguments starting
# with "_" are not part of the cache key (the token must not be).
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_snapshot(owner: str, repo: str, _client: GitHubClient) -> RepoSnapshot:
    return _client.fetch(owner, repo)


@st.cache_data(ttl=3600, show_spinner=False)
def run_analysis(full_name: str, pushed_at: str, model: str, _snap, _metrics, _llm) -> dict:
    return analyze(_snap, _metrics, _llm)


state = st.session_state
state.setdefault("analyses", 0)

st.title("GitHub Repo Analyzer")
st.caption("Paste a public GitHub repository. You get key facts, health checks and an AI review with recommendations.")

with st.sidebar:
    st.header("Optional keys")
    gh_token = st.text_input(
        "GitHub token",
        type="password",
        help="Without a token GitHub allows 60 API calls per hour (about 7 analyses). Any token raises it to 5,000.",
    )
    llm_key = st.text_input("Groq API key", type="password", help="Leave empty to use the demo key.")
    st.caption(f"Model: {model_name()}")
    st.caption("Keys typed here are used only for your session and are never stored.")

with st.form("repo_form"):
    col1, col2 = st.columns([4, 1])
    repo_text = col1.text_input(
        "Repository", key="repo", placeholder="owner/repo or https://github.com/owner/repo", label_visibility="collapsed"
    )
    submitted = col2.form_submit_button("Analyze", type="primary", width="stretch")

if not submitted:
    st.info("Try: pallets/flask, streamlit/streamlit or https://github.com/psf/requests")
    st.stop()

try:
    owner, repo = parse_repo(repo_text)
    client = GitHubClient(token=gh_token or os.getenv("GITHUB_TOKEN"))
    with st.spinner(f"Reading {owner}/{repo} from GitHub..."):
        snap = fetch_snapshot(owner.lower(), repo.lower(), client)
except GitHubError as exc:
    st.error(str(exc))
    st.stop()

metrics = compute_metrics(snap)
meta = snap.metadata

# ---------------------------------------------------------------- facts
st.subheader(metrics["full_name"])
if metrics["description"]:
    st.write(metrics["description"])
if metrics["archived"]:
    st.warning("This repository is archived (read-only).")

c = st.columns(5)
c[0].metric("Stars", f"{metrics['stars']:,}")
c[1].metric("Forks", f"{metrics['forks']:,}")
c[2].metric("Open issues + PRs", f"{metrics['open_issues_and_prs']:,}")
c[3].metric("Days since last push", metrics["days_since_push"] if metrics["days_since_push"] is not None else "-")
c[4].metric("Health checks", metrics["health_score"])

st.caption(
    f"Language: {metrics['primary_language']}  |  License: {metrics['license']}  |  "
    f"Files: {metrics['file_count']:,}{' (list truncated by GitHub)' if metrics['file_list_truncated'] else ''}  |  "
    f"Created: {metrics['created_at']}  |  Topics: {', '.join(metrics['topics']) or 'none'}"
)

left, right = st.columns(2)
with left:
    st.markdown("**Languages (% of code)**")
    if metrics["languages_percent"]:
        top = dict(list(metrics["languages_percent"].items())[:8])
        st.bar_chart(pd.DataFrame({"percent": top}), horizontal=True)
    else:
        st.write("No language data.")
with right:
    st.markdown("**Health checks**")
    st.dataframe(
        pd.DataFrame(
            {"Check": list(metrics["checks"]), "Found": ["Yes" if v else "No" for v in metrics["checks"].values()]}
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(f"Package files: {', '.join(metrics['package_files']) or 'none at root'}")

# ------------------------------------------------------------ AI review
st.divider()
st.subheader("AI review")

analysis = None
if state.analyses >= MAX_ANALYSES_PER_SESSION and not llm_key:
    st.warning(
        f"This demo allows {MAX_ANALYSES_PER_SESSION} AI reviews per session on the shared key. "
        "Add your own free Groq key in the sidebar to keep going."
    )
elif not (llm_key or get_api_key()):
    st.warning("No LLM API key configured, so only the facts above are shown. Add a Groq key in the sidebar.")
else:
    try:
        llm_client = make_client(llm_key or None)
        with st.spinner("Writing the review..."):
            analysis = run_analysis(
                metrics["full_name"].lower(),
                meta.get("pushed_at", ""),
                model_name(),
                snap,
                metrics,
                lambda m: chat(llm_client, m, response_format={"type": "json_object"}),
            )
        state.analyses += 1
    except (LLMError, AnalysisError) as exc:
        st.error(f"The AI review failed: {exc}")

if analysis:
    st.markdown("**Summary**")
    st.write(analysis["summary"])
    st.markdown("**Tech stack:** " + ", ".join(analysis["tech_stack"]))
    st.markdown("**Architecture and code organisation**")
    st.write(analysis["architecture"])

    a, b = st.columns(2)
    with a:
        st.markdown("**Strengths**")
        for s in analysis["strengths"]:
            st.markdown(f"- {s}")
    with b:
        st.markdown("**Where a new contributor could help**")
        for item in analysis["contribution_opportunities"]:
            st.markdown(f"- {item}")

    st.markdown("**Recommendations**")
    for r in analysis["recommendations"]:
        st.markdown(f"- **{r['title']}** ({r['priority']} priority): {r['detail']}")

if snap.good_first_issues:
    st.markdown("**Open 'good first issue' issues**")
    for issue in snap.good_first_issues:
        st.markdown(f"- [{issue['title']}]({issue['url']}) ({issue['comments']} comments)")

with st.expander(f"Top-level files and folders ({len(metrics['top_level'])})"):
    st.code("\n".join(metrics["top_level"]) or "(empty repository)")

if analysis:
    st.download_button(
        "Download report (Markdown)",
        to_markdown(metrics, analysis, snap.good_first_issues),
        file_name=f"{snap.owner}-{snap.name}-analysis.md",
        mime="text/markdown",
    )

if client.rate_limit_remaining is not None:
    st.sidebar.caption(f"GitHub API calls left this hour: {client.rate_limit_remaining}")
