# Ghanex Tech – GenAI & Open-Source Developer Intern Assessment

This repository contains the technical assessment for the **GenAI and Open-Source Developer Intern** role at [Ghanex Tech](https://ghanextech.com/).

---

## About the Role

**Position:** GenAI & Open-Source Developer Intern  
**Duration:** 3 months  
**Mode:** Fully remote  
**Stipend:** Performance-based (discussed during the selection process)  
**Experience level:** Freshers are welcome

---

## Purpose of This Assessment

We use this assessment to identify candidates who can build, understand, debug, and deploy real software — not just generate code with AI tools.

You will work with Python, LLMs, vector search, REST APIs, and GitHub. The tasks are intentionally practical: each one maps to work you would actually do during the internship.

We want to see how you structure a project, handle edge cases, write readable code, and communicate your design decisions. AI coding tools are allowed, but you must own and understand everything you submit.

---

## Assessment Structure

The assessment consists of three independent tasks. All three must be completed and deployed within **3 to 5 days**.

| Task | Description | File |
|------|-------------|------|
| Task 1 | AI Document Assistant using Retrieval-Augmented Generation | [TASK-01-RAG.md](./TASK-01-RAG.md) |
| Task 2 | AI Agent with Tool Calling | [TASK-02-AI-AGENT.md](./TASK-02-AI-AGENT.md) |
| Task 3 | GitHub Repository AI Analyzer + Open-Source Contribution | [TASK-03-GITHUB-ANALYZER.md](./TASK-03-GITHUB-ANALYZER.md) |

Read each task file carefully before beginning.

---

## Technology

There is no mandatory stack. The suggestions below are based on what works well and what is free to use.

- **Language:** Python 3.10+
- **LLMs:** Google Gemini API (free tier), Groq API (free tier), or locally hosted models via Ollama
- **Frameworks:** LangChain, LlamaIndex, or plain Python
- **Vector databases:** FAISS, ChromaDB
- **UI:** Streamlit, Gradio, or FastAPI
- **APIs:** GitHub REST API, Open-Meteo, DuckDuckGo Search, Wikipedia API

None of the tasks require paid APIs or paid hosting. If you choose a service with a free tier, document any rate limits or constraints.

---

## Deployment

Deploy each project on a free hosting platform:

- [Streamlit Community Cloud](https://streamlit.io/cloud)
- [Hugging Face Spaces](https://huggingface.co/spaces)
- Any other free platform you prefer

Each deployed app must be accessible via a public URL. If a deployment is temporarily unavailable, document the issue and include clear local setup instructions so the application can be verified locally.

**Do not commit API keys, tokens, or any secrets to your repository.** Use environment variables or the secrets management provided by your hosting platform.

---

## AI Tool Policy

You may use AI assistants (ChatGPT, Claude, Gemini, Copilot, Cursor, or others) during this assessment. Use them as development aids — for looking up syntax, drafting initial code, or debugging.

Before submitting:

- Review and test every part of the code yourself.
- Refactor AI-generated code to fit a coherent structure and style.
- Write your own documentation that accurately describes what the application does.
- Remove generic boilerplate, placeholder comments, and anything you do not understand.

Shortlisted candidates will be asked to explain their implementation, make a small modification, or debug a specific part during the interview. If you cannot walk through your own submission, that is a problem.

Please disclose which AI tools you used and how in the submission form.

---

## GitHub Instructions

1. **Fork this repository** to your personal GitHub account.
2. Create one repository per task, or organize all three in a single well-structured repository.
3. Make regular commits with clear, descriptive commit messages throughout development — not one large commit at the end.
4. Keep all repositories **public** until your evaluation is complete.

Do not push directly to this repository.

---

## Submission

Fill out the submission template in [SUBMISSION.md](./SUBMISSION.md) and send it as instructed by the recruitment team.

Submissions with broken links, inaccessible repositories, or incomplete sections may not be considered for further evaluation.

---

## Questions

If something in the assessment is unclear, raise a GitHub issue in this repository or contact the team through the official recruitment channel.

Good luck.
