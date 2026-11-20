from conftest import FakeEmbedder, make_pdf
from rag.pipeline import NOT_FOUND, RAGPipeline, build_messages

HANDBOOK = make_pdf(
    [
        "Employee handbook. Office hours are nine to five on weekdays.",
        "Leave policy. Every employee gets twenty paid leave days per year.",
        "Security. Laptops must be locked when you leave your desk.",
    ]
)
MENU = make_pdf(["Cafeteria menu. Paneer tikka is served on Fridays."])


def new_pipeline():
    return RAGPipeline(FakeEmbedder(), FakeEmbedder.dimension, top_k=2)


def test_index_report_counts_files_pages_and_chunks():
    p = new_pipeline()
    report = p.index_files([("handbook.pdf", HANDBOOK), ("menu.pdf", MENU)])
    assert report.files_indexed == ["handbook.pdf", "menu.pdf"]
    assert report.pages == 4
    assert report.chunks == 4
    assert report.errors == []


def test_bad_file_is_reported_and_good_file_still_indexed():
    p = new_pipeline()
    report = p.index_files([("notes.txt", b"hello"), ("handbook.pdf", HANDBOOK)])
    assert report.files_indexed == ["handbook.pdf"]
    assert len(report.errors) == 1 and "notes.txt" in report.errors[0]


def test_retrieves_the_relevant_page():
    p = new_pipeline()
    p.index_files([("handbook.pdf", HANDBOOK)])
    top = p.retrieve("How many paid leave days does an employee get?")[0]
    assert top.chunk.page_number == 2


def test_prompt_contains_numbered_passages_with_page_refs():
    p = new_pipeline()
    p.index_files([("handbook.pdf", HANDBOOK)])
    messages = build_messages("leave days?", p.retrieve("leave days"))
    assert messages[0]["role"] == "system" and "Use only" in messages[0]["content"]
    assert "[1] (handbook.pdf, page 2)" in messages[1]["content"]
    assert "Question: leave days?" in messages[1]["content"]


def test_answer_returns_llm_text_and_sources():
    p = new_pipeline()
    p.index_files([("handbook.pdf", HANDBOOK)])
    seen = {}

    def fake_llm(messages):
        seen["messages"] = messages
        return "Employees get twenty paid leave days per year [1]."

    result = p.answer("How many leave days?", fake_llm)
    assert result.text.startswith("Employees get twenty")
    assert result.sources and result.sources[0].chunk.page_number == 2
    assert "twenty paid leave days" in seen["messages"][1]["content"]


def test_not_found_answer_drops_sources():
    p = new_pipeline()
    p.index_files([("handbook.pdf", HANDBOOK)])
    result = p.answer("What is the CEO's salary?", lambda m: NOT_FOUND)
    assert result.text == NOT_FOUND
    assert result.sources == []


def test_reindexing_replaces_old_documents():
    p = new_pipeline()
    p.index_files([("handbook.pdf", HANDBOOK)])
    p.index_files([("menu.pdf", MENU)])
    sources = {r.chunk.source for r in p.retrieve("leave policy office hours paneer")}
    assert sources == {"menu.pdf"}


def test_empty_index_answers_not_found_without_calling_llm():
    p = new_pipeline()

    def llm_must_not_run(messages):
        raise AssertionError("LLM should not be called")

    assert p.answer("anything?", llm_must_not_run).text == NOT_FOUND
