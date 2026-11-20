"""Retrieval check with the real embedding model.

Skipped by default because it downloads the model (about 90 MB).
Run with:  RUN_INTEGRATION=1 pytest tests/test_real_embeddings.py -v
"""

import os

import pytest

from conftest import make_pdf

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION") != "1", reason="set RUN_INTEGRATION=1 to run"
)

PAGES = [
    "Solar panels convert sunlight into electricity using photovoltaic cells.\n"
    "Most home systems produce between three and eight kilowatts.",
    "The warranty covers manufacturing defects for twenty five years.\n"
    "Damage from storms or incorrect installation is not covered.",
    "Inverters change direct current from the panels into alternating current\n"
    "that household appliances can use. A typical inverter lasts ten to fifteen years.",
    "Cleaning the panels twice a year with plain water keeps output high.\n"
    "Do not use pressure washers because they can crack the glass.",
]

QUESTIONS = [
    ("How long does the warranty last?", 2),
    ("What does an inverter do?", 3),
    ("How should I clean my panels?", 4),
    ("How much power does a home solar system make?", 1),
    ("Is hail damage covered?", 2),
]


@pytest.fixture(scope="module")
def pipeline():
    from rag.embedder import DIMENSION, Embedder
    from rag.pipeline import RAGPipeline

    p = RAGPipeline(Embedder(), DIMENSION, top_k=4)
    p.index_files([("solar_guide.pdf", make_pdf(PAGES))])
    return p


@pytest.mark.parametrize("question, expected_page", QUESTIONS)
def test_top_result_is_the_right_page(pipeline, question, expected_page):
    top = pipeline.retrieve(question)[0]
    assert top.chunk.page_number == expected_page
