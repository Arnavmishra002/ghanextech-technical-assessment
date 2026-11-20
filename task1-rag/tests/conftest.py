import hashlib
import re
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from pypdf import PdfWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_pdf(pages: list[str]) -> bytes:
    """Build a small, valid text PDF in memory (one string per page)."""

    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        None,  # pages object, filled in below
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    kids = []
    for text in pages:
        lines = " T* ".join(f"({esc(line)}) Tj" for line in text.split("\n"))
        stream = f"BT /F1 11 Tf 14 TL 50 750 Td {lines} ET"
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
        content_id = len(objects)
        objects.append(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
        )
        kids.append(f"{len(objects)} 0 R")
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"

    out = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


def encrypt_pdf(data: bytes, user_password: str) -> bytes:
    writer = PdfWriter(clone_from=BytesIO(data))
    writer.encrypt(user_password=user_password, owner_password="owner-secret", algorithm="AES-128")
    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()


def blank_pdf(n_pages: int = 2) -> bytes:
    writer = PdfWriter()
    for _ in range(n_pages):
        writer.add_blank_page(width=612, height=792)
    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()


class FakeEmbedder:
    """Deterministic bag-of-words embedder, so unit tests need no model download."""

    dimension = 256

    def embed(self, texts):
        vectors = np.zeros((len(texts), self.dimension), dtype="float32")
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                if len(word) < 3:
                    continue
                bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dimension
                vectors[row, bucket] += 1.0
        return vectors


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()
