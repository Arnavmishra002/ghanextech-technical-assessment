"""Read PDF bytes and return the text of each page.

Every failure is turned into a DocumentError with a message that can be shown
to the user as-is, so the UI never has to deal with raw pypdf exceptions.
"""

from dataclasses import dataclass
from io import BytesIO
import logging

from pypdf import PdfReader
from pypdf.errors import FileNotDecryptedError, PdfReadError, PyPdfError

# pypdf logs a warning for every small defect it repairs; that is noise here.
logging.getLogger("pypdf").setLevel(logging.ERROR)


class DocumentError(Exception):
    """A problem with an uploaded file that the user should be told about."""


@dataclass
class Page:
    source: str  # file name
    page_number: int  # 1-based, as a person would count
    text: str


def _clean(text: str) -> str:
    # Join words that were hyphenated across a line break, and drop the
    # trailing spaces pypdf leaves at the end of lines.
    text = text.replace("-\n", "")
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def extract_pages(data: bytes, filename: str) -> list[Page]:
    """Return the non-empty pages of a PDF, or raise DocumentError."""
    if not filename.lower().endswith(".pdf") or not data.startswith(b"%PDF"):
        raise DocumentError(f"{filename}: not a PDF file. Only PDF files are supported.")

    try:
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted:
            # Some PDFs are "encrypted" with an empty password only to block
            # editing. Those can still be read; real passwords cannot.
            try:
                if not reader.decrypt(""):
                    raise DocumentError(
                        f"{filename}: this PDF is password-protected. "
                        "Remove the password and upload it again."
                    )
            except (FileNotDecryptedError, NotImplementedError) as exc:
                raise DocumentError(
                    f"{filename}: this PDF is password-protected. "
                    "Remove the password and upload it again."
                ) from exc

        pages = []
        for index, page in enumerate(reader.pages, start=1):
            text = _clean(page.extract_text() or "")
            if text:
                pages.append(Page(source=filename, page_number=index, text=text))
    except DocumentError:
        raise
    except FileNotDecryptedError as exc:
        raise DocumentError(f"{filename}: this PDF is password-protected.") from exc
    except (PdfReadError, PyPdfError, ValueError, KeyError) as exc:
        raise DocumentError(f"{filename}: the file looks damaged and could not be read.") from exc

    if not pages:
        raise DocumentError(
            f"{filename}: no text found. It may be a scanned image or an empty PDF. "
            "This app does not do OCR, so please upload a PDF with selectable text."
        )
    return pages
