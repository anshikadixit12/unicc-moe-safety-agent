# core/pdf_utils.py
# ─────────────────────────────────────────────────────────────────────────────
# Extracts plain text from an uploaded PDF file using PyMuPDF (fitz).
# Returns a clean string ready to pass to the router and experts.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import io


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Given raw PDF bytes (from FastAPI's UploadFile.read()),
    returns extracted plain text.
    Raises ValueError if the file is not a valid PDF or is empty.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF not installed. Run: pip install PyMuPDF")

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        raise ValueError(f"Could not open PDF: {e}")

    if doc.page_count == 0:
        raise ValueError("PDF has no pages.")

    pages_text = []
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            pages_text.append(f"[Page {page_num + 1}]\n{text.strip()}")

    doc.close()

    full_text = "\n\n".join(pages_text)

    if not full_text.strip():
        raise ValueError("PDF appears to be scanned/image-only. No extractable text found.")

    # Truncate to 12,000 chars to stay within LLM context limits
    if len(full_text) > 12000:
        full_text = full_text[:12000] + "\n\n[... truncated for LLM context limit ...]"

    return full_text


def build_evaluation_text(pdf_text: str | None, prompt: str | None, ai_response: str | None) -> str:
    """
    Combine whichever inputs the user provided into one text block
    that gets passed to the router and all experts.
    """
    parts = []

    if pdf_text:
        parts.append(f"=== DOCUMENT CONTENT ===\n{pdf_text}")

    if prompt:
        parts.append(f"=== USER PROMPT ===\n{prompt}")

    if ai_response:
        parts.append(f"=== AI SYSTEM RESPONSE ===\n{ai_response}")

    if not parts:
        raise ValueError("No input provided. Submit a PDF, a prompt, or a prompt+response pair.")

    return "\n\n".join(parts)
