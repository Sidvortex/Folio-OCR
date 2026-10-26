"""Folio-OCR: a page parser fine-tuned from Infinity-Parser2-Flash.

Output format (shared with Infinity-Parser2's ``doc2json`` task, so the base
model needs no new format): a JSON list of layout blocks in reading order,

    [{"bbox": [x1, y1, x2, y2], "category": "text", "text": "..."}, ...]

with boxes normalised to 0-1000 and text as Markdown (HTML for tables, LaTeX
for formulas). Markdown output is derived from it by ``schema.to_markdown``.
"""

__version__ = "0.1.0"
