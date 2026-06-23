"""Job-description query text (shared by precompute A10 and Phase-B retrieval).

Kept in its own module — with no heavy deps — so the timed ``rank.py`` path can
reconstruct the BM25 query text without importing the embedder/torch. The same
text is embedded once in Phase A (``jd_embedding.npy``) for the S1 semantic side.
"""

from __future__ import annotations

from aptus.config import S2_CFG

# Must-have JD signals (from job_description.docx / config) appended to the anchors
# so the single JD vector + BM25 query capture the role's core.
_MUST_HAVES = (
    "Senior AI Engineer building production retrieval ranking and recommendation "
    "systems, embeddings, vector search, strong Python, product company experience, "
    "ranking evaluation NDCG MAP."
)


def build_jd_query() -> str:
    """Return the JD query text (anchors + must-haves), deterministic, model-free."""
    anchors = " ".join(S2_CFG["anchors"])
    return f"{anchors} {_MUST_HAVES}"
