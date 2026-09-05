"""Small shared helpers for app.py (Stage 8)."""


def extract_text(content) -> str:
    """Normalize a LangChain message's .content into a plain string.

    Same helper as Event-Planner-AI/utils.py: usually .content is already a
    string, but Gemini's "thinking" models can return a list of content
    blocks instead, e.g. [{"type": "text", "text": "...", "extras": {...}}].
    This pulls out just the text."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
        return "\n".join(parts)

    return str(content)


# Transformed feature names (e.g. "nom__InternetService_Fiber optic",
# "num__tenure") -> a label a non-technical reader can understand. Falls back
# to a light auto-cleanup for anything not explicitly listed.
FEATURE_LABELS = {
    "num__tenure": "How long they've been a customer",
    "num__MonthlyCharges": "Their monthly bill",
    "num__TotalCharges": "Total amount billed to date",
    "ord__Contract": "Contract commitment (month-to-month vs. 1-2 year)",
    "remainder__gender": "Gender",
    "remainder__SeniorCitizen": "Senior citizen status",
    "remainder__Partner": "Has a partner",
    "remainder__Dependents": "Has dependents",
    "remainder__PhoneService": "Has phone service",
    "remainder__PaperlessBilling": "Paperless billing",
}


def humanize_feature_name(name: str) -> str:
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    # e.g. "nom__InternetService_Fiber optic" -> "Internet service: Fiber optic"
    stripped = name.split("__", 1)[-1]
    if "_" in stripped:
        field, _, value = stripped.rpartition("_")
        return f"{field}: {value}"
    return stripped
