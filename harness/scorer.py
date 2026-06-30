import json
import re
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ScoreResult:
    score: float          # 0.0 – 1.0
    max_score: float      # always 1.0 for rubric mode
    method: str           # "rubric_keyword" | "rubric_json" | "llm_judge"
    detail: str           # human-readable breakdown


def score(task: str, output: str, reference: str, rubric: str) -> ScoreResult:
    """Dispatch to the appropriate scorer based on task type."""
    if task == "classification":
        return _score_classification(output, reference, rubric)
    if task == "extraction":
        return _score_extraction(output, reference, rubric)
    # Default: summarization or anything else → keyword rubric
    return _score_rubric_keywords(output, rubric)


# ── Summarization: keyword / phrase presence ──────────────────────────────────

def _score_rubric_keywords(output: str, rubric: str) -> ScoreResult:
    """Score by checking how many rubric keywords/phrases appear in the output.

    The rubric field should list required elements after "must mention:" or
    as a comma-separated list. Each present element adds equal weight.
    """
    keywords = _extract_rubric_keywords(rubric)
    if not keywords:
        return ScoreResult(score=0.0, max_score=1.0, method="rubric_keyword",
                           detail="No keywords found in rubric.")

    output_lower = output.lower()
    hits = [kw for kw in keywords if kw.lower() in output_lower]
    score_val = len(hits) / len(keywords)

    detail = (
        f"{len(hits)}/{len(keywords)} keywords present. "
        f"Hit: {hits}. "
        f"Missed: {[kw for kw in keywords if kw not in hits]}"
    )
    return ScoreResult(score=score_val, max_score=1.0, method="rubric_keyword", detail=detail)


def _extract_rubric_keywords(rubric: str) -> list[str]:
    """Pull keywords from rubric strings like 'must mention: X, Y, and Z'."""
    lower = rubric.lower()
    for marker in ("must mention:", "must include:", "should contain:", "required fields:"):
        idx = lower.find(marker)
        if idx != -1:
            rest = rubric[idx + len(marker):]
            # Strip trailing periods / parens, split on commas and "and"
            rest = re.sub(r"[.()\[\]]", "", rest)
            parts = re.split(r",|\band\b", rest, flags=re.IGNORECASE)
            return [p.strip().strip('"').strip("'") for p in parts if p.strip()]

    # Fallback: treat the whole rubric as a comma list
    parts = re.split(r",|\band\b", rubric, flags=re.IGNORECASE)
    return [p.strip() for p in parts if len(p.strip()) > 2]


# ── Extraction: JSON field presence + value match ─────────────────────────────

def _score_extraction(output: str, reference: str, rubric: str) -> ScoreResult:
    """Score JSON extraction by comparing fields to a reference JSON object.

    Partial credit: each correct field is worth 1/N of the total score.
    Values are compared case-insensitively as strings (flexible for dates, etc.).
    """
    pred = _parse_json_from_text(output)
    ref = _parse_json_from_text(reference)

    if ref is None:
        logger.warning("Reference is not valid JSON for extraction task; falling back to keyword scorer.")
        return _score_rubric_keywords(output, rubric)

    if pred is None:
        return ScoreResult(
            score=0.0, max_score=1.0, method="rubric_json",
            detail="Model output did not contain parseable JSON."
        )

    fields = list(ref.keys())
    if not fields:
        return ScoreResult(score=1.0, max_score=1.0, method="rubric_json", detail="No fields to check.")

    hits, misses = [], []
    for field in fields:
        ref_val = str(ref.get(field, "")).lower().strip()
        pred_val = str(pred.get(field, "")).lower().strip()
        if pred_val and ref_val and (ref_val in pred_val or pred_val in ref_val):
            hits.append(field)
        else:
            misses.append(field)

    score_val = len(hits) / len(fields)
    detail = f"{len(hits)}/{len(fields)} fields correct. Hit: {hits}. Missed: {misses}."
    return ScoreResult(score=score_val, max_score=1.0, method="rubric_json", detail=detail)


def _parse_json_from_text(text: str) -> dict | None:
    """Extract and parse a JSON object from text, tolerating markdown code fences."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from ```json ... ``` block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding a bare { ... } block
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


# ── Classification: exact or near-exact label match ──────────────────────────

def _score_classification(output: str, reference: str, rubric: str) -> ScoreResult:
    """Score classification by checking if the expected label appears in output."""
    expected = reference.strip().lower()
    output_lower = output.lower()

    if expected in output_lower:
        return ScoreResult(score=1.0, max_score=1.0, method="rubric_keyword",
                           detail=f"Label '{expected}' found in output.")

    # Check for common paraphrases (e.g. "urgent" ↔ "high priority")
    paraphrases = {
        "urgent": ["urgent", "high priority", "immediately", "emergent"],
        "routine": ["routine", "non-urgent", "standard", "regular"],
        "informational": ["informational", "informative", "for information", "fyi"],
    }
    for canonical, synonyms in paraphrases.items():
        if expected == canonical:
            if any(s in output_lower for s in synonyms):
                return ScoreResult(
                    score=0.8, max_score=1.0, method="rubric_keyword",
                    detail=f"Paraphrase of '{expected}' found (0.8 partial credit)."
                )

    return ScoreResult(score=0.0, max_score=1.0, method="rubric_keyword",
                       detail=f"Label '{expected}' not found in output.")
