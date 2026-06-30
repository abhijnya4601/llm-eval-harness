import pytest
from harness.scorer import score, _score_rubric_keywords, _score_extraction, _score_classification, _parse_json_from_text


# ── Rubric keyword scorer ─────────────────────────────────────────────────────

RUBRIC = "Summary must mention: chief complaint, current medications, and follow-up plan."


def test_rubric_all_keywords_present():
    output = "Patient presents with chest pain (chief complaint). Current medications include metformin. Follow-up plan is in 2 weeks."
    result = _score_rubric_keywords(output, RUBRIC)
    assert result.score == 1.0
    assert result.method == "rubric_keyword"


def test_rubric_partial_keywords():
    # Contains "chief complaint" and "current medications" but not "follow-up plan"
    output = "Chief complaint is chest pain. Current medications include metformin."
    result = _score_rubric_keywords(output, RUBRIC)
    assert 0 < result.score < 1.0


def test_rubric_no_keywords():
    output = "The sky is blue."
    result = _score_rubric_keywords(output, RUBRIC)
    assert result.score == 0.0


def test_rubric_case_insensitive():
    output = "CHIEF COMPLAINT is hypertension. CURRENT MEDICATIONS: lisinopril. FOLLOW-UP PLAN: 2 weeks."
    result = _score_rubric_keywords(output, RUBRIC)
    assert result.score == 1.0


# ── Extraction / JSON scorer ──────────────────────────────────────────────────

REFERENCE_JSON = '{"diagnosis": "pneumonia", "medications": ["amoxicillin"], "follow_up_date": "July 15", "allergies": "penicillin"}'
RUBRIC_JSON = "Required fields: diagnosis, medications, follow_up_date, allergies."


def test_extraction_perfect_match():
    output = '{"diagnosis": "pneumonia", "medications": ["amoxicillin"], "follow_up_date": "July 15", "allergies": "penicillin"}'
    result = _score_extraction(output, REFERENCE_JSON, RUBRIC_JSON)
    assert result.score == 1.0
    assert result.method == "rubric_json"


def test_extraction_partial_match():
    output = '{"diagnosis": "pneumonia", "medications": [], "follow_up_date": "unknown", "allergies": "none"}'
    result = _score_extraction(output, REFERENCE_JSON, RUBRIC_JSON)
    assert 0 < result.score < 1.0


def test_extraction_no_json_in_output():
    result = _score_extraction("I cannot find any fields.", REFERENCE_JSON, RUBRIC_JSON)
    assert result.score == 0.0


def test_extraction_json_in_code_fence():
    output = '```json\n{"diagnosis": "pneumonia", "medications": ["amoxicillin"], "follow_up_date": "July 15", "allergies": "penicillin"}\n```'
    result = _score_extraction(output, REFERENCE_JSON, RUBRIC_JSON)
    assert result.score == 1.0


def test_extraction_fallback_when_reference_not_json():
    result = _score_extraction("Patient has pneumonia", "pneumonia", RUBRIC_JSON)
    # Falls back to keyword scorer — shouldn't crash
    assert 0.0 <= result.score <= 1.0


# ── Classification scorer ─────────────────────────────────────────────────────

def test_classification_exact_match():
    result = _score_classification("urgent", "urgent", "Classify as: urgent, routine, or informational.")
    assert result.score == 1.0


def test_classification_label_in_sentence():
    result = _score_classification("This case is urgent and requires immediate attention.", "urgent", "")
    assert result.score == 1.0


def test_classification_wrong_label():
    result = _score_classification("routine", "urgent", "")
    assert result.score == 0.0


def test_classification_paraphrase_gets_partial_credit():
    result = _score_classification("This is high priority and needs immediate attention.", "urgent", "")
    assert result.score == 0.8


# ── JSON parser utility ───────────────────────────────────────────────────────

def test_parse_json_direct():
    assert _parse_json_from_text('{"a": 1}') == {"a": 1}


def test_parse_json_code_fence():
    text = '```json\n{"a": 1}\n```'
    assert _parse_json_from_text(text) == {"a": 1}


def test_parse_json_bare_in_prose():
    text = 'Here is the output: {"a": 1} — done.'
    assert _parse_json_from_text(text) == {"a": 1}


def test_parse_json_invalid_returns_none():
    assert _parse_json_from_text("not json at all") is None


# ── Top-level dispatch ────────────────────────────────────────────────────────

def test_score_dispatches_summarization():
    result = score("summarization", "Chief complaint is headache. Current medications: aspirin. Follow-up plan: 2 weeks.", "", RUBRIC)
    assert result.method == "rubric_keyword"


def test_score_dispatches_extraction():
    result = score("extraction", '{"diagnosis": "flu"}', '{"diagnosis": "flu", "medications": [], "follow_up_date": "unknown", "allergies": "none"}', RUBRIC_JSON)
    assert result.method == "rubric_json"


def test_score_dispatches_classification():
    result = score("classification", "routine", "routine", "")
    assert result.method == "rubric_keyword"
    assert result.score == 1.0
