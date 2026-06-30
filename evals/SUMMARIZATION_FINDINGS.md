# Summarization Scoring — Root Cause & Fix

## Observed symptom

Cold-run scores for the summarization task were unexpectedly low:

| Provider | Raw score (before fix) | Score (after fix) |
|----------|------------------------|-------------------|
| Groq     | 0.333                  | 0.917             |
| Ollama   | 0.208                  | 0.875             |

Both models produced clinically accurate 3-sentence summaries covering all three required
elements, yet scored only 0.2–0.3 out of 1.0.

## Root cause — rubric label mismatch

The rubric for all 8 summarization examples is:

```
Summary must mention: chief complaint, current medications, and follow-up plan.
```

The scorer (`harness/scorer.py: _score_rubric_keywords`) extracts the three phrases after
`"must mention:"` and does exact substring matching against model output.

The keywords extracted are abstract category labels:

- `"chief complaint"`
- `"current medications"`
- `"follow-up plan"`

Both models write the *content* without ever using the *label*. Examples from the cold run:

| Keyword           | Model writes instead                                               |
|-------------------|--------------------------------------------------------------------|
| chief complaint   | "presents with persistent chest tightness and shortness of breath" |
| current medications | "Her current medications include metformin, lisinopril…"         |
| follow-up plan    | "follow up in 2 weeks to monitor her blood pressure"               |

`"current medications"` matched ~60 % of the time because models happen to use that exact
phrase. `"chief complaint"` almost never matched because models naturally write the actual
symptom. `"follow-up plan"` almost never matched because models write "follow up in N weeks".

## Fix applied

Added a `_RUBRIC_PARAPHRASES` mapping in `harness/scorer.py` that expands each rubric
category term to a list of clinically equivalent phrases before matching:

```python
_RUBRIC_PARAPHRASES = {
    "chief complaint":       ["chief complaint", "presents with", "presented with", "cc:", "complains of", ...],
    "current medications":   ["current medications", "medications include", "is currently taking", ...],
    "follow-up plan":        ["follow-up plan", "follow up", "follow-up", "recheck", "return in", ...],
    ...
}
```

The scorer now passes if the output contains *any* phrase in the paraphrase set for that
keyword. This matches the clinical writing style produced by both llama3.2:3b (Ollama) and
llama-3.1-8b-instant (Groq) without penalizing fluent prose.

## Remaining misses after fix

Two examples still score below 1.0:

- **S002/Groq (0.67)** — mentions medications and chief complaint but uses "plan is to refer"
  instead of an explicit follow-up time frame. The "recheck" / "follow up" paraphrases do not
  match.
- **S003/Ollama, S005/Ollama, S008/Ollama (0.67)** — same pattern: the Ollama 3b model
  occasionally omits an explicit follow-up phrase and substitutes a procedural sentence
  ("manage symptoms").

These are genuine partial misses, not scorer errors, and reflect that the 3b model is slightly
less reliable at following all three structural requirements.

## Lesson for rubric design

Rubrics for open-text generation tasks should specify the *content* expected, not the
*category label*. Either:

1. Write rubrics in terms of observable output phrases, or
2. Use a reference-comparison scorer (semantic similarity, LLM-as-judge) for free-form text.

The paraphrase-expansion approach is a pragmatic middle ground that works well when the set of
acceptable phrasings is finite and domain-predictable.
