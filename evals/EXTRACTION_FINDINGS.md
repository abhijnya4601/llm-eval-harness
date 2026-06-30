# Extraction Scoring — Root Cause & Fix

## Observed symptom

Both providers scored 0.607 on extraction despite producing structurally correct JSON
with all four required fields present. E003 aside, every extraction output contained the
right medication names, the right allergy, and the correct follow-up date.

## Root cause — naïve string comparison of list fields

The scorer compared fields with:

```python
ref_val = str(ref.get(field, "")).lower().strip()
pred_val = str(pred.get(field, "")).lower().strip()
if ref_val in pred_val or pred_val in ref_val: ...
```

This converts Python lists to their repr string before comparing. Three failure modes:

**1. Models append dosage to medication names.**

Reference: `["amoxicillin-clavulanate", "azithromycin"]`
→ `str(...)` → `"['amoxicillin-clavulanate', 'azithromycin']"`

Model output: `["amoxicillin-clavulanate 875mg BID for 7 days", "azithromycin 500mg QD for 5 days"]`
→ `str(...)` → `"['amoxicillin-clavulanate 875mg bid for 7 days', 'azithromycin 500mg qd for 5 days']"`

Neither repr is a substring of the other → **medications field always MISS** when dosage is
included, which is almost every example for both providers.

**2. Ollama returns `{name, dosage}` objects instead of flat strings.**

Ollama's preferred medication format:
`[{"name": "amoxicillin-clavulanate", "dosage": "875mg BID"}, ...]`

Converting this to `str()` produces a Python dict repr that shares no substring with the
reference's flat list repr.

**3. `[]` / `null` / `"None Known (NKDA)"` are not treated as equivalent to `"none"`.**

Reference for E002 allergies: `"none"` → `str(...)` → `"none"`
Groq output: `[]` → `str([])` → `"[]"` → `"none"` not in `"[]"` → **MISS**
Groq output for E004: `"None Known (NKDA)"` → `str(...)` → `"none known (nkda)"` → **MISS**

## Fix applied

Replaced the naïve comparison with `_extraction_field_matches()` in `harness/scorer.py`,
backed by `_flatten_val()` and `_is_none_val()`:

- **`_flatten_val(val)`** — recursively flattens any JSON value (list, dict, scalar) to a
  single lowercase string. A `{name, dosage}` dict becomes `"amoxicillin-clavulanate 875mg bid"`.
  A flat list becomes items space-joined.

- **`_is_none_val(s)`** — returns True for `""`, `"[]"`, `"none"`, `"none known"`, `"nkda"`,
  `"null"`, and any string containing `"nkda"` or both `"none"` and `"known"`.

- **List reference matching** — for list-valued references, checks that every reference item
  appears somewhere in the flattened prediction string, rather than comparing the whole lists.

- **Comma-separated string references** — splits `"sulfa drugs, latex"` on commas and requires
  each part to appear in the prediction, so `["sulfa drugs (hives)", "latex (contact)"]` matches.

## Results after fix

| Provider | Extraction (before) | Extraction (after) |
|----------|---------------------|--------------------|
| Groq     | 0.607               | 0.964              |
| Ollama   | 0.607               | 0.964              |

The one remaining miss (E003, both providers, `follow_up_date`) is a genuine model error:

- Reference: `"unscheduled"` — the note states no date has been set
- Groq output: `null` — correct intent, wrong value for the scorer
- Ollama output: `"4 weeks from now"` — relative date instead of the reference label

This is a prompt-following gap. The input note says "patient will call to schedule in 4 weeks"
and neither model echoes the reference's label `"unscheduled"`. A rubric that accepted `null`
or relative-date strings as equivalent would score this correctly, but doing so would require
semantic judgement the current scorer intentionally avoids.

## Overall impact

| Provider | Overall (before fix) | Overall (after fix) |
|----------|----------------------|---------------------|
| Groq     | 0.743                | 0.843               |
| Ollama   | 0.730                | 0.830               |
