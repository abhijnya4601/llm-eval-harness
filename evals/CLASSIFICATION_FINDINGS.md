# Classification Scoring — "Informational" Gap

## Observed symptom

Both providers score 0.700 on classification (7/10 correct). The three misses are C003,
C006, and C008 — every example with the reference label `"informational"`. Both models
output `"Routine"` for all three. This is not a scorer bug; it is a confirmed model
capability gap consistent across both providers and both model sizes (3B and 8B).

## The three failing examples

**C003** — Lab result notification  
> "Lab result notification: HbA1c improved from 9.1% to 7.4% since last visit. Patient doing well with dietary changes."  
> Reference: `informational` | Both models: `Routine`

**C006** — Care management status note  
> "Monthly care management note: patient with CHF has been stable, no hospitalizations this quarter, weight within 2 lbs of dry weight, compliant with fluid restriction."  
> Reference: `informational` | Both models: `Routine`

**C008** — Preventive care reminder  
> "Flu vaccine reminder: patient is due for annual influenza immunization. Provider note indicates patient is eligible this season."  
> Reference: `informational` | Both models: `Routine`

## Why this is a genuine model error

The clinical distinction between `routine` and `informational` is subtle:

- **Routine** = requires a scheduled action (a visit, a refill, a follow-up call)
- **Informational** = status update or notification; no action required from the recipient

Both Llama models treat preventive reminders and stable-patient status notes as actionable
routine items. From a general-purpose language model's perspective, "patient needs a flu
vaccine" and "patient needs a refill" look similar — both describe something that needs
doing. The difference is whether the recipient of the message needs to *act* or just *be
aware*.

Neither model was trained on a clinical workflow ontology that distinguishes these categories.
Consistent failure across all three examples and across both model sizes (3B and 8B Llama
variants) suggests this is a systematic judgement gap rather than a borderline case.

## Why the scorer was not adjusted

The classification scorer has a paraphrase fallback for 0.8 partial credit. "Informational"
synonyms (`"informative"`, `"for information"`, `"fyi"`) were already included and were not
triggered — the models did not use any of those forms either. Expanding the paraphrase set
further would not help because the models are not producing wrong synonyms for
"informational"; they are producing the correct label for the wrong category.

Adjusting the rubric or scorer to award partial credit for "Routine" on these examples would
obscure a real failure mode. The 0.700 classification score is an accurate signal.

## Implications

In a production clinical triage system, misclassifying informational notifications as routine
tasks creates unnecessary work queue pollution — care managers would see stable-patient
status notes mixed in with actionable items. These small models (3B–8B parameters) are not
reliable for fine-grained clinical workflow routing without additional fine-tuning on
domain-specific triage ontologies or few-shot prompting with explicit label definitions.

## Potential mitigations (not applied)

1. **Add label definitions to the prompt.** Explicitly telling the model "informational means
   no action is required" in the system prompt may reduce the confusion between routine and
   informational. Not implemented here — the current prompt intentionally tests zero-shot
   capability.

2. **Few-shot examples.** Providing one `informational` example in the prompt would likely
   help both models calibrate on the distinction. This would be the lowest-cost fix.

3. **Fine-tuning or a larger model.** Models with stronger instruction-following tend to
   respect multi-class definitions more precisely.
