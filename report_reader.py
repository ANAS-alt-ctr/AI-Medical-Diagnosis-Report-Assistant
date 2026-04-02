"""
report_reader.py
────────────────
spaCy + pdfplumber based PDF medical report extractor.
Identifies diseases, medications, lab values, and symptoms.
"""

import re
from pathlib import Path
from typing import Dict, List, Any


# ── Medical patterns (used by both spaCy rules and regex fallback) ────────────

DISEASE_PATTERNS = [
    r"\b(?:diabetes|hypertension|pneumonia|tuberculosis|TB|asthma|COPD|"
    r"hepatitis|HIV|AIDS|malaria|dengue|typhoid|cholera|cancer|carcinoma|"
    r"lymphoma|leukemia|anemia|arrhythmia|angina|myocardial infarction|"
    r"stroke|epilepsy|Parkinson|Alzheimer|lupus|arthritis|osteoporosis|"
    r"psoriasis|eczema|dermatitis|hypothyroidism|hyperthyroidism|"
    r"gastritis|colitis|Crohn|appendicitis|hernia|fibrosis|edema|"
    r"pneumothorax|pleural effusion|atelectasis|consolidation|"
    r"cardiomegaly|emphysema|nodule|infiltration|mass|COVID-19|"
    r"coronary artery disease|heart failure|renal failure|CKD)\b",
]

MEDICATION_PATTERNS = [
    r"\b(?:metformin|insulin|amlodipine|atenolol|lisinopril|atorvastatin|"
    r"aspirin|clopidogrel|warfarin|heparin|amoxicillin|ciprofloxacin|"
    r"azithromycin|doxycycline|metronidazole|fluconazole|paracetamol|"
    r"ibuprofen|naproxen|omeprazole|pantoprazole|ranitidine|salbutamol|"
    r"prednisolone|dexamethasone|hydrocortisone|levothyroxine|"
    r"carbamazepine|phenytoin|valproate|sertraline|fluoxetine|"
    r"diazepam|lorazepam|morphine|tramadol|codeine|furosemide|"
    r"spironolactone|hydrochlorothiazide|digoxin|amiodarone)\b",
]

LAB_PATTERNS = {
    "HbA1c":     r"HbA1c\s*[:\=]?\s*(\d+\.?\d*)\s*%?",
    "Blood Glucose":    r"(?:blood\s+)?glucose\s*[:\=]?\s*(\d+\.?\d*)\s*(?:mg/dL|mmol/L)?",
    "Hemoglobin":       r"hemoglobin\s*[:\=]?\s*(\d+\.?\d*)\s*(?:g/dL)?",
    "WBC":              r"WBC\s*[:\=]?\s*(\d+\.?\d*)\s*(?:×10³/μL|K/μL)?",
    "Creatinine":       r"creatinine\s*[:\=]?\s*(\d+\.?\d*)\s*(?:mg/dL)?",
    "eGFR":             r"eGFR\s*[:\=]?\s*(\d+\.?\d*)",
    "Cholesterol":      r"(?:total\s+)?cholesterol\s*[:\=]?\s*(\d+\.?\d*)\s*(?:mg/dL)?",
    "LDL":              r"LDL\s*[:\=]?\s*(\d+\.?\d*)\s*(?:mg/dL)?",
    "HDL":              r"HDL\s*[:\=]?\s*(\d+\.?\d*)\s*(?:mg/dL)?",
    "Triglycerides":    r"triglycerides?\s*[:\=]?\s*(\d+\.?\d*)\s*(?:mg/dL)?",
    "Blood Pressure":   r"BP\s*[:\=]?\s*(\d+/\d+)\s*(?:mmHg)?",
    "SpO2":             r"SpO2\s*[:\=]?\s*(\d+\.?\d*)\s*%?",
    "Heart Rate":       r"(?:HR|heart\s+rate)\s*[:\=]?\s*(\d+)\s*(?:bpm)?",
}

SYMPTOM_PATTERNS = [
    r"\b(?:fever|cough|dyspnea|shortness of breath|chest pain|fatigue|"
    r"headache|nausea|vomiting|diarrhea|constipation|weight loss|"
    r"weight gain|night sweats|chills|dizziness|palpitations|"
    r"edema|swelling|rash|itching|pain|weakness|confusion|"
    r"syncope|malaise|anorexia|hemoptysis|hematuria|jaundice)\b",
]


def _extract_with_spacy(text: str) -> Dict[str, Any]:
    """Use spaCy NER with custom medical patterns."""
    try:
        import spacy
        from spacy.matcher import Matcher

        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Model not found, try to download and load
            import subprocess
            import sys
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
            nlp = spacy.load("en_core_web_sm")

        doc = nlp(text[:100_000])  # limit tokens

        # Rule-based matcher for medical terms
        matcher = Matcher(nlp.vocab)

        # Add disease patterns
        disease_words = [
            "diabetes","hypertension","pneumonia","tuberculosis","asthma",
            "COPD","hepatitis","cancer","carcinoma","lymphoma","leukemia",
            "anemia","stroke","epilepsy","lupus","arthritis","osteoporosis",
            "psoriasis","eczema","fibrosis","edema","pneumothorax","consolidation",
            "cardiomegaly","emphysema","atelectasis","COVID-19","heart failure",
        ]
        matcher.add("DISEASE", [
            [{"LOWER": w.lower()}] for w in disease_words
        ])

        matches = matcher(doc)
        detected_diseases = list({
            doc[start:end].text.title() for _, start, end in matches
        })

        # Named entities for drugs / medications
        meds_from_ner = [
            ent.text for ent in doc.ents
            if ent.label_ in ("CHEMICAL", "PRODUCT")
        ]

        return {
            "spacy_diseases": detected_diseases,
            "spacy_meds":     meds_from_ner,
        }
    except Exception as e:
        return {"spacy_diseases": [], "spacy_meds": [], "spacy_error": str(e)}


def _extract_with_regex(text: str) -> Dict[str, Any]:
    """Fallback regex extraction when spaCy is unavailable."""
    text_lower = text.lower()

    diseases = []
    for pattern in DISEASE_PATTERNS:
        found = re.findall(pattern, text, re.IGNORECASE)
        diseases.extend(found)
    diseases = list({d.strip().title() for d in diseases})

    medications = []
    for pattern in MEDICATION_PATTERNS:
        found = re.findall(pattern, text, re.IGNORECASE)
        medications.extend(found)
    medications = list({m.strip().title() for m in medications})

    symptoms = []
    for pattern in SYMPTOM_PATTERNS:
        found = re.findall(pattern, text, re.IGNORECASE)
        symptoms.extend(found)
    symptoms = list({s.strip().title() for s in symptoms})

    lab_values = {}
    for key, pattern in LAB_PATTERNS.items():
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            lab_values[key] = m.group(1)

    return {
        "diseases":    diseases,
        "medications": medications,
        "symptoms":    symptoms,
        "lab_values":  lab_values,
    }


def _generate_summary(diseases, medications, lab_values, symptoms, raw_text: str) -> str:
    """Generate a brief plain-language summary of the report."""
    parts = []

    if diseases:
        parts.append(f"The report identifies the following condition(s): {', '.join(diseases[:4])}.")
    if medications:
        parts.append(f"Mentioned medication(s): {', '.join(medications[:4])}.")
    if lab_values:
        lab_str = "; ".join([f"{k}: {v}" for k, v in list(lab_values.items())[:5]])
        parts.append(f"Key lab values — {lab_str}.")
    if symptoms:
        parts.append(f"Reported symptom(s): {', '.join(symptoms[:5])}.")

    if not parts:
        # Summarise first 400 chars of raw text
        excerpt = raw_text[:400].replace("\n", " ").strip()
        parts.append(f"Report excerpt: {excerpt}…")

    return " ".join(parts)


def extract_report(pdf_path: str) -> Dict[str, Any]:
    """
    Extract medical entities from a PDF report.

    Returns:
        {
            "diseases":    list[str],
            "medications": list[str],
            "symptoms":    list[str],
            "lab_values":  dict[str, str],
            "summary":     str,
            "raw_text":    str,
            "pages":       int,
        }
    """
    result = {
        "diseases":    [],
        "medications": [],
        "symptoms":    [],
        "lab_values":  {},
        "summary":     "",
        "raw_text":    "",
        "pages":       0,
    }

    # ── Extract text with pdfplumber ──────────────────────────────────────────
    raw_text = ""
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            result["pages"] = len(pdf.pages)
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                raw_text += page_text + "\n"
    except ImportError:
        result["error"] = "pdfplumber not installed"
        return result
    except Exception as e:
        result["error"] = f"PDF read error: {e}"
        return result

    result["raw_text"] = raw_text

    # ── Entity extraction ─────────────────────────────────────────────────────
    regex_result = _extract_with_regex(raw_text)
    spacy_result = _extract_with_spacy(raw_text)

    # Merge results (spaCy + regex)
    all_diseases = list({
        *regex_result["diseases"],
        *spacy_result.get("spacy_diseases", [])
    })
    all_meds = list({
        *regex_result["medications"],
        *spacy_result.get("spacy_meds", [])
    })

    result["diseases"]    = all_diseases[:10]
    result["medications"] = all_meds[:10]
    result["symptoms"]    = regex_result["symptoms"][:10]
    result["lab_values"]  = regex_result["lab_values"]

    result["summary"] = _generate_summary(
        result["diseases"], result["medications"],
        result["lab_values"], result["symptoms"], raw_text
    )

    return result
