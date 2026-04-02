"""
ai_doctor.py
────────────
NVIDIA LLaMA3-70B-Instruct integration.
Provides bilingual (English + Urdu) AI medical explanations.
"""

from typing import Dict, Any, List


SYSTEM_PROMPT = """You are a compassionate, expert AI medical assistant named MedAI Doctor.
Your role is to explain medical diagnoses clearly and kindly to patients.

IMPORTANT RULES:
1. Always recommend the patient consult a real, qualified doctor for actual treatment.
2. Explain in simple, non-technical language that anyone can understand.
3. Be empathetic and reassuring — do not cause unnecessary alarm.
4. Provide practical, actionable lifestyle advice.
5. Never prescribe dosages — only mention medicine names to ask the doctor about.

RESPONSE FORMAT (strict JSON):
{
  "english": "<2-3 paragraphs explaining the diagnosis in plain English>",
  "urdu": "<Urdu translation of the key points, 3-5 sentences>",
  "next_steps": ["step 1", "step 2", "step 3"],
  "medicines": ["medicine 1 (ask your doctor)", "medicine 2"],
  "lifestyle": ["tip 1", "tip 2", "tip 3"]
}

Respond ONLY with valid JSON. No markdown, no preamble."""


def _build_prompt(disease: str, risk_result: Dict, report_result: Dict, patient_info: Dict) -> str:
    """Build the prompt for the LLM."""
    score = risk_result.get("risk_score", 0)
    level = risk_result.get("risk_level", "UNKNOWN")
    age   = patient_info.get("age", "unknown")
    gender = patient_info.get("gender", "unknown")
    symptoms = ", ".join(patient_info.get("symptoms", [])) or "none reported"
    duration = patient_info.get("duration", "unknown")

    report_summary = report_result.get("summary", "") if report_result else ""
    lab_vals = ""
    if report_result and report_result.get("lab_values"):
        lab_vals = "; ".join([f"{k}={v}" for k, v in report_result["lab_values"].items()])

    return f"""Patient profile:
- Age: {age}, Gender: {gender}
- Symptom duration: {duration}
- Reported symptoms: {symptoms}
- AI-detected condition: {disease}
- Risk score: {score:.1f}% ({level} RISK)
{f'- Lab values: {lab_vals}' if lab_vals else ''}
{f'- Report summary: {report_summary}' if report_summary else ''}

Please explain this diagnosis to the patient in English and Urdu, provide next steps, relevant medicines to ask their doctor about, and lifestyle suggestions. Remember to remind them to see a real doctor."""


def _parse_response(raw: str) -> Dict:
    """Parse JSON response from LLM, with fallback."""
    import json, re

    # Strip markdown fences if present
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to extract JSON object
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass

    # Fallback: return raw text in english field
    return {
        "english":    raw.strip(),
        "urdu":       "",
        "next_steps": [],
        "medicines":  [],
        "lifestyle":  [],
    }


def get_diagnosis(
    disease: str,
    risk_result: Dict,
    report_result: Dict,
    patient_info: Dict,
    api_key: str = "",
) -> Dict:
    """
    Call NVIDIA LLaMA3 API to get a diagnosis explanation.
    Falls back to a rule-based response if API is unavailable.
    """
    prompt = _build_prompt(disease, risk_result, report_result, patient_info)

    if api_key:
        try:
            from openai import OpenAI

            client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=api_key,
            )

            full_response = ""
            completion = client.chat.completions.create(
                model="meta/llama-3.3-70b-instruct",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.6,
                top_p=0.7,
                max_tokens=1024,
                stream=True,
            )

            for chunk in completion:
                delta = chunk.choices[0].delta.content
                if delta is not None:
                    full_response += delta

            return _parse_response(full_response)

        except Exception as e:
            # Fall through to rule-based
            pass

    # ── Rule-based fallback ────────────────────────────────────────────────────
    return _rule_based_response(disease, risk_result, patient_info)


def _rule_based_response(disease: str, risk_result: Dict, patient_info: Dict) -> Dict:
    """Generate a canned but personalised response without the API."""
    level  = risk_result.get("risk_level", "UNKNOWN")
    score  = risk_result.get("risk_score", 0)
    age    = patient_info.get("age", 20)
    gender = patient_info.get("gender", "the patient")

    sev_map = {
        "LOW":      "a mild concern",
        "MEDIUM":   "a moderate concern requiring attention",
        "HIGH":     "a significant medical concern",
        "CRITICAL": "a critical condition requiring urgent care",
    }
    sev_phrase = sev_map.get(level, "a medical condition")

    english = (
        f"Based on the AI analysis, the detected condition is **{disease}**, "
        f"which has been assessed as {sev_phrase} with a risk score of {score:.0f}%.\n\n"
        f"The analysis considered your age ({age}), gender ({gender}), reported symptoms, "
        f"and symptom duration. While AI tools can help identify patterns, "
        f"they are not a replacement for professional medical diagnosis.\n\n"
        f"It is strongly recommended that you consult a qualified physician promptly, "
        f"especially given the {level.lower()} risk level indicated by this assessment."
    )

    urdu = (
        f"اے آئی تجزیے کے مطابق، آپ کی حالت **{disease}** ہے، "
        f"جس کا خطرے کا اسکور {score:.0f}٪ ہے۔ "
        f"یہ ایک {sev_phrase} ہے۔ "
        f"براہ کرم جلد از جلد کسی مستند ڈاکٹر سے رجوع کریں۔ "
        f"اے آئی ڈاکٹر کا متبادل نہیں ہے۔"
    )

    next_steps = [
        f"Schedule an appointment with your physician within {'24 hours' if level == 'CRITICAL' else '1-2 weeks'}.",
        "Bring this AI report to your doctor's visit.",
        "Monitor your symptoms and note any changes.",
        "Avoid self-medicating without medical supervision.",
    ]

    medicines = [
        "Please consult your doctor for appropriate medications.",
        "Do not take any new medications without a prescription.",
    ]

    lifestyle = [
        "Maintain adequate hydration (8+ glasses of water per day).",
        "Ensure sufficient sleep (7-9 hours per night).",
        "Avoid strenuous activity until cleared by a doctor.",
        "Maintain a balanced, nutritious diet.",
        "Reduce stress through gentle activities like walking or meditation.",
    ]

    return {
        "english":    english,
        "urdu":       urdu,
        "next_steps": next_steps,
        "medicines":  medicines,
        "lifestyle":  lifestyle,
    }


def chat_with_doctor(
    question: str,
    history: List[Dict],
    context: Dict,
    api_key: str = "",
) -> str:
    """
    Chat interface: answer a follow-up medical question.
    """
    vision  = context.get("vision")  or {}
    risk    = context.get("risk")    or {}
    patient = context.get("patient") or {}

    disease = vision.get("disease", "the diagnosed condition")
    score   = risk.get("risk_score", 0)
    level   = risk.get("risk_level", "unknown")

    system = f"""You are MedAI Doctor, a compassionate AI medical assistant.
Context: Patient has been diagnosed with '{disease}' (risk score: {score:.0f}%, {level} risk).
Patient age: {patient.get('age','unknown')}, gender: {patient.get('gender','unknown')}.
Symptoms: {', '.join(patient.get('symptoms', [])) or 'none reported'}.

Answer the patient's question clearly and kindly in plain English.
Always remind them that AI advice does not replace a real doctor.
Keep responses concise (3-5 sentences)."""

    messages = [{"role": "system", "content": system}]
    for h in history[-8:]:  # keep last 8 turns
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": question})

    if api_key:
        try:
            from openai import OpenAI

            client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=api_key,
            )

            response = ""
            completion = client.chat.completions.create(
                model="meta/llama-3.3-70b-instruct",
                messages=messages,
                temperature=0.6,
                top_p=0.7,
                max_tokens=512,
                stream=True,
            )
            for chunk in completion:
                delta = chunk.choices[0].delta.content
                if delta is not None:
                    response += delta
            return response.strip()

        except Exception as e:
            return f"⚠️ API error: {e}. Please verify your NVIDIA API key in the sidebar."

    # Offline fallback
    faq = {
        "medication": "Please discuss medication options with your doctor, who can prescribe the most appropriate treatment for your specific condition.",
        "treatment":  "Treatment depends on the severity and your medical history. Your doctor will outline the best treatment plan after a physical examination.",
        "diet":       "Generally, a balanced diet rich in fruits, vegetables, lean proteins, and whole grains supports recovery and immune function.",
        "exercise":   "Light to moderate exercise (like walking) is often beneficial, but avoid strenuous activity until your doctor gives clearance.",
        "contagious": "Please consult your doctor to understand the transmission risk of your specific condition.",
        "prognosis":  f"With proper medical care, many conditions like {disease} can be managed effectively. Your doctor will provide a personalised prognosis.",
    }

    q_lower = question.lower()
    for key, answer in faq.items():
        if key in q_lower:
            return answer

    return (
        f"Thank you for your question about {disease}. "
        f"While I'm currently offline (no API key), I recommend discussing this with your doctor, "
        f"who can provide personalised guidance based on your full medical history. "
        f"Add your NVIDIA API key in the sidebar for live AI responses."
    )
