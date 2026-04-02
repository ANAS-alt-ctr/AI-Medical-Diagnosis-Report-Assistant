# 🧬 MedAI — AI Medical Diagnosis Assistant

> AI-powered medical analysis · YOLOv11 · spaCy NLP · XGBoost · NVIDIA LLaMA3 · Bilingual Urdu/English

---

## ✨ Features

| Module | Technology | What it does |
|---|---|---|
| 🩻 Computer Vision | YOLOv11 (Ultralytics) | Detects diseases in X-ray & skin images |
| 📄 NLP Report Reader | spaCy + pdfplumber | Extracts conditions, meds & lab values from PDFs |
| 🤖 Risk Prediction | XGBoost + SHAP | Predicts 0-100% risk score with explainability |
| 💬 AI Doctor | NVIDIA LLaMA3-70B | Bilingual English/Urdu diagnosis explanations |
| 📊 Dashboard | Plotly | Interactive health charts & gauges |

---

## 🚀 Quick Start

### 1 — Clone & Install

```bash
git clone <repo-url>
cd medical-ai-assistant
pip install -r requirements.txt
# Note: spaCy model is now automatically handled via requirements.txt
```

### 2 — Get Your NVIDIA API Key (Free)

1. Visit [build.nvidia.com](https://build.nvidia.com)
2. Sign up → Generate API key
3. Copy the `nvapi-...` key

### 3 — Train the Risk Model (Optional)

```bash
# Uses synthetic data if data/symptoms_data.csv is not present
python train_model.py
```

### 4 — Run the App

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 📁 File Structure

```
medical-ai-assistant/
├── app.py              ← Streamlit UI (dark theme)
├── vision_model.py     ← YOLOv11 image scanner
├── report_reader.py    ← spaCy PDF text extractor
├── risk_predictor.py   ← XGBoost risk prediction
├── ai_doctor.py        ← NVIDIA LLaMA3 AI doctor
├── dashboard.py        ← Plotly health charts
├── train_model.py      ← Train & save XGBoost model
├── data/
│   └── symptoms_data.csv   ← Auto-generated if absent
├── models/
│   └── risk_model.pkl      ← Saved after training
└── requirements.txt
```

---

## 🔄 App Flow

```
Upload Image/PDF
      ↓
YOLOv11 Detection (image) | spaCy NLP (PDF)
      ↓
XGBoost Risk Prediction (0–100%)
      ↓
NVIDIA LLaMA3 — Bilingual AI Explanation
      ↓
Plotly Dashboard + AI Doctor Chat
```

---

## 🎨 UI Design

- **Background**: `#080C10` (deep dark)
- **Accent**: `#00D4FF` (medical cyan)
- **Success**: `#00FF88`
- **Danger**: `#FF4444`
- **Font**: Syne (display) + Space Mono (data)
- Fully mobile responsive

---

## 🔑 Environment Variables (optional)

Create a `.env` file:

```
NVIDIA_API_KEY=nvapi-your-key-here
```

Or enter the key directly in the sidebar at runtime.

---

## ⚠️ Medical Disclaimer

> This application is for **educational and research purposes only**.  
> It does **NOT** replace professional medical diagnosis or treatment.  
> Always consult a qualified healthcare provider.

---

## 📦 Datasets (Kaggle)

| Dataset | Purpose |
|---|---|
| NIH Chest X-ray 14 | Chest disease detection training |
| ISIC Skin Lesion | Skin disease detection training |
| Disease Symptom Dataset | XGBoost training |

---

## 🏆 Tech Stack

- **Streamlit** 1.32 + custom CSS
- **Ultralytics YOLOv11**
- **spaCy** 3.7 + en_core_web_sm
- **XGBoost** 2.0 + SHAP
- **NVIDIA API** (LLaMA3-70B-Instruct)
- **Plotly** 5.20
- **pdfplumber** 0.11
- **Pandas / NumPy / Pillow**
