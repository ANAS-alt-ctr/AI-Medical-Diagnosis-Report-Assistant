"""
dashboard.py
────────────
Plotly chart factory for the MedAI health dashboard.
All charts use a dark medical theme matching the Streamlit UI.
"""

from typing import Dict, List, Any, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px

# ── Shared theme ──────────────────────────────────────────────────────────────

DARK_BG    = "#080C10"
CARD_BG    = "#111820"
ACCENT     = "#00D4FF"
ACCENT2    = "#00FF88"
DANGER     = "#FF4444"
WARNING    = "#FFB830"
TEXT_MAIN  = "#E8EDF3"
TEXT_MUTED = "#6B7A8D"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=CARD_BG,
    plot_bgcolor=CARD_BG,
    font=dict(family="Syne, sans-serif", color=TEXT_MAIN),
    margin=dict(l=20, r=20, t=40, b=20),
)


def _risk_color(level: str) -> str:
    return {
        "LOW":      ACCENT2,
        "MEDIUM":   WARNING,
        "HIGH":     "#FF643C",
        "CRITICAL": DANGER,
    }.get(level, ACCENT)


# ═════════════════════════════════════════════════════════════════════════════
# 1. RISK GAUGE
# ═════════════════════════════════════════════════════════════════════════════

def render_risk_gauge(risk_score: float, risk_level: str) -> go.Figure:
    """Gauge chart showing 0-100 risk score."""
    color = _risk_color(risk_level)

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score,
        number={"suffix": "%", "font": {"size": 36, "color": color, "family": "Space Mono"}},
        delta={"reference": 50, "increasing": {"color": DANGER}, "decreasing": {"color": ACCENT2}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickcolor": TEXT_MUTED,
                "tickfont": {"size": 11, "color": TEXT_MUTED},
                "nticks": 6,
            },
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": DARK_BG,
            "borderwidth": 0,
            "steps": [
                {"range": [0,  25], "color": "rgba(0,255,136,0.12)"},
                {"range": [25, 50], "color": "rgba(255,184,48,0.10)"},
                {"range": [50, 75], "color": "rgba(255,100,60,0.10)"},
                {"range": [75,100], "color": "rgba(255,68,68,0.12)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.8,
                "value": risk_score,
            },
        },
        title={
            "text": f"<b>RISK SCORE</b><br><span style='font-size:14px;color:{color}'>{risk_level} RISK</span>",
            "font": {"size": 16, "color": TEXT_MAIN},
        },
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=300,
        title_text="",
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 2. SYMPTOM SEVERITY BAR CHART
# ═════════════════════════════════════════════════════════════════════════════

SYMPTOM_SEVERITY = {
    "Fever":              0.70,
    "Cough":              0.45,
    "Shortness Of Breath":0.80,
    "Chest Pain":         0.85,
    "Fatigue":            0.40,
    "Headache":           0.35,
    "Nausea":             0.30,
    "Skin Rash":          0.50,
    "Joint Pain":         0.55,
    "Weight Loss":        0.65,
    "Night Sweats":       0.60,
    "Dizziness":          0.45,
}


def render_symptom_chart(symptoms: List[str]) -> go.Figure:
    """Horizontal bar chart of symptom severities."""
    if not symptoms:
        symptoms = ["No symptoms reported"]

    labels = [s.title() for s in symptoms]
    values = [SYMPTOM_SEVERITY.get(s.title(), 0.3) * 100 for s in symptoms]
    colors = [
        DANGER if v >= 75 else WARNING if v >= 50 else ACCENT
        for v in values
    ]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker=dict(
            color=colors,
            opacity=0.85,
            line=dict(width=0),
        ),
        text=[f"{v:.0f}%" for v in values],
        textposition="outside",
        textfont=dict(size=11, color=TEXT_MUTED),
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=max(220, 60 * len(labels)),
        title_text="<b>Symptom Severity</b>",
        title_font=dict(size=14, color=TEXT_MAIN),
        xaxis=dict(
            range=[0, 110],
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
            ticksuffix="%",
            tickfont=dict(color=TEXT_MUTED, size=11),
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(color=TEXT_MAIN, size=12),
        ),
        bargap=0.3,
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 3. DISEASE PROBABILITY PIE CHART
# ═════════════════════════════════════════════════════════════════════════════

def render_disease_chart(vision_result: Optional[Dict], report_result: Optional[Dict]) -> go.Figure:
    """Donut chart showing detected disease distribution."""
    labels = []
    values = []

    if vision_result and vision_result.get("all_detections"):
        for det in vision_result["all_detections"][:5]:
            labels.append(det.get("disease", "Unknown"))
            values.append(max(det.get("confidence", 0) * 100, 0.5))

    if not labels and vision_result and vision_result.get("disease"):
        d = vision_result.get("disease", "Unknown")
        c = vision_result.get("confidence", 0.5) * 100
        labels = [d, "Other Conditions"]
        values = [c, max(100 - c, 1)]

    if not labels and report_result and report_result.get("diseases"):
        total = len(report_result["diseases"])
        for i, disease in enumerate(report_result["diseases"][:5]):
            labels.append(disease)
            values.append(max(100 - i * 15, 5))

    if not labels:
        labels = ["No Detection"]
        values = [100]

    colors = [
        DANGER, WARNING, ACCENT, ACCENT2,
        "#9B59B6", "#3498DB", "#E74C3C", "#2ECC71"
    ][:len(labels)]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(
            colors=colors,
            line=dict(color=CARD_BG, width=2),
        ),
        textinfo="percent+label",
        textfont=dict(size=11, color=TEXT_MAIN),
        insidetextorientation="radial",
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=300,
        title_text="<b>Detection Confidence</b>",
        title_font=dict(size=14, color=TEXT_MAIN),
        showlegend=False,
        annotations=[dict(
            text="Diagnosis",
            x=0.5, y=0.5,
            font_size=13,
            font_color=TEXT_MUTED,
            showarrow=False,
        )],
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 4. SHAP FEATURE IMPORTANCE CHART
# ═════════════════════════════════════════════════════════════════════════════

def render_shap_chart(top_factors: List[Tuple[str, float]]) -> go.Figure:
    """Waterfall-style SHAP contribution chart."""
    if not top_factors:
        top_factors = [("No data", 0.0)]

    labels = [f[0] for f in top_factors]
    values = [f[1] for f in top_factors]

    bar_colors = [
        DANGER if v > 0 else ACCENT2 for v in values
    ]

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker=dict(
            color=bar_colors,
            opacity=0.85,
            line=dict(width=0),
        ),
        text=[f"{'+' if v > 0 else ''}{v:.3f}" for v in values],
        textposition="outside",
        textfont=dict(size=11, color=TEXT_MUTED),
        width=0.5,
    ))

    fig.add_hline(y=0, line_color=TEXT_MUTED, line_width=1, opacity=0.5)

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=300,
        title_text="<b>Risk Factor Contribution (SHAP)</b>",
        title_font=dict(size=14, color=TEXT_MAIN),
        xaxis=dict(
            tickfont=dict(color=TEXT_MAIN, size=12),
            showgrid=False,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
            tickfont=dict(color=TEXT_MUTED, size=11),
            zeroline=False,
        ),
        bargap=0.3,
    )

    # Annotation legend
    fig.add_annotation(
        text="■ Increases risk  ■ Decreases risk",
        xref="paper", yref="paper",
        x=0.5, y=-0.12,
        showarrow=False,
        font=dict(size=10, color=TEXT_MUTED),
        align="center",
    )

    return fig
