"""
generate_report.py
Generates a personalized ADHD assessment PDF report for each user.
Place this file next to views.py in your Django app directory.

Usage in views.py:
    from .generate_report import generate_adhd_report
    pdf_bytes = generate_adhd_report(user_data, game_data, prediction)
    response = HttpResponse(pdf_bytes, content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="adhd_report.pdf"'
    return response
"""

import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    Table, TableStyle, HRFlowable, PageBreak
)
from reportlab.graphics.shapes import Drawing, Rect, String
from datetime import datetime


# ── Colour palette ──────────────────────────────────────────────────────────
BLUE        = colors.HexColor("#3B82F6")
LIGHT_BLUE  = colors.HexColor("#EFF6FF")
RED         = colors.HexColor("#EF4444")
LIGHT_RED   = colors.HexColor("#FEF2F2")
GREEN       = colors.HexColor("#22C55E")
LIGHT_GREEN = colors.HexColor("#F0FDF4")
ORANGE      = colors.HexColor("#F97316")
PURPLE      = colors.HexColor("#8B5CF6")
DARK        = colors.HexColor("#1E293B")
GRAY        = colors.HexColor("#64748B")
LIGHT_GRAY  = colors.HexColor("#F1F5F9")
WHITE       = colors.white


# ── Thresholds used to interpret each feature ───────────────────────────────
THRESHOLDS = {
    "InattentionScore":    {"low": 3,  "high": 6,  "max": 9,  "label": "Inattention"},
    "ImpulsivityScore":    {"low": 3,  "high": 6,  "max": 9,  "label": "Impulsivity"},
    "HyperactivityScore":  {"low": 3,  "high": 6,  "max": 9,  "label": "Hyperactivity"},
    "Daydreaming":         {"low": 1,  "high": 2,  "max": 3,  "label": "Daydreaming"},
    "RSD":                 {"low": 1,  "high": 2,  "max": 3,  "label": "Rejection Sensitivity"},
    "SleepHours":          {"low": 6,  "high": 9,  "max": 12, "label": "Sleep Hours", "invert": True},
    "ScreenTime":          {"low": 2,  "high": 5,  "max": 12, "label": "Screen Time (hrs)"},
    "AcademicScore":       {"low": 40, "high": 70, "max": 100,"label": "Academic Score",    "invert": True},
    "ComorbidAnxiety":     {"low": 0,  "high": 0,  "max": 1,  "label": "Anxiety"},
    "ComorbidDepression":  {"low": 0,  "high": 0,  "max": 1,  "label": "Depression"},
    "FamilyHistoryADHD":   {"low": 0,  "high": 0,  "max": 1,  "label": "Family History"},
}

SCORE_KEYS = ["InattentionScore", "ImpulsivityScore", "HyperactivityScore",
              "Daydreaming", "RSD"]
LIFESTYLE_KEYS = ["SleepHours", "ScreenTime", "AcademicScore"]
BINARY_KEYS = ["ComorbidAnxiety", "ComorbidDepression", "FamilyHistoryADHD"]


def _buf_to_rl_image(fig, width_cm=14):
    """Convert a matplotlib figure to a ReportLab Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img = Image(buf)
    aspect = img.imageHeight / img.imageWidth
    w = width_cm * cm
    img.drawWidth  = w
    img.drawHeight = w * aspect
    return img


def _risk_color(value, threshold):
    """Return a matplotlib hex colour based on risk level."""
    t = THRESHOLDS[threshold]
    invert = t.get("invert", False)
    low, high = t["low"], t["high"]
    if invert:
        # higher value = lower risk
        if value >= high:   return "#22C55E"
        if value >= low:    return "#F97316"
        return "#EF4444"
    else:
        if value <= low:    return "#22C55E"
        if value <= high:   return "#F97316"
        return "#EF4444"


# ── Chart generators ─────────────────────────────────────────────────────────

def _chart_symptom_scores(user_data):
    """Horizontal bar chart: core ADHD symptom scores."""
    keys   = SCORE_KEYS
    labels = [THRESHOLDS[k]["label"] for k in keys]
    values = [user_data.get(k, 0) for k in keys]
    maxes  = [THRESHOLDS[k]["max"]  for k in keys]
    bar_colors = [_risk_color(v, k) for v, k in zip(values, keys)]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    y = np.arange(len(labels))
    # background (max) bars
    ax.barh(y, maxes, color="#E2E8F0", height=0.5, zorder=1)
    # value bars
    ax.barh(y, values, color=bar_colors, height=0.5, zorder=2)

    for i, (v, m) in enumerate(zip(values, maxes)):
        ax.text(m + 0.1, i, f"{v}/{m}", va='center', fontsize=9, color="#1E293B")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(0, max(maxes) + 1.5)
    ax.set_xlabel("Score", fontsize=9)
    ax.set_title("Core ADHD Symptom Scores", fontsize=12, fontweight='bold', pad=10)
    ax.spines[['top','right','left']].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')

    patches = [
        mpatches.Patch(color="#22C55E", label="Low risk"),
        mpatches.Patch(color="#F97316", label="Moderate risk"),
        mpatches.Patch(color="#EF4444", label="High risk"),
    ]
    ax.legend(handles=patches, loc='lower right', fontsize=8, framealpha=0.7)
    fig.tight_layout()
    return _buf_to_rl_image(fig)


def _chart_lifestyle(user_data):
    """Grouped bar chart for lifestyle indicators."""
    keys   = LIFESTYLE_KEYS
    labels = [THRESHOLDS[k]["label"] for k in keys]
    values = [user_data.get(k, 0) for k in keys]
    maxes  = [THRESHOLDS[k]["max"]  for k in keys]
    bar_colors = [_risk_color(v, k) for v, k in zip(values, keys)]

    fig, axes = plt.subplots(1, 3, figsize=(8, 3))
    for i, (ax, label, value, max_val, color) in enumerate(
            zip(axes, labels, values, maxes, bar_colors)):
        ax.bar([label], [value], color=color, width=0.4)
        ax.bar([label], [max_val], color="#E2E8F0", width=0.4, zorder=0)
        ax.set_ylim(0, max_val * 1.25)
        ax.set_title(label, fontsize=9, fontweight='bold')
        ax.text(0, value + max_val * 0.02, f"{value}", ha='center', fontsize=11, fontweight='bold')
        ax.spines[['top','right','left']].set_visible(False)
        ax.set_xticks([])

    fig.suptitle("Lifestyle Indicators", fontsize=12, fontweight='bold')
    fig.tight_layout()
    return _buf_to_rl_image(fig, width_cm=14)


def _chart_go_nogo(game_data):
    """
    Visualise Go/No-Go game results.
    game_data keys:
        total_trials, correct_go, missed_go, correct_inhibit,
        commission_errors (clicked on No-Go), reaction_times (list of ms values)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

    # ── Pie: accuracy breakdown ──────────────────────────────────────────────
    labels  = ['Correct Go', 'Missed Go', 'Correct Inhibit', 'Commission Errors']
    sizes   = [
        game_data.get('correct_go', 0),
        game_data.get('missed_go', 0),
        game_data.get('correct_inhibit', 0),
        game_data.get('commission_errors', 0),
    ]
    clrs    = ['#22C55E', '#F97316', '#3B82F6', '#EF4444']
    explode = [0, 0.05, 0, 0.05]
    wedges, texts, autotexts = ax1.pie(
        sizes, labels=labels, colors=clrs, explode=explode,
        autopct=lambda p: f'{p:.0f}%' if p > 0 else '',
        startangle=90, textprops={'fontsize': 8}
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax1.set_title("Response Accuracy", fontsize=11, fontweight='bold')

    # ── Histogram: reaction times ────────────────────────────────────────────
    rts = game_data.get('reaction_times', [])
    if rts:
        ax2.hist(rts, bins=min(15, len(rts)), color='#3B82F6', edgecolor='white', alpha=0.85)
        ax2.axvline(np.mean(rts), color='#EF4444', linewidth=1.5,
                    linestyle='--', label=f'Mean: {np.mean(rts):.0f} ms')
        ax2.legend(fontsize=8)
    else:
        ax2.text(0.5, 0.5, 'No reaction time\ndata', ha='center', va='center',
                 transform=ax2.transAxes, fontsize=10, color='gray')

    ax2.set_xlabel("Reaction Time (ms)", fontsize=9)
    ax2.set_ylabel("Count", fontsize=9)
    ax2.set_title("Reaction Time Distribution", fontsize=11, fontweight='bold')
    ax2.spines[['top','right']].set_visible(False)

    fig.suptitle("Go / No-Go Game Performance", fontsize=12, fontweight='bold')
    fig.tight_layout()
    return _buf_to_rl_image(fig, width_cm=15)


def _chart_radar(user_data):
    """Radar chart comparing user to 'typical ADHD' profile."""
    cats   = ["Inattention", "Impulsivity", "Hyperactivity", "Daydreaming", "RSD"]
    keys   = ["InattentionScore", "ImpulsivityScore", "HyperactivityScore", "Daydreaming", "RSD"]
    maxes  = [THRESHOLDS[k]["max"] for k in keys]
    user_v = [user_data.get(k, 0) / m for k, m in zip(keys, maxes)]
    adhd_v = [0.78, 0.72, 0.80, 0.75, 0.70]  # reference ADHD profile (normalised)

    N = len(cats)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    user_v  += [user_v[0]]
    adhd_v  += [adhd_v[0]]
    angles  += [angles[0]]
    cat_angles = angles[:-1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.plot(angles, user_v, 'o-', linewidth=2, color='#3B82F6', label='You')
    ax.fill(angles, user_v, alpha=0.2, color='#3B82F6')
    ax.plot(angles, adhd_v, 's--', linewidth=1.5, color='#EF4444', label='Typical ADHD')
    ax.fill(angles, adhd_v, alpha=0.1, color='#EF4444')

    ax.set_xticks(cat_angles)
    ax.set_xticklabels(cats, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=7)
    ax.set_title("Your Profile vs Typical ADHD", fontsize=11, fontweight='bold', pad=18)
    ax.legend(loc='lower right', fontsize=9, bbox_to_anchor=(1.35, -0.05))
    fig.tight_layout()
    return _buf_to_rl_image(fig, width_cm=10)


def _chart_binary_factors(user_data):
    """Simple visual for binary (yes/no) risk factors."""
    keys   = BINARY_KEYS
    labels = [THRESHOLDS[k]["label"] for k in keys]
    values = [user_data.get(k, 0) for k in keys]

    fig, ax = plt.subplots(figsize=(6, 2))
    y = np.arange(len(labels))
    bar_colors = ["#EF4444" if v else "#22C55E" for v in values]
    ax.barh(y, [1] * len(labels), color="#E2E8F0", height=0.4)
    ax.barh(y, values, color=bar_colors, height=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(0, 1.4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No', 'Yes'])
    ax.set_title("Risk Factors Present", fontsize=11, fontweight='bold', pad=8)
    ax.spines[['top','right','left']].set_visible(False)

    for i, v in enumerate(values):
        ax.text(1.05, i, "YES" if v else "NO",
                va='center', fontsize=10, fontweight='bold',
                color="#EF4444" if v else "#22C55E")
    fig.tight_layout()
    return _buf_to_rl_image(fig, width_cm=10)


# ── Text helpers ─────────────────────────────────────────────────────────────

def _symptom_interpretation(user_data, prediction, game_data):
    """Return a list of (heading, body) tuples explaining the result."""
    inatt  = user_data.get("InattentionScore", 0)
    impuls = user_data.get("ImpulsivityScore", 0)
    hyper  = user_data.get("HyperactivityScore", 0)
    rsd    = user_data.get("RSD", 0)
    day    = user_data.get("Daydreaming", 0)
    sleep  = user_data.get("SleepHours", 0)
    screen = user_data.get("ScreenTime", 0)
    acad   = user_data.get("AcademicScore", 0)
    anxiety = user_data.get("ComorbidAnxiety", 0)
    depression = user_data.get("ComorbidDepression", 0)
    family  = user_data.get("FamilyHistoryADHD", 0)

    commission = game_data.get("commission_errors", 0)
    total      = game_data.get("total_trials", 1)
    missed_go  = game_data.get("missed_go", 0)
    rts        = game_data.get("reaction_times", [])
    mean_rt    = int(np.mean(rts)) if rts else None
    commission_rate = commission / total if total else 0
    missed_rate     = missed_go / total if total else 0

    items = []

    # Core symptoms
    sym_parts = []
    if inatt >= 6:
        sym_parts.append(f"Your inattention score ({inatt}/9) is high, suggesting significant difficulty sustaining focus.")
    elif inatt >= 3:
        sym_parts.append(f"Your inattention score ({inatt}/9) is moderate.")
    else:
        sym_parts.append(f"Your inattention score ({inatt}/9) is low, suggesting good attention control.")

    if impuls >= 6:
        sym_parts.append(f"Impulsivity ({impuls}/9) is elevated, indicating a tendency to act before thinking.")
    elif impuls >= 3:
        sym_parts.append(f"Impulsivity ({impuls}/9) is in the moderate range.")
    else:
        sym_parts.append(f"Impulsivity ({impuls}/9) is low.")

    if hyper >= 6:
        sym_parts.append(f"Hyperactivity ({hyper}/9) is high, pointing to restlessness or excessive movement.")
    elif hyper >= 3:
        sym_parts.append(f"Hyperactivity ({hyper}/9) is moderate.")
    else:
        sym_parts.append(f"Hyperactivity ({hyper}/9) is low.")

    items.append(("Core Symptom Scores", " ".join(sym_parts)))

    # Game analysis
    game_parts = []
    if commission_rate > 0.20:
        game_parts.append(
            f"You made commission errors on {commission_rate*100:.0f}% of No-Go trials — "
            f"clicking when you should have stopped. This is a key marker of impulsivity.")
    else:
        game_parts.append(
            f"Your commission error rate ({commission_rate*100:.0f}%) was within a normal range, "
            f"suggesting good response inhibition.")

    if missed_rate > 0.25:
        game_parts.append(
            f"You missed {missed_rate*100:.0f}% of Go targets, which can reflect inattention or slow processing.")
    else:
        game_parts.append(f"You responded to most Go targets correctly ({100-missed_rate*100:.0f}% hit rate).")

    if mean_rt:
        if mean_rt > 600:
            game_parts.append(f"Your average reaction time ({mean_rt} ms) was slower than typical, which may indicate sustained attention difficulties.")
        elif mean_rt < 250:
            game_parts.append(f"Your average reaction time ({mean_rt} ms) was very fast, sometimes indicating impulsive responding.")
        else:
            game_parts.append(f"Your average reaction time ({mean_rt} ms) was in a typical range.")

    items.append(("Go / No-Go Game Interpretation", " ".join(game_parts)))

    # Lifestyle
    life_parts = []
    if sleep < 6:
        life_parts.append(f"You sleep only {sleep} hours — chronic sleep deprivation can amplify ADHD-like symptoms significantly.")
    elif sleep > 9:
        life_parts.append(f"You sleep {sleep} hours, which is above average. Excessive sleep can sometimes accompany depression.")
    else:
        life_parts.append(f"Your sleep duration ({sleep} hrs) is in the healthy range.")

    if screen > 5:
        life_parts.append(f"Screen time of {screen} hrs/day is high and is associated with attention difficulties.")
    else:
        life_parts.append(f"Screen time ({screen} hrs/day) appears manageable.")

    if acad < 50:
        life_parts.append(f"An academic score of {acad} is low and may reflect learning impact from attention-related difficulties.")
    elif acad >= 70:
        life_parts.append(f"Academic score of {acad} is good, suggesting strong ability to manage tasks despite any difficulties.")
    else:
        life_parts.append(f"Academic score is average ({acad}).")

    items.append(("Lifestyle & Academic Indicators", " ".join(life_parts)))

    # Risk factors
    risk_parts = []
    if family:
        risk_parts.append("Family history of ADHD is present, which increases genetic predisposition.")
    if anxiety:
        risk_parts.append("Comorbid anxiety was reported — anxiety and ADHD frequently co-occur and can exacerbate each other.")
    if depression:
        risk_parts.append("Comorbid depression was noted — this is common in individuals with ADHD and warrants attention.")
    if not (family or anxiety or depression):
        risk_parts.append("No additional biological or psychological risk factors were reported.")

    items.append(("Additional Risk Factors", " ".join(risk_parts)))

    # Conclusion
    if prediction == 1:
        conclusion = (
            "Based on the combined analysis of your symptom scores, reaction-based game performance, "
            "lifestyle indicators, and risk factors, the model identified a pattern consistent with ADHD. "
            "This does not constitute a clinical diagnosis. Please consult a licensed psychiatrist or "
            "psychologist for a formal evaluation."
        )
    else:
        conclusion = (
            "Your overall pattern of responses did not meet the threshold the model associates with ADHD. "
            "This does not rule out ADHD or other conditions — only a qualified clinician can provide a "
            "formal assessment. If you have concerns, speaking with a healthcare professional is always a good step."
        )
    items.append(("How the Conclusion Was Reached", conclusion))
    return items


# ── Main public function ──────────────────────────────────────────────────────

def generate_adhd_report(user_data: dict, game_data: dict, prediction: int,
                         user_name: str = "User") -> bytes:
    """
    Generate a PDF report and return it as bytes.

    Parameters
    ----------
    user_data : dict
        Keys matching dataset feature names, e.g.:
        {
            'Age': 22, 'Gender': 'Male', 'EducationStage': 'University',
            'InattentionScore': 7, 'ImpulsivityScore': 5, 'HyperactivityScore': 6,
            'Daydreaming': 2, 'RSD': 2, 'SleepHours': 5.5, 'ScreenTime': 6,
            'ComorbidAnxiety': 1, 'ComorbidDepression': 0, 'FamilyHistoryADHD': 1,
            'Medication': 'No', 'SchoolSupport': 'Yes', 'AcademicScore': 55,
        }
    game_data : dict
        {
            'total_trials': 60,
            'correct_go': 30,
            'missed_go': 5,
            'correct_inhibit': 18,
            'commission_errors': 7,
            'reaction_times': [320, 410, 290, ...]   # in milliseconds
        }
    prediction : int   1 = ADHD signs detected, 0 = not detected
    user_name  : str   Name to display on the report
    """

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=1.5*cm, bottomMargin=1.5*cm,
    )

    styles = getSampleStyleSheet()
    # Custom styles
    title_style = ParagraphStyle('ReportTitle', parent=styles['Title'],
                                 fontSize=22, textColor=DARK, spaceAfter=4)
    subtitle_style = ParagraphStyle('ReportSubtitle', parent=styles['Normal'],
                                    fontSize=11, textColor=GRAY, spaceAfter=6)
    section_style = ParagraphStyle('SectionHead', parent=styles['Heading2'],
                                   fontSize=13, textColor=BLUE, spaceBefore=14, spaceAfter=4)
    body_style = ParagraphStyle('Body', parent=styles['Normal'],
                                fontSize=10, textColor=DARK, leading=16,
                                alignment=TA_JUSTIFY, spaceAfter=6)
    caption_style = ParagraphStyle('Caption', parent=styles['Normal'],
                                   fontSize=8, textColor=GRAY, alignment=TA_CENTER)
    small_bold = ParagraphStyle('SmallBold', parent=styles['Normal'],
                                fontSize=9, textColor=DARK, fontName='Helvetica-Bold')

    story = []

    # ── Header ───────────────────────────────────────────────────────────────
    result_color = RED if prediction == 1 else GREEN
    result_text  = "ADHD INDICATORS DETECTED" if prediction == 1 else "NO ADHD INDICATORS DETECTED"

    # Top banner table
    banner_data = [[
        Paragraph(f"<font size='20'><b>ADHD Assessment Report</b></font>", styles['Normal']),
        Paragraph(
            f"<font color='{'#EF4444' if prediction==1 else '#22C55E'}' size='11'><b>{result_text}</b></font><br/>"
            f"<font size='9' color='#64748B'>Generated: {datetime.now().strftime('%d %B %Y')}</font>",
            styles['Normal']
        )
    ]]
    banner_table = Table(banner_data, colWidths=[10*cm, 7*cm])
    banner_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), LIGHT_GRAY),
        ('ROWPADDING', (0,0), (-1,-1), 12),
        ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN',      (1,0), (1,0),  'RIGHT'),
        ('LEFTPADDING',(0,0), (0,-1), 14),
        ('ROUNDEDCORNERS', [6]),
    ]))
    story.append(banner_table)
    story.append(Spacer(1, 10))

    # User info row
    info_items = [
        ("Name",      user_name),
        ("Age",       str(user_data.get("Age", "—"))),
        ("Gender",    user_data.get("Gender", "—")),
        ("Education", user_data.get("EducationStage", "—")),
    ]
    info_data = [[Paragraph(f"<b>{k}</b><br/>{v}", small_bold) for k, v in info_items]]
    info_table = Table(info_data, colWidths=[4.25*cm]*4)
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), LIGHT_BLUE),
        ('ROWPADDING', (0,0), (-1,-1), 8),
        ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
        ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
        ('BOX',        (0,0), (-1,-1), 0.5, BLUE),
        ('INNERGRID',  (0,0), (-1,-1), 0.5, BLUE),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 14))

    # ── Disclaimer ───────────────────────────────────────────────────────────
    disclaimer = (
        "<b>⚠ Important Disclaimer:</b> This report is generated by a machine learning model "
        "for educational purposes only. It is <b>not</b> a clinical diagnosis. Please consult a "
        "licensed healthcare professional for a formal evaluation."
    )
    disc_table = Table([[Paragraph(disclaimer, body_style)]], colWidths=[17*cm])
    disc_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,-1), colors.HexColor("#FFFBEB")),
        ('BOX',        (0,0),(-1,-1), 0.8, colors.HexColor("#F59E0B")),
        ('ROWPADDING', (0,0),(-1,-1), 10),
    ]))
    story.append(disc_table)
    story.append(Spacer(1, 16))

    # ── Section 1: Core Symptom Scores ──────────────────────────────────────
    story.append(Paragraph("1. Core ADHD Symptom Scores", section_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BLUE, spaceAfter=8))
    story.append(_chart_symptom_scores(user_data))
    story.append(Paragraph(
        "Bars show your score vs the maximum possible. Green = low risk, Orange = moderate, Red = high.",
        caption_style))
    story.append(Spacer(1, 10))

    # ── Section 2: Go/No-Go Game ─────────────────────────────────────────────
    story.append(Paragraph("2. Go / No-Go Game Performance", section_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BLUE, spaceAfter=8))

    # Stats summary box
    commission_rate = (game_data.get('commission_errors', 0) /
                       max(game_data.get('total_trials', 1), 1) * 100)
    hit_rate = (game_data.get('correct_go', 0) /
                max(game_data.get('total_trials', 1), 1) * 100)
    rts = game_data.get('reaction_times', [])
    mean_rt = f"{int(np.mean(rts))} ms" if rts else "N/A"

    stat_data = [
        ["Total Trials", "Hit Rate", "Commission Error Rate", "Mean Reaction Time"],
        [str(game_data.get('total_trials', 0)),
         f"{hit_rate:.0f}%",
         f"{commission_rate:.0f}%",
         mean_rt]
    ]
    stat_table = Table(stat_data, colWidths=[4.25*cm]*4)
    stat_table.setStyle(TableStyle([
        ('BACKGROUND',  (0,0), (-1,0), BLUE),
        ('TEXTCOLOR',   (0,0), (-1,0), WHITE),
        ('FONTNAME',    (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',    (0,0), (-1,-1), 9),
        ('ALIGN',       (0,0), (-1,-1), 'CENTER'),
        ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
        ('ROWPADDING',  (0,0), (-1,-1), 8),
        ('BACKGROUND',  (0,1), (-1,1), LIGHT_BLUE),
        ('BOX',         (0,0), (-1,-1), 0.5, BLUE),
        ('INNERGRID',   (0,0), (-1,-1), 0.5, colors.HexColor("#BFDBFE")),
    ]))
    story.append(stat_table)
    story.append(Spacer(1, 8))
    story.append(_chart_go_nogo(game_data))
    story.append(Spacer(1, 10))

    # ── Section 3: Profile Radar + Binary Factors ────────────────────────────
    story.append(Paragraph("3. Profile Comparison & Risk Factors", section_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BLUE, spaceAfter=8))

    # Side-by-side radar + binary chart
    radar_img  = _chart_radar(user_data)
    binary_img = _chart_binary_factors(user_data)
    radar_img.drawWidth  = 8.5 * cm
    radar_img.drawHeight = 8.5 * cm
    binary_img.drawWidth  = 8 * cm
    binary_img.drawHeight = 4 * cm

    two_col = Table([[radar_img, binary_img]], colWidths=[8.7*cm, 8.3*cm])
    two_col.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN',  (0,0), (-1,-1), 'CENTER'),
    ]))
    story.append(two_col)
    story.append(Spacer(1, 10))

    # ── Section 4: Lifestyle ─────────────────────────────────────────────────
    story.append(Paragraph("4. Lifestyle Indicators", section_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BLUE, spaceAfter=8))
    story.append(_chart_lifestyle(user_data))
    story.append(Spacer(1, 10))

    # ── Section 5: Detailed Interpretation ──────────────────────────────────
    story.append(Paragraph("5. Detailed Interpretation", section_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BLUE, spaceAfter=8))

    interpretations = _symptom_interpretation(user_data, prediction, game_data)
    for heading, body in interpretations:
        is_conclusion = "Conclusion" in heading or "Reached" in heading
        bg = (LIGHT_RED if (is_conclusion and prediction == 1)
              else LIGHT_GREEN if (is_conclusion and prediction == 0)
              else LIGHT_GRAY)
        border = RED if (is_conclusion and prediction == 1) else GREEN if is_conclusion else GRAY

        interp_table = Table([
            [Paragraph(f"<b>{heading}</b>", small_bold)],
            [Paragraph(body, body_style)],
        ], colWidths=[17*cm])
        interp_table.setStyle(TableStyle([
            ('BACKGROUND',  (0,0), (-1,0), bg),
            ('ROWPADDING',  (0,0), (-1,-1), 8),
            ('BOX',         (0,0), (-1,-1), 0.8, border),
            ('LINEBELOW',   (0,0), (-1,0),  0.5, border),
        ]))
        story.append(interp_table)
        story.append(Spacer(1, 8))

    # ── Footer ───────────────────────────────────────────────────────────────
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=0.5, color=GRAY))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "This report was generated automatically by a machine learning based ADHD screening tool. "
        "It is intended for informational purposes only and does not replace professional medical advice, "
        "diagnosis, or treatment. Always seek the guidance of a qualified health provider with any "
        "questions you may have regarding a medical condition.",
        ParagraphStyle('Footer', parent=styles['Normal'],
                       fontSize=7.5, textColor=GRAY, alignment=TA_CENTER)
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ── Quick standalone test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_user = {
        'Age': 22, 'Gender': 'Male', 'EducationStage': 'University',
        'InattentionScore': 7, 'ImpulsivityScore': 6, 'HyperactivityScore': 5,
        'Daydreaming': 2, 'RSD': 2, 'SleepHours': 5.5, 'ScreenTime': 7,
        'ComorbidAnxiety': 1, 'ComorbidDepression': 0, 'FamilyHistoryADHD': 1,
        'Medication': 'No', 'SchoolSupport': 'Yes', 'AcademicScore': 52,
    }
    sample_game = {
        'total_trials': 60,
        'correct_go': 28,
        'missed_go': 7,
        'correct_inhibit': 17,
        'commission_errors': 8,
        'reaction_times': [
            320, 410, 290, 510, 380, 450, 310, 490, 360, 420,
            280, 530, 370, 480, 300, 440, 390, 520, 340, 460,
            315, 425, 295, 505, 385, 455, 305, 495, 365, 415,
        ]
    }
    pdf_bytes = generate_adhd_report(sample_user, sample_game, prediction=1, user_name="Alex")
    with open("/mnt/user-data/outputs/adhd_report_sample.pdf", "wb") as f:
        f.write(pdf_bytes)
    print("Report saved.")
