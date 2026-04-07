import streamlit as st
import pandas as pd
import numpy as np
from ortools.sat.python import cp_model
import plotly.graph_objects as go
import json

st.set_page_config(
    page_title="TimetableAI",
    page_icon="🗓️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
h1, h2, h3 { font-family: 'Syne', sans-serif; font-weight: 800; letter-spacing: -0.02em; }
.stApp { background-color: #0d0d0d; color: #f0ede6; }
section[data-testid="stSidebar"] { background-color: #111; border-right: 1px solid #222; }
section[data-testid="stSidebar"] * { color: #f0ede6 !important; }
.stButton > button {
    background: #c8f135; color: #0d0d0d; border: none; border-radius: 2px;
    font-family: 'Syne', sans-serif; font-weight: 700; font-size: 14px;
    padding: 0.6rem 1.5rem; width: 100%; letter-spacing: 0.04em;
}
.stButton > button:hover { opacity: 0.85; }
.block-container { padding: 2rem 2.5rem; }
.tag {
    display: inline-block; background: #1e1e1e; border: 1px solid #333;
    color: #aaa; font-family: 'DM Mono', monospace; font-size: 11px;
    padding: 2px 8px; border-radius: 2px; margin: 2px;
}
.stat-box { background: #111; border: 1px solid #222; border-radius: 2px; padding: 1rem 1.2rem; text-align: center; }
.stat-num { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; color: #c8f135; }
.stat-label { font-family: 'DM Mono', monospace; font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 0.08em; }
.conflict-warn {
    background: #1a0f00; border-left: 3px solid #f59e0b; padding: 0.6rem 1rem;
    font-family: 'DM Mono', monospace; font-size: 12px; color: #f59e0b; margin: 4px 0;
}
.success-msg {
    background: #0a1a00; border-left: 3px solid #c8f135; padding: 0.6rem 1rem;
    font-family: 'DM Mono', monospace; font-size: 12px; color: #c8f135; margin: 4px 0;
}
.section-label {
    font-family: 'DM Mono', monospace; font-size: 11px; color: #555;
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem;
    border-bottom: 1px solid #1e1e1e; padding-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

# ── constants ─────────────────────────────────────────────────────────────────

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
SLOTS = [f"{h:02d}:00" for h in range(6, 24)]

ENERGY_DEFAULTS = {
    "06:00": 3, "07:00": 5, "08:00": 7, "09:00": 8, "10:00": 9,
    "11:00": 9, "12:00": 7, "13:00": 5, "14:00": 6, "15:00": 7,
    "16:00": 8, "17:00": 8, "18:00": 7, "19:00": 7, "20:00": 6,
    "21:00": 5, "22:00": 4, "23:00": 3,
}

SUBJECT_COLORS = [
    "#c8f135", "#38bdf8", "#f472b6", "#fb923c",
    "#a78bfa", "#34d399", "#fbbf24", "#f87171",
    "#60a5fa", "#4ade80",
]

def subject_color(idx): return SUBJECT_COLORS[idx % len(SUBJECT_COLORS)]
def energy_score(slot, energy_map): return energy_map.get(slot, 5)


# ── OR-Tools optimizer ────────────────────────────────────────────────────────

def optimize_timetable(subjects, busy_slots, energy_map, max_daily_hours, spread_subjects):
    model = cp_model.CpModel()
    n_days, n_slots, n_subj = len(DAYS), len(SLOTS), len(subjects)

    assign = {}
    for s in range(n_subj):
        for d in range(n_days):
            for t in range(n_slots):
                assign[(s, d, t)] = model.NewBoolVar(f"a_{s}_{d}_{t}")

    # Each subject gets exactly the required hours
    for s, subj in enumerate(subjects):
        model.Add(sum(assign[(s, d, t)] for d in range(n_days) for t in range(n_slots)) == subj["hours"])

    # No overlap — only one subject per slot per day
    for d in range(n_days):
        for t in range(n_slots):
            model.Add(sum(assign[(s, d, t)] for s in range(n_subj)) <= 1)

    # Busy slots blocked
    for (bd, bt) in busy_slots:
        for s in range(n_subj):
            model.Add(assign[(s, bd, bt)] == 0)

    # Max daily hours
    for d in range(n_days):
        model.Add(sum(assign[(s, d, t)] for s in range(n_subj) for t in range(n_slots)) <= max_daily_hours)

    # Spread: cap sessions per subject per day
    if spread_subjects:
        for s, subj in enumerate(subjects):
            cap = max(1, (subj["hours"] + 3) // 4)
            for d in range(n_days):
                model.Add(sum(assign[(s, d, t)] for t in range(n_slots)) <= cap)

    # Objective: maximise energy-weighted assignments (harder subjects → high-energy slots)
    model.Maximize(sum(
        assign[(s, d, t)] * energy_score(slot, energy_map) * subjects[s].get("difficulty", 3)
        for s in range(n_subj)
        for d in range(n_days)
        for t, slot in enumerate(SLOTS)
    ))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    status = solver.Solve(model)

    schedule = []
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for s, subj in enumerate(subjects):
            for d in range(n_days):
                for t, slot in enumerate(SLOTS):
                    if solver.Value(assign[(s, d, t)]):
                        schedule.append({
                            "subject": subj["name"],
                            "day": DAYS[d],
                            "day_idx": d,
                            "slot": slot,
                            "slot_idx": t,
                            "energy": energy_score(slot, energy_map),
                            "difficulty": subj.get("difficulty", 3),
                            "color": subject_color(s),
                        })
        return schedule, status == cp_model.OPTIMAL
    return [], False


# ── conflict detector ─────────────────────────────────────────────────────────

def detect_conflicts(subjects, max_daily_hours):
    warnings = []
    total = sum(s["hours"] for s in subjects)
    available = max_daily_hours * 7
    if total > available:
        warnings.append(f"Total study hours ({total}h) exceed weekly capacity ({available}h). Reduce hours or increase daily limit.")
    for s in subjects:
        if s["hours"] > max_daily_hours * 2:
            warnings.append(f"'{s['name']}' needs {s['hours']}h — consider splitting into shorter daily sessions.")
    exam_subjects = [s for s in subjects if s.get("exam_soon")]
    if len(exam_subjects) > 3:
        warnings.append(f"{len(exam_subjects)} subjects have exams soon — the schedule may be very tight.")
    return warnings


# ── charts ────────────────────────────────────────────────────────────────────

def build_timetable_chart(schedule):
    if not schedule:
        return None
    fig = go.Figure()
    seen = set()
    for entry in schedule:
        is_first = entry["subject"] not in seen
        seen.add(entry["subject"])
        fig.add_trace(go.Bar(
            x=[entry["day"]], y=[1], base=[entry["slot_idx"]],
            marker_color=entry["color"], marker_line_width=0,
            name=entry["subject"], showlegend=is_first, width=0.6,
            hovertemplate=f"<b>{entry['subject']}</b><br>{entry['day']} · {entry['slot']}<br>Energy: {entry['energy']}/10<extra></extra>",
        ))
    fig.update_layout(
        barmode="stack",
        paper_bgcolor="#0d0d0d", plot_bgcolor="#111",
        font=dict(family="DM Mono", color="#f0ede6", size=11),
        xaxis=dict(categoryorder="array", categoryarray=DAYS, gridcolor="#1e1e1e", linecolor="#222"),
        yaxis=dict(
            tickvals=list(range(0, len(SLOTS), 2)),
            ticktext=[SLOTS[i] for i in range(0, len(SLOTS), 2)],
            gridcolor="#1e1e1e", linecolor="#222", range=[0, len(SLOTS)]
        ),
        legend=dict(bgcolor="#111", bordercolor="#222", borderwidth=1, font=dict(family="DM Mono", size=11)),
        margin=dict(l=60, r=20, t=20, b=40), height=520,
    )
    return fig

def build_energy_chart(energy_map):
    fig = go.Figure(go.Scatter(
        x=list(energy_map.keys()), y=list(energy_map.values()),
        fill="tozeroy", line=dict(color="#c8f135", width=2),
        fillcolor="rgba(200,241,53,0.10)", mode="lines",
    ))
    fig.update_layout(
        paper_bgcolor="#0d0d0d", plot_bgcolor="#111",
        font=dict(family="DM Mono", color="#f0ede6", size=11),
        xaxis=dict(gridcolor="#1e1e1e", linecolor="#222", tickangle=-45),
        yaxis=dict(gridcolor="#1e1e1e", linecolor="#222", range=[0, 10]),
        margin=dict(l=40, r=10, t=10, b=60), height=180,
    )
    return fig


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⬛ TimetableAI")
    st.markdown('<div class="section-label">Subjects (edit JSON)</div>', unsafe_allow_html=True)

    if "subjects" not in st.session_state:
        st.session_state.subjects = [
            {"name": "Mathematics",     "hours": 6, "difficulty": 5, "exam_soon": True},
            {"name": "Data Structures", "hours": 5, "difficulty": 4, "exam_soon": False},
            {"name": "Machine Learning","hours": 4, "difficulty": 4, "exam_soon": True},
            {"name": "English",         "hours": 2, "difficulty": 2, "exam_soon": False},
        ]

    subjects_json = st.text_area(
        "Subjects JSON",
        value=json.dumps(st.session_state.subjects, indent=2),
        height=230,
        help="Fields: name (str), hours (int), difficulty (1–5), exam_soon (bool)"
    )
    try:
        st.session_state.subjects = json.loads(subjects_json)
        st.markdown('<div class="success-msg">✓ subjects valid</div>', unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f'<div class="conflict-warn">JSON error: {e}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:1rem">Constraints</div>', unsafe_allow_html=True)
    max_daily = st.slider("Max study hours / day", 2, 12, 6)
    spread = st.checkbox("Spread subjects across days", value=True)

    st.markdown('<div class="section-label" style="margin-top:1rem">Block out busy slots</div>', unsafe_allow_html=True)
    busy_days  = st.multiselect("Busy days",  DAYS,  default=["Saturday", "Sunday"])
    busy_times = st.multiselect("Busy times", SLOTS, default=["08:00", "09:00", "10:00"])

    st.markdown('<div class="section-label" style="margin-top:1rem">Energy profile</div>', unsafe_allow_html=True)
    use_custom = st.checkbox("Customise my energy levels", value=False)

    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("⚡  Generate timetable")


# ── main ──────────────────────────────────────────────────────────────────────

st.markdown("# Smart Timetable Optimizer")
st.markdown(
    '<p style="font-family:\'DM Mono\',monospace;color:#444;font-size:13px;margin-top:-0.5rem">'
    'OR-Tools CP-SAT · energy-aware scheduling · conflict detection</p>',
    unsafe_allow_html=True
)

subjects = st.session_state.subjects

# Energy map
energy_map = dict(ENERGY_DEFAULTS)
if use_custom:
    st.markdown("### Your energy across the day")
    st.caption("Set your alertness level for each hour — harder subjects will be placed in your peak slots.")
    cols = st.columns(6)
    for i, slot in enumerate(SLOTS):
        with cols[i % 6]:
            energy_map[slot] = st.slider(slot, 1, 10, ENERGY_DEFAULTS[slot], key=f"e_{slot}")

st.markdown("### Daily energy curve")
st.plotly_chart(build_energy_chart(energy_map), use_container_width=True)

# Conflict warnings
warnings = detect_conflicts(subjects, max_daily)
if warnings:
    st.markdown("### ⚠ Conflict detection")
    for w in warnings:
        st.markdown(f'<div class="conflict-warn">{w}</div>', unsafe_allow_html=True)

# Stats
st.markdown("### Overview")
c1, c2, c3, c4 = st.columns(4)
total_h = sum(s["hours"] for s in subjects)
avg_e   = round(sum(energy_map.values()) / len(energy_map), 1)
with c1: st.markdown(f'<div class="stat-box"><div class="stat-num">{len(subjects)}</div><div class="stat-label">Subjects</div></div>', unsafe_allow_html=True)
with c2: st.markdown(f'<div class="stat-box"><div class="stat-num">{total_h}h</div><div class="stat-label">Total hours</div></div>', unsafe_allow_html=True)
with c3: st.markdown(f'<div class="stat-box"><div class="stat-num">{max_daily}h</div><div class="stat-label">Max / day</div></div>', unsafe_allow_html=True)
with c4: st.markdown(f'<div class="stat-box"><div class="stat-num">{avg_e}</div><div class="stat-label">Avg energy</div></div>', unsafe_allow_html=True)

# Run
if run:
    busy_set = set()
    for bd in busy_days:
        di = DAYS.index(bd)
        for bt in busy_times:
            ti = SLOTS.index(bt)
            busy_set.add((di, ti))

    with st.spinner("Running CP-SAT optimizer..."):
        schedule, optimal = optimize_timetable(subjects, busy_set, energy_map, max_daily, spread)

    if schedule:
        label = "OPTIMAL" if optimal else "FEASIBLE"
        st.markdown(f'<div class="success-msg">✓ Solution found · {label} · {len(schedule)} sessions placed</div>', unsafe_allow_html=True)

        st.markdown("### Weekly timetable")
        st.plotly_chart(build_timetable_chart(schedule), use_container_width=True)

        st.markdown("### Day-by-day breakdown")
        df = pd.DataFrame(schedule).sort_values(["day_idx", "slot_idx"])
        for day in DAYS:
            day_df = df[df["day"] == day]
            if day_df.empty:
                continue
            with st.expander(f"**{day}** — {len(day_df)}h"):
                for _, row in day_df.iterrows():
                    bar = "█" * row["energy"] + "░" * (10 - row["energy"])
                    st.markdown(
                        f'<span class="tag">{row["slot"]}</span> '
                        f'<span style="color:{row["color"]};font-weight:600">{row["subject"]}</span>'
                        f'<span style="font-family:\'DM Mono\',monospace;font-size:11px;color:#444"> · energy {bar} {row["energy"]}/10</span>',
                        unsafe_allow_html=True
                    )

        st.markdown("### Subject summary")
        summary = df.groupby("subject").agg(
            Hours=("slot", "count"),
            Avg_Energy=("energy", "mean"),
            Study_Days=("day", lambda x: ", ".join(sorted(set(x))))
        ).reset_index()
        summary["Avg_Energy"] = summary["Avg_Energy"].round(1)
        summary.columns = ["Subject", "Hours", "Avg Energy", "Study Days"]
        st.dataframe(summary, use_container_width=True, hide_index=True)

        csv = df[["subject", "day", "slot", "energy", "difficulty"]].to_csv(index=False)
        st.download_button("⬇ Download schedule (CSV)", csv, "timetable.csv", "text/csv")

    else:
        st.error("No feasible schedule found. Try reducing total hours, increasing max daily hours, or removing busy slots.")

else:
    st.markdown(
        '<div style="text-align:center;padding:5rem 0;color:#2a2a2a;font-family:\'DM Mono\',monospace;font-size:13px;">'
        'configure subjects & constraints in the sidebar → hit generate</div>',
        unsafe_allow_html=True
    )
