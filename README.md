# TimetableAI

Energy-aware weekly study scheduler built with OR-Tools CP-SAT + Streamlit.

## Features
- Schedules subjects into your peak energy hours
- Blocks out busy slots (classes, gym, etc.)
- Detects conflicts before they happen
- Exports schedule as CSV

## Run locally
\```bash
pip install -r requirements.txt
streamlit run app.py
\```

## How it works
Uses **Google OR-Tools CP-SAT** to solve a binary assignment problem — placing study sessions across days and time slots while maximising `energy × difficulty`. Harder subjects land in your most alert hours.

## Tech stack
`OR-Tools` · `Streamlit` · `Plotly` · `Pandas`
