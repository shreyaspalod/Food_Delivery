# Case 3: Food Delivery Demand Pulse

**Live demo:** https://huggingface.co/spaces/YOUR_USERNAME/case3-demand-pulse  
**Repo:** https://github.com/YOUR_USERNAME/case3-demand-pulse  
**Demo video:** *(add Loom/YouTube link here)*

## What this is

An interactive Streamlit dashboard that reveals when food delivery demand truly spikes across 7 Indian cities, exposes the surge-incentive mismatch, and surfaces 3 concrete policy recommendations the Ops Head can act on Monday.

## How to run locally

```bash
git clone https://github.com/YOUR_USERNAME/case3-demand-pulse.git
cd case3-demand-pulse
pip install -r requirements.txt
# Copy case3_food_delivery_orders.csv into the same folder
streamlit run app.py
```

Then open http://localhost:8501

## How to deploy to Hugging Face Spaces

1. Sign up at https://huggingface.co
2. Profile → New Space → Name: `case3-demand-pulse` · SDK: **Streamlit** · Hardware: CPU basic (Free)
3. Clone the Space repo, drop in `app.py`, `requirements.txt`, and `case3_food_delivery_orders.csv`
4. Push — Space auto-builds in ~2 min

## Stack

| Tool | Why |
|------|-----|
| **Pandas / NumPy** | Data wrangling — 50K rows, feature engineering |
| **Prophet** | 7-day demand forecast with weekly seasonality; handles gaps cleanly |
| **Plotly** | Interactive charts (heatmap, dual-axis, forecast with CI) |
| **Streamlit** | Rapid dashboard with sidebar filters — no front-end code needed |
| **Hugging Face Spaces** | Free CPU hosting, auto-deploys from git push |

## Deliverables

| File | Description |
|------|-------------|
| `app.py` | Streamlit dashboard — 4 tabs: Demand, City Cohorts, Forecast, Recommendations |
| `requirements.txt` | HF Spaces dependencies |
| `analysis_notebook.ipynb` | Clean, narrated EDA + forecast notebook (runnable top-to-bottom) |
| `exec_summary.docx` | One-page Word exec summary with 3 recommendations + impact |
| `deck.pptx` | 5-slide insight deck for the Ops Head |
| `forecast_delhi_7day.csv` | Prophet 7-day forecast output for Delhi |

## What's NOT done

- Live API integration (data is bundled as CSV — no real-time feed)
- Weather / holiday augmentation (justified in DECISIONS.md)
- Multi-city simultaneous forecasting (Delhi only — per time-box constraint)

## In production, I would also add

- Nightly forecast re-training triggered by Airflow DAG
- City-level hourly Prophet models for per-shift rider pre-positioning
- A/B test instrumentation layer tracking cost-per-order and delivery-time guardrails
- Weather + Indian holiday calendar merge for special-event demand spikes
- Slack alert if MAPE > 20% for 3 consecutive days
