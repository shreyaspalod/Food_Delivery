# Decisions Log — Case 3: Food Delivery Demand Pulse

## Assumptions I made

1. **Surge = binary flag** — The `surge_applied` column is 0/1 (not a multiplier), so I measure *when* surge fires, not by how much.
2. **"Peak" = top-quartile demand hours** — Hours 12–14 and 18–21 consistently land in the 75th percentile of hourly demand across all cities. I use this as the empirical definition of peak.
3. **Delhi as forecast city** — Delhi is a high-volume, representative Tier A city and geographically central. Results generalise to other Tier A cities (Bangalore, Mumbai) with minimal adjustment.
4. **90-day history is sufficient for weekly seasonality** — Prophet documentation suggests 2–3 full seasonal cycles; 90 days = ~13 weeks, which is ample for weekly patterns.
5. **No weather/holiday data augmentation** — The 90-day synthetic dataset doesn't contain obvious holiday spikes (no single-day outliers > 3σ). Adding external data would require live API keys not appropriate for a sandbox; I document the extension path in README.

## Trade-offs

| Choice | Alternative | Why I picked this |
|--------|-------------|-------------------|
| Prophet | SARIMA / LSTM | Prophet handles missing days and weekly seasonality out-of-the-box with minimal tuning. Explainable to non-technical Ops Head. |
| Daily forecast granularity | Hourly | Hourly forecast over 90 days of data risks overfitting; daily is actionable for Monday briefing cadence. |
| Seasonal-naïve fallback | No fallback | Ensures the dashboard works even if Prophet install fails on HF Spaces free tier. |
| Streamlit | Gradio | Streamlit layout primitives (tabs, columns, sidebar) map better to a multi-view ops dashboard than Gradio's row/column model. |
| Plotly | Matplotlib | Plotly renders interactively in the browser — users can hover, zoom, filter. Critical for a dashboard the Ops Head "plays with." |
| City tiering (Tier A/B) | Per-city individual thresholds | Two tiers balance specificity with operational simplicity. 7 individual thresholds would require a rule-engine rewrite. |

## What I de-scoped and why

- **Multi-city Prophet forecasting** — Time-box constraint. Delhi is representative; the model code is parameterised by city, so extension is a single loop.
- **Weather / holiday calendar merge** — No API keys available in sandbox; no clear holiday signal in synthetic data.
- **Hour-level forecast** — Too noisy at 90-day history; daily is operationally sufficient.
- **Cuisine-level demand model** — Added complexity for marginal ops value; cuisine patterns are surfaced in the EDA tab.

## What I'd do differently with another day

- Run per-city Prophet models (all 7 cities, hourly granularity)
- Implement proper rolling walk-forward backtest to validate MAPE
- Integrate Indian public holiday calendar (2025) and weather API
- Add Slack webhook to push Monday morning forecast summary automatically
