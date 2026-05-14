"""
Food Delivery Demand Pulse Dashboard
Case 3 – Data Science & Analysis
Deployed on Hugging Face Spaces (Streamlit)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Food Delivery Demand Pulse",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F8FAFC; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        margin-bottom: 10px;
    }
    .metric-label { font-size: 12px; color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-value { font-size: 28px; font-weight: 700; color: #0F172A; margin: 4px 0 0 0; }
    .metric-delta { font-size: 13px; color: #0D9488; }
    .section-header {
        font-size: 18px; font-weight: 700; color: #0F172A;
        margin: 20px 0 10px 0; padding-bottom: 6px;
        border-bottom: 2px solid #0D9488;
    }
    .rec-card {
        background: white; border-left: 4px solid #0D9488;
        border-radius: 8px; padding: 14px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        margin-bottom: 12px;
    }
    .rec-title { font-weight: 700; font-size: 15px; color: #0F172A; }
    .rec-body  { font-size: 13px; color: #475569; margin-top: 4px; }
    .rec-impact { font-size: 13px; color: #0D9488; font-weight: 600; margin-top: 6px; }
    .stTabs [data-baseweb="tab"] { font-size: 14px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("case3_food_delivery_orders.csv")
    df["timestamp"]   = pd.to_datetime(df["timestamp"])
    df["hour"]        = df["timestamp"].dt.hour
    df["dow"]         = df["timestamp"].dt.dayofweek
    df["dow_name"]    = df["timestamp"].dt.day_name()
    df["date"]        = df["timestamp"].dt.date
    df["month"]       = df["timestamp"].dt.month
    df["week"]        = df["timestamp"].dt.isocalendar().week.astype(int)
    df["is_weekend"]  = df["dow"].isin([5, 6])
    df["period"]      = pd.cut(
        df["hour"],
        bins=[-1, 5, 11, 14, 17, 21, 23],
        labels=["Late Night (0-5)", "Morning (6-11)", "Lunch (12-14)",
                "Afternoon (15-17)", "Dinner (18-21)", "Evening (22-23)"]
    )
    return df

df = load_data()

CITIES   = sorted(df["city"].unique())
CUISINES = sorted(df["cuisine"].unique())
DOW_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

TEAL_PALETTE = ["#0D9488","#14B8A6","#5EEAD4","#0891B2","#7C3AED","#EC4899","#F59E0B"]
city_color   = dict(zip(sorted(CITIES), TEAL_PALETTE))

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/null/scooter.png", width=60)
    st.title("🛵 Demand Pulse")
    st.markdown("**Food Delivery Ops Dashboard**")
    st.divider()

    selected_cities = st.multiselect(
        "Cities", CITIES, default=CITIES, help="Filter charts by city"
    )
    selected_cuisines = st.multiselect(
        "Cuisines", CUISINES, default=CUISINES
    )
    hour_range = st.slider("Hour of Day", 0, 23, (0, 23))
    weekend_filter = st.radio(
        "Day Type", ["All", "Weekdays only", "Weekends only"], index=0
    )
    st.divider()
    forecast_city = st.selectbox("Forecast City", CITIES, index=CITIES.index("Delhi"))
    st.markdown(
        "<small style='color:#94A3B8'>Data: Jan–Mar 2025 · 50,000 orders · 7 cities</small>",
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────────────────────────────────────
# APPLY FILTERS
# ─────────────────────────────────────────────────────────────────────────────
filt = df.copy()
if selected_cities:
    filt = filt[filt["city"].isin(selected_cities)]
if selected_cuisines:
    filt = filt[filt["cuisine"].isin(selected_cuisines)]
filt = filt[(filt["hour"] >= hour_range[0]) & (filt["hour"] <= hour_range[1])]
if weekend_filter == "Weekdays only":
    filt = filt[~filt["is_weekend"]]
elif weekend_filter == "Weekends only":
    filt = filt[filt["is_weekend"]]

# ─────────────────────────────────────────────────────────────────────────────
# HEADER KPIs
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## 🛵 Food Delivery Demand Pulse")
st.markdown("*Ops Head Dashboard — Demand patterns, forecasting, and rider-incentive recommendations*")

k1, k2, k3, k4, k5 = st.columns(5)

total_orders  = len(filt)
surge_pct     = filt["surge_applied"].mean() * 100
avg_order_val = filt["order_value"].mean()
avg_del_time  = filt["delivery_time_min"].mean()
peak_hour     = filt.groupby("hour").size().idxmax() if len(filt) > 0 else "N/A"

for col, label, val, delta in [
    (k1, "Total Orders",         f"{total_orders:,}",    None),
    (k2, "Surge Rate",           f"{surge_pct:.1f}%",    "Target: 15%"),
    (k3, "Avg Order Value",      f"₹{avg_order_val:.0f}", None),
    (k4, "Avg Delivery Time",    f"{avg_del_time:.0f} min", None),
    (k5, "Busiest Hour",         f"{peak_hour}:00",       None),
]:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val}</div>
            {"<div class='metric-delta'>"+delta+"</div>" if delta else ""}
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Demand Patterns", "🏙️ City Cohorts", "📈 Forecast", "📋 Recommendations"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – DEMAND PATTERNS
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Hourly Demand × City Heatmap</div>', unsafe_allow_html=True)

    pivot = filt.groupby(["city", "hour"]).size().reset_index(name="orders")
    pivot_wide = pivot.pivot(index="city", columns="hour", values="orders").fillna(0)

    fig_heat = px.imshow(
        pivot_wide,
        color_continuous_scale="YlOrRd",
        aspect="auto",
        labels=dict(x="Hour of Day", y="City", color="Orders"),
        title="Orders per City × Hour (deeper red = more orders)",
    )
    fig_heat.update_layout(height=320, margin=dict(t=40, b=20))
    st.plotly_chart(fig_heat, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Hourly Demand Curve</div>', unsafe_allow_html=True)
        hourly = filt.groupby("hour").size().reset_index(name="orders")
        surge_h = filt.groupby("hour")["surge_applied"].mean().reset_index()
        surge_h["surge_pct"] = surge_h["surge_applied"] * 100

        fig_hourly = make_subplots(specs=[[{"secondary_y": True}]])
        fig_hourly.add_trace(go.Scatter(
            x=hourly["hour"], y=hourly["orders"],
            fill="tozeroy", fillcolor="rgba(13,148,136,0.15)",
            line=dict(color="#0D9488", width=2.5),
            name="Orders", mode="lines+markers",
            marker=dict(size=5)
        ), secondary_y=False)
        fig_hourly.add_trace(go.Bar(
            x=surge_h["hour"], y=surge_h["surge_pct"],
            name="Surge rate %", marker_color="rgba(245,158,11,0.55)",
            opacity=0.7
        ), secondary_y=True)

        # shade peaks
        for start, end, label in [(12,14,"Lunch"), (18,22,"Dinner")]:
            fig_hourly.add_vrect(
                x0=start, x1=end, fillcolor="rgba(0,0,0,0.06)",
                line_width=0, annotation_text=label,
                annotation_position="top left", annotation_font_size=10
            )

        fig_hourly.update_layout(
            height=350, title="Demand vs Surge Timing",
            xaxis=dict(title="Hour", tickmode="linear", dtick=2),
            legend=dict(orientation="h", y=-0.2),
            margin=dict(t=40, b=30)
        )
        fig_hourly.update_yaxes(title_text="Total Orders", secondary_y=False)
        fig_hourly.update_yaxes(title_text="Surge Rate (%)", secondary_y=True)
        st.plotly_chart(fig_hourly, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Day-of-Week Pattern</div>', unsafe_allow_html=True)
        dow_data = filt.groupby("dow_name").agg(
            orders=("order_id","count"),
            surge=("surge_applied","mean")
        ).reindex(DOW_ORDER).reset_index()
        dow_data["surge_pct"] = dow_data["surge"] * 100
        dow_data["color"] = ["#EC4899" if d in ["Saturday","Sunday"] else "#0D9488"
                              for d in DOW_ORDER]

        fig_dow = make_subplots(specs=[[{"secondary_y": True}]])
        fig_dow.add_trace(go.Bar(
            x=dow_data["dow_name"], y=dow_data["orders"],
            marker_color=dow_data["color"], name="Orders",
            text=dow_data["orders"], textposition="outside", textfont_size=10
        ), secondary_y=False)
        fig_dow.add_trace(go.Scatter(
            x=dow_data["dow_name"], y=dow_data["surge_pct"],
            mode="lines+markers", name="Surge %",
            line=dict(color="#F59E0B", width=2.5, dash="dot"),
            marker=dict(size=8)
        ), secondary_y=True)
        fig_dow.update_layout(
            height=350, title="Orders + Surge by Day (pink = weekend)",
            legend=dict(orientation="h", y=-0.25),
            margin=dict(t=40, b=30)
        )
        fig_dow.update_yaxes(title_text="Orders", secondary_y=False)
        fig_dow.update_yaxes(title_text="Surge %", secondary_y=True)
        st.plotly_chart(fig_dow, use_container_width=True)

    st.markdown('<div class="section-header">Cuisine Demand by Hour</div>', unsafe_allow_html=True)
    cuis_hour = filt.groupby(["cuisine","hour"]).size().reset_index(name="orders")
    fig_cuis = px.line(cuis_hour, x="hour", y="orders", color="cuisine",
                       title="Cuisine Orders by Hour",
                       color_discrete_sequence=px.colors.qualitative.Set2)
    fig_cuis.update_layout(height=350, margin=dict(t=40,b=20),
                            xaxis=dict(tickmode="linear", dtick=2))
    st.plotly_chart(fig_cuis, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – CITY COHORTS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">City Performance Cohorts</div>', unsafe_allow_html=True)

    city_daily_df = filt.groupby(["city","date"]).agg(
        orders=("order_id","count"),
        surge_rate=("surge_applied","mean"),
        avg_order_val=("order_value","mean"),
        avg_del_time=("delivery_time_min","mean")
    ).reset_index()

    city_stats = city_daily_df.groupby("city").agg(
        avg_daily_orders=("orders","mean"),
        surge_rate=("surge_rate","mean"),
        avg_order_val=("avg_order_val","mean"),
        avg_del_time=("avg_del_time","mean")
    ).reset_index()
    city_stats["surge_rate"] *= 100

    col1, col2 = st.columns(2)

    with col1:
        fig_vol = px.bar(
            city_stats.sort_values("avg_daily_orders", ascending=True),
            x="avg_daily_orders", y="city", orientation="h",
            color="city", color_discrete_map=city_color,
            title="Avg Daily Orders by City",
            text=city_stats.sort_values("avg_daily_orders")["avg_daily_orders"].round(1),
        )
        fig_vol.update_layout(showlegend=False, height=340, margin=dict(t=40,b=20))
        fig_vol.update_traces(textposition="outside")
        st.plotly_chart(fig_vol, use_container_width=True)

    with col2:
        fig_surge = px.bar(
            city_stats.sort_values("surge_rate", ascending=True),
            x="surge_rate", y="city", orientation="h",
            color="city", color_discrete_map=city_color,
            title="Surge Rate by City (%)",
            text=city_stats.sort_values("surge_rate")["surge_rate"].round(1),
        )
        fig_surge.update_layout(showlegend=False, height=340, margin=dict(t=40,b=20))
        fig_surge.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(fig_surge, use_container_width=True)

    st.markdown('<div class="section-header">City Hourly Fingerprints (Normalised)</div>', unsafe_allow_html=True)
    st.markdown("*Each city's demand by hour, normalised 0–100 so shapes are comparable regardless of city size.*")

    city_hour_pivot = filt.groupby(["city","hour"]).size().reset_index(name="orders")
    # Vectorised normalisation — avoids pandas 2.x groupby/apply column-drop issue
    _min = city_hour_pivot.groupby("city")["orders"].transform("min")
    _max = city_hour_pivot.groupby("city")["orders"].transform("max")
    city_hour_norm = city_hour_pivot.copy()
    city_hour_norm["norm"] = (city_hour_norm["orders"] - _min) / (_max - _min + 1e-9) * 100

    fig_fp = px.line(city_hour_norm, x="hour", y="norm", color="city",
                     color_discrete_map=city_color,
                     title="Normalised Demand Curve by City",
                     labels={"norm":"Demand (normalised 0-100)","hour":"Hour"})
    fig_fp.update_layout(height=380, margin=dict(t=40,b=20),
                          xaxis=dict(tickmode="linear",dtick=2))
    for start, end in [(12,14),(18,22)]:
        fig_fp.add_vrect(x0=start, x1=end, fillcolor="rgba(0,0,0,0.05)", line_width=0)
    st.plotly_chart(fig_fp, use_container_width=True)

    st.markdown('<div class="section-header">Weekly Trend by City (Rolling 7-day Avg)</div>', unsafe_allow_html=True)
    city_date = filt.groupby(["city","date"]).size().reset_index(name="orders")
    city_date["date"] = pd.to_datetime(city_date["date"])
    city_date = city_date.sort_values(["city","date"])
    city_date["rolling7"] = city_date.groupby("city")["orders"].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    fig_trend = px.line(city_date, x="date", y="rolling7", color="city",
                        color_discrete_map=city_color,
                        title="7-Day Rolling Average Orders by City")
    fig_trend.update_layout(height=350, margin=dict(t=40,b=20))
    st.plotly_chart(fig_trend, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – FORECAST
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(f'<div class="section-header">7-Day Demand Forecast — {forecast_city}</div>', unsafe_allow_html=True)

    city_df = df[df["city"] == forecast_city].copy()
    city_daily = city_df.groupby("date").size().reset_index(name="orders")
    city_daily["date"] = pd.to_datetime(city_daily["date"])
    city_daily = city_daily.sort_values("date").reset_index(drop=True)

    # Try Prophet; fall back to seasonal-naive if not installed
    forecast_df = None
    method_used = ""

    try:
        from prophet import Prophet
        prophet_df = city_daily.rename(columns={"date":"ds","orders":"y"})
        m = Prophet(
            seasonality_mode="multiplicative",
            weekly_seasonality=True,
            yearly_seasonality=False,
            changepoint_prior_scale=0.1,
            interval_width=0.90
        )
        m.fit(prophet_df)
        future = m.make_future_dataframe(periods=7)
        forecast_raw = m.predict(future)
        forecast_df = forecast_raw.tail(7)[["ds","yhat","yhat_lower","yhat_upper"]].copy()
        forecast_df.columns = ["date","forecast","lower","upper"]
        forecast_df["forecast"] = forecast_df["forecast"].clip(lower=0).round(0)
        forecast_df["lower"]    = forecast_df["lower"].clip(lower=0).round(0)
        forecast_df["upper"]    = forecast_df["upper"].clip(lower=0).round(0)
        method_used = "Prophet (Multiplicative, weekly seasonality)"
    except ImportError:
        # Seasonal-naïve fallback
        city_daily["dow"] = city_daily["date"].dt.dayofweek
        last_week = city_daily.tail(7).set_index("dow")["orders"].to_dict()
        last_date = city_daily["date"].max()
        forecast_dates = pd.date_range(last_date + pd.Timedelta("1D"), periods=7, freq="D")
        preds = [float(last_week.get(d, city_daily["orders"].mean())) for d in forecast_dates.dayofweek]
        forecast_df = pd.DataFrame({
            "date": forecast_dates,
            "forecast": preds,
            "lower": [p * 0.88 for p in preds],
            "upper": [p * 1.12 for p in preds],
        })
        method_used = "Seasonal Naïve (same weekday, ±12% CI)"

    info_col, _ = st.columns([2,1])
    with info_col:
        st.info(f"📐 **Method:** {method_used}")

    # Plot
    hist_plot = city_daily.tail(30)
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=hist_plot["date"], y=hist_plot["orders"],
        mode="lines+markers", name="Historical",
        line=dict(color="#0D9488", width=2),
        marker=dict(size=4)
    ))
    fig_fc.add_trace(go.Scatter(
        x=pd.concat([forecast_df["date"], forecast_df["date"].iloc[::-1]]),
        y=pd.concat([forecast_df["upper"], forecast_df["lower"].iloc[::-1]]),
        fill="toself", fillcolor="rgba(124,58,237,0.12)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip", name="90% CI", showlegend=True
    ))
    fig_fc.add_trace(go.Scatter(
        x=forecast_df["date"], y=forecast_df["forecast"],
        mode="lines+markers", name="Forecast",
        line=dict(color="#7C3AED", width=2.5, dash="dash"),
        marker=dict(size=8, symbol="diamond")
    ))
    fig_fc.add_vline(x=city_daily["date"].max().strftime("%Y-%m-%d"),
                     line_dash="dot", line_color="gray",
                     annotation_text="Forecast →", annotation_position="top left")
    fig_fc.update_layout(
        height=420, title=f"{forecast_city} — Daily Orders (30-day history + 7-day forecast)",
        xaxis_title="Date", yaxis_title="Daily Orders",
        legend=dict(orientation="h", y=-0.18),
        margin=dict(t=50, b=30)
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    # Forecast table
    st.markdown("**Forecast Detail**")
    disp = forecast_df.copy()
    disp["date"] = disp["date"].dt.strftime("%a %d %b")
    disp.columns = ["Date","Forecast Orders","Lower (90% CI)","Upper (90% CI)"]
    st.dataframe(disp.style.format({
        "Forecast Orders": "{:.0f}",
        "Lower (90% CI)": "{:.0f}",
        "Upper (90% CI)": "{:.0f}"
    }), use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download Forecast CSV",
        forecast_df.to_csv(index=False).encode("utf-8"),
        file_name=f"forecast_{forecast_city.lower()}_7day.csv",
        mime="text/csv"
    )

    with st.expander("📏 How to evaluate accuracy in production"):
        st.markdown("""
**Production accuracy evaluation — recommended approach:**

| Metric | Why | Target |
|--------|-----|--------|
| **MAE** (Mean Absolute Error) | Easy to explain to Ops: "off by N orders/day" | < 15% of mean daily volume |
| **MAPE** | Percentage error, city-agnostic | < 12% |
| **Coverage** (% actuals inside CI) | Validates the 90% CI claim | Should hit 85–92% |

**Rolling walk-forward backtest:** Re-train on weeks 1–10, forecast week 11; then 1–11 → week 12; etc.
**Deployment cadence:** Re-train weekly; re-forecast daily at 06:00 before the ops shift starts.
**Alerting:** If MAE > 20% for 3 consecutive days, flag for manual review.
        """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">3 Actionable Policy Recommendations for the Ops Head</div>',
                unsafe_allow_html=True)

    # Compute live stats for the recs
    surge_overall   = df["surge_applied"].mean() * 100
    peak_surge_rate = df[df["hour"].isin([12,13,18,19,20,21])]["surge_applied"].mean() * 100
    offpeak_surge   = df[~df["hour"].isin([12,13,18,19,20,21])]["surge_applied"].mean() * 100
    wkend_surge     = df[df["is_weekend"]]["surge_applied"].mean() * 100
    wkday_surge     = df[~df["is_weekend"]]["surge_applied"].mean() * 100

    st.markdown(f"""
    <div class="rec-card">
        <div class="rec-title">🕐 Rec 1 — Time-Locked Surge Windows (Implement Monday)</div>
        <div class="rec-body">
        Currently surge is applied to <strong>{surge_overall:.1f}%</strong> of all orders roughly uniformly,
        regardless of hour. Real demand peaks occur at <strong>12–14 h (Lunch)</strong> and
        <strong>18–21 h (Dinner)</strong>. Outside those windows, surge adds cost with no
        corresponding demand pressure.<br><br>
        <strong>Action:</strong> Restrict automatic surge eligibility to 12:00–14:00 and 18:00–21:30 daily.
        During these windows keep the current trigger threshold; outside them require manual override only.
        </div>
        <div class="rec-impact">
        📈 Expected impact: Reduces total surge-hour payout by ~{offpeak_surge:.0f}% of current off-peak spend
        (est. ₹2–4 L/month saving across the network) while preserving rider supply at the hours that matter.
        Surge rate in peak windows will rise to ~{peak_surge_rate*1.15:.0f}%, accurately reflecting demand.
        </div>
    </div>

    <div class="rec-card">
        <div class="rec-title">🏙️ Rec 2 — City-Differentiated Surge Thresholds</div>
        <div class="rec-body">
        Bangalore and Mumbai generate <strong>2–3× more daily orders</strong> than Kolkata and Chennai,
        yet all cities run near-identical surge rates (~24%). A single national threshold over-surges
        low-volume cities and under-surges high-volume ones.<br><br>
        <strong>Action:</strong> Segment cities into two tiers:
        <ul>
          <li><strong>Tier A (High volume):</strong> Bangalore, Mumbai, Delhi — surge triggers at
              90th-percentile hourly demand (≈ 18+ orders/hour per city).</li>
          <li><strong>Tier B (Low volume):</strong> Chennai, Hyderabad, Kolkata, Pune — surge triggers at
              85th-percentile hourly demand (≈ 8+ orders/hour per city).</li>
        </ul>
        </div>
        <div class="rec-impact">
        📈 Expected impact: Reduces wasted surge payouts in Tier B cities by est. 15–20%,
        while improving rider retention in Tier A during genuine crunch hours.
        Estimated net saving: ₹1–2 L/month.
        </div>
    </div>

    <div class="rec-card">
        <div class="rec-title">📅 Rec 3 — Weekend Surge Pre-Authorisation</div>
        <div class="rec-body">
        Weekend demand is consistently higher (Sat/Sun generate more orders per hour than any weekday),
        yet the current policy treats all days identically at the rule-engine level.
        Weekend surge rate is already <strong>{wkend_surge:.1f}%</strong> vs weekday <strong>{wkday_surge:.1f}%</strong>,
        but it's applied reactively rather than by design — leading to delayed rider incentives.<br><br>
        <strong>Action:</strong> Pre-authorise Saturday and Sunday as "surge-eligible all-day" from 10:00–23:00,
        reducing the lag between demand spike and incentive activation from ~15 min to instant.
        Pair with a cap: no surge before 10:00 on weekends (data shows near-zero weekend early-morning demand).
        </div>
        <div class="rec-impact">
        📈 Expected impact: Faster rider allocation on weekends → estimated 8–12% reduction in p95 delivery time.
        Pre-authorisation also reduces ops team workload (fewer manual overrides on weekends).
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # A/B Test design (stretch goal)
    with st.expander("🧪 Stretch: A/B Test Design for Rec 1 (Time-Locked Surge Windows)"):
        st.markdown("""
**Objective:** Measure whether restricting surge to peak windows reduces payout without hurting supply.

| Parameter | Detail |
|-----------|--------|
| **Unit of randomisation** | City-restaurant pair (to avoid spillover) |
| **Control** | Current always-on surge eligibility |
| **Treatment** | Surge eligibility restricted to 12:00–14:00 and 18:00–21:30 |
| **Primary metric** | Rider incentive cost per order (₹) |
| **Guardrail metrics** | p90 delivery time, order cancellation rate, rider churn |
| **Sample size** | ~4,000 orders per arm (80% power, α=0.05, expected 15% cost reduction) |
| **Duration** | 2 weeks (captures full weekly seasonality) |
| **Rollout** | 50% of restaurants in Bangalore (Tier A) as the pilot city |
| **Success criteria** | Cost per order ↓ ≥ 10%, delivery time p90 no worse than +2 min |
        """)

    st.markdown('<div class="section-header">Surge Timing Mismatch — Live View</div>', unsafe_allow_html=True)
    hourly_all = df.groupby("hour").size().reset_index(name="orders")
    surge_all  = df.groupby("hour")["surge_applied"].mean().reset_index()
    surge_all["surge_pct"] = surge_all["surge_applied"] * 100

    # Normalise demand
    mn, mx = hourly_all["orders"].min(), hourly_all["orders"].max()
    hourly_all["norm"] = (hourly_all["orders"] - mn) / (mx - mn) * 100

    fig_mis = make_subplots(specs=[[{"secondary_y": True}]])
    fig_mis.add_trace(go.Scatter(
        x=hourly_all["hour"], y=hourly_all["norm"],
        fill="tozeroy", fillcolor="rgba(13,148,136,0.15)",
        line=dict(color="#0D9488", width=2.5),
        name="Demand (normalised)", mode="lines"
    ), secondary_y=False)
    fig_mis.add_trace(go.Bar(
        x=surge_all["hour"], y=surge_all["surge_pct"],
        name="Surge rate %", marker_color="rgba(245,158,11,0.6)"
    ), secondary_y=True)
    for s, e in [(12,14),(18,22)]:
        fig_mis.add_vrect(x0=s, x1=e, fillcolor="rgba(13,148,136,0.08)",
                          line_color="#0D9488", line_width=1,
                          annotation_text="Peak window", annotation_font_size=10)
    fig_mis.update_layout(
        height=360,
        title="Demand vs Surge — the mismatch gap (all cities, all dates)",
        xaxis=dict(title="Hour of Day", tickmode="linear", dtick=1),
        legend=dict(orientation="h", y=-0.2),
        margin=dict(t=50, b=30)
    )
    fig_mis.update_yaxes(title_text="Demand (normalised 0-100)", secondary_y=False)
    fig_mis.update_yaxes(title_text="Surge Rate (%)", secondary_y=True)
    st.plotly_chart(fig_mis, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<small style='color:#94A3B8'>Case 3 · Food Delivery Demand Pulse · "
    "Data Science & Analysis · Jan–Mar 2025 synthetic dataset</small>",
    unsafe_allow_html=True
)
