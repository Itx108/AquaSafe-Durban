from pathlib import Path
import re

import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import requests
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AquaSafe Durban",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent

EVENTS_FILE = BASE_DIR / "AquaSafe_eThekwini_Official_Water_Events.csv"
RISK_FILE = BASE_DIR / "AquaSafe_Upgraded_Community_Risk_Snapshot.csv"
WEEKLY_FILE = BASE_DIR / "AquaSafe_Upgraded_Weekly_System_Series.csv"
LEADERBOARD_FILE = BASE_DIR / "AquaSafe_3_Model_Leaderboard.csv"
HORIZON_FILE = BASE_DIR / "AquaSafe_RF_Forecast_7_14_30_Days.csv"
FORECAST_FILE = BASE_DIR / "AquaSafe_RF_12_Week_Forecast.csv"

SUBURBS_QUERY = (
    "https://gis.durban.gov.za/server/rest/services/"
    "WebViewers/EXT_Cadastral/MapServer/19/query"
)


# ============================================================
# VISUAL STYLE
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.3rem;
        padding-bottom: 2rem;
    }

    [data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 12px;
        padding: 14px;
    }

    .aquasafe-card {
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 14px;
        padding: 16px 18px;
        margin-bottom: 12px;
    }

    .small-note {
        opacity: 0.75;
        font-size: 0.88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

def require_file(path: Path):
    if not path.exists():
        st.error(
            f"Required file is missing: {path.name}\n\n"
            "Place the Streamlit app and all AquaSafe CSV files in the same folder."
        )
        st.stop()


for required in [
    EVENTS_FILE,
    RISK_FILE,
    WEEKLY_FILE,
    LEADERBOARD_FILE,
    HORIZON_FILE,
    FORECAST_FILE,
]:
    require_file(required)


@st.cache_data
def load_data():
    events = pd.read_csv(EVENTS_FILE, parse_dates=["event_date"])

    risk = pd.read_csv(
        RISK_FILE,
        parse_dates=["first_event_date", "last_event_date"],
    )

    weekly = pd.read_csv(WEEKLY_FILE, parse_dates=["week"])
    leaderboard = pd.read_csv(LEADERBOARD_FILE)
    horizon = pd.read_csv(HORIZON_FILE)
    future = pd.read_csv(FORECAST_FILE, parse_dates=["week_start"])

    return events, risk, weekly, leaderboard, horizon, future


events, risk, weekly, leaderboard, horizon, future = load_data()

leaderboard = leaderboard.sort_values(
    ["mean_RMSE", "mean_MAE"],
    ascending=True
).reset_index(drop=True)

selected_model = leaderboard.iloc[0]
selected_model_name = str(selected_model["model"])


# ============================================================
# HELPERS
# ============================================================

def normalize_name(value):
    value = str(value).upper().strip()
    return re.sub(r"[^A-Z0-9]+", "", value)


RISK_COLORS = {
    "High": [205, 65, 65, 190],
    "Medium": [230, 160, 40, 185],
    "Low": [45, 150, 95, 175],
    "No matched history": [170, 170, 170, 70],
}


@st.cache_data(ttl=3600, show_spinner=False)
def load_official_suburbs_geojson():
    params = {
        "where": "1=1",
        "outFields": "OBJECTID,SUBURB,SUBURB_ID,DISTRICT",
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "geojson",
    }

    response = requests.get(
        SUBURBS_QUERY,
        params=params,
        timeout=60,
    )
    response.raise_for_status()

    payload = response.json()

    if "features" not in payload:
        raise RuntimeError("The eThekwini GIS service returned no suburb features.")

    return payload


def build_risk_geojson(risk_frame):
    geojson = load_official_suburbs_geojson()

    lookup = {}
    for _, row in risk_frame.iterrows():
        lookup[normalize_name(row["community"])] = {
            "community": row["community"],
            "event_count": int(row["event_count"]),
            "historical_risk_score": float(row["historical_risk_score"]),
            "historical_risk_band": str(row["historical_risk_band"]),
            "last_event_date": row["last_event_date"].date().isoformat()
            if pd.notna(row["last_event_date"])
            else "Unknown",
        }

    matched = 0

    for feature in geojson["features"]:
        props = feature.setdefault("properties", {})
        official_name = props.get("SUBURB", "")
        match = lookup.get(normalize_name(official_name))

        if match:
            matched += 1
            props.update(match)
            props["map_risk_band"] = match["historical_risk_band"]
            props["fill_color"] = RISK_COLORS.get(
                match["historical_risk_band"],
                RISK_COLORS["No matched history"],
            )
        else:
            props["community"] = official_name
            props["event_count"] = 0
            props["historical_risk_score"] = None
            props["historical_risk_band"] = "No matched history"
            props["last_event_date"] = "No matched event history"
            props["map_risk_band"] = "No matched history"
            props["fill_color"] = RISK_COLORS["No matched history"]

    return geojson, matched


def risk_badge(label):
    icons = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}
    return f"{icons.get(str(label), '⚪')} {label}"


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("💧 AquaSafe Durban")
st.sidebar.caption("Random Forest community-water forecasting")

year_options = sorted(events["year"].unique().tolist())
selected_years = st.sidebar.multiselect(
    "Event years",
    options=year_options,
    default=year_options,
)

category_options = sorted(events["event_category"].dropna().unique().tolist())
selected_categories = st.sidebar.multiselect(
    "Event categories",
    options=category_options,
    default=category_options,
)

risk_options = ["High", "Medium", "Low"]
selected_risk = st.sidebar.multiselect(
    "Historical risk bands",
    options=risk_options,
    default=risk_options,
)

community_search = st.sidebar.text_input(
    "Search a community",
    placeholder="e.g. Umlazi",
)

filtered_events = events[
    events["year"].isin(selected_years)
    & events["event_category"].isin(selected_categories)
].copy()

filtered_risk = risk[
    risk["historical_risk_band"].isin(selected_risk)
].copy()

if community_search.strip():
    q = community_search.strip()
    filtered_events = filtered_events[
        filtered_events["affected_area_normalized"].str.contains(
            q, case=False, na=False
        )
    ]
    filtered_risk = filtered_risk[
        filtered_risk["community"].str.contains(
            q, case=False, na=False
        )
    ]


# ============================================================
# HEADER
# ============================================================

st.title("💧 AquaSafe Durban")
st.subheader("Community Water Problems, Historical Risk and Random Forest Forecasting")

st.caption(
    "Built from factual eThekwini community water events documented in official "
    "municipal notices. Linear Regression, Decision Tree and Random Forest were "
    "compared with walk-forward validation; Random Forest was selected by lowest mean RMSE."
)


# ============================================================
# KPI ROW
# ============================================================

latest_event = events["event_date"].max()
independent_events = events["event_id"].nunique()
unique_areas = events["affected_area_normalized"].nunique()
latest_week_count = int(weekly.iloc[-1]["affected_area_records"])

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Independent events",
    f"{independent_events:,}",
    help="A single event may affect multiple communities.",
)

c2.metric(
    "Affected-area labels",
    f"{unique_areas:,}",
)

c3.metric(
    "Latest recorded event",
    latest_event.strftime("%d %b %Y"),
)

c4.metric(
    "Latest weekly affected areas",
    f"{latest_week_count:,}",
)


# ============================================================
# MAIN TABS
# ============================================================

overview_tab, map_tab, forecast_tab, model_tab, events_tab, data_tab = st.tabs(
    [
        "Overview",
        "Community Risk Map",
        "Forecasting",
        "Model Evaluation",
        "Events & Sources",
        "About the Data",
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

with overview_tab:
    left, right = st.columns([1.45, 1])

    with left:
        st.markdown("### Historical weekly activity")

        weekly_plot = weekly[
            ["week", "affected_area_records", "incident_count"]
        ].copy()

        fig = px.line(
            weekly_plot,
            x="week",
            y="affected_area_records",
            markers=True,
            labels={
                "week": "Week",
                "affected_area_records": "Affected-area records",
            },
        )
        fig.update_layout(
            height=390,
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Zero weeks mean that this public-notice dataset contains no affected-area "
            "record for that week. That is not proof that no household experienced a fault."
        )

    with right:
        st.markdown("### Highest historical community risk")

        top = (
            filtered_risk[
                [
                    "community",
                    "event_count",
                    "historical_risk_score",
                    "historical_risk_band",
                ]
            ]
            .sort_values("historical_risk_score", ascending=False)
            .head(12)
            .copy()
        )

        top["risk"] = top["historical_risk_band"].map(risk_badge)
        top["historical_risk_score"] = top["historical_risk_score"].round(1)

        st.dataframe(
            top[
                ["community", "event_count", "historical_risk_score", "risk"]
            ],
            hide_index=True,
            use_container_width=True,
            column_config={
                "community": "Community / Area",
                "event_count": "Events",
                "historical_risk_score": st.column_config.ProgressColumn(
                    "Risk score",
                    min_value=0,
                    max_value=100,
                    format="%.1f",
                ),
                "risk": "Band",
            },
        )

    st.markdown("### Water-event categories")

    event_level = filtered_events[
        ["event_id", "event_category"]
    ].drop_duplicates()

    category_counts = (
        event_level["event_category"]
        .value_counts()
        .rename_axis("event_category")
        .reset_index(name="independent_events")
    )

    if not category_counts.empty:
        fig = px.bar(
            category_counts,
            x="independent_events",
            y="event_category",
            orientation="h",
            labels={
                "independent_events": "Independent events",
                "event_category": "Event category",
            },
        )
        fig.update_layout(
            height=max(380, 28 * len(category_counts)),
            yaxis={"categoryorder": "total ascending"},
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No event records match the sidebar filters.")


# ============================================================
# COMMUNITY RISK MAP
# ============================================================

with map_tab:
    st.markdown("### Real eThekwini suburb map")

    st.write(
        "The map queries the official eThekwini Municipality suburb polygon layer. "
        "Only community names that match an official suburb polygon receive a risk score."
    )

    st.info(
        "Historical risk is based on past documented events. It does not mean that a "
        "community currently has no water or that a future interruption is certain."
    )

    map_filter = st.radio(
        "Map layer",
        ["All risk bands", "High only", "High + Medium"],
        horizontal=True,
    )

    try:
        geojson, matched = build_risk_geojson(risk)

        if map_filter != "All risk bands":
            allowed = (
                {"High"}
                if map_filter == "High only"
                else {"High", "Medium"}
            )

            for feature in geojson["features"]:
                props = feature["properties"]
                if props["historical_risk_band"] not in allowed:
                    props["fill_color"] = [175, 175, 175, 35]

        layer = pdk.Layer(
            "GeoJsonLayer",
            geojson,
            pickable=True,
            stroked=True,
            filled=True,
            get_fill_color="properties.fill_color",
            get_line_color=[90, 90, 90, 100],
            line_width_min_pixels=0.5,
            auto_highlight=True,
        )

        view_state = pdk.ViewState(
            latitude=-29.8587,
            longitude=31.0218,
            zoom=9.25,
            pitch=0,
        )

        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            map_style=None,
            tooltip={
                "html": (
                    "<b>{SUBURB}</b><br/>"
                    "District: {DISTRICT}<br/>"
                    "Risk: {historical_risk_band}<br/>"
                    "Risk score: {historical_risk_score}<br/>"
                    "Independent events: {event_count}<br/>"
                    "Last event: {last_event_date}"
                )
            },
        )

        st.pydeck_chart(deck, use_container_width=True)

        st.caption(
            f"{matched} official suburb polygons matched the current historical "
            "community-risk table. Non-matches remain grey."
        )

        legend_cols = st.columns(4)
        legend_cols[0].markdown("🔴 **High historical risk**")
        legend_cols[1].markdown("🟠 **Medium historical risk**")
        legend_cols[2].markdown("🟢 **Low historical risk**")
        legend_cols[3].markdown("⚪ **No matched history**")

    except Exception as exc:
        st.warning(
            "The live eThekwini GIS map could not be loaded. "
            "Your internet connection or the municipal GIS service may be unavailable."
        )
        st.code(str(exc))

        st.markdown("#### Fallback: top community risk table")
        fallback = risk[
            [
                "community",
                "event_count",
                "last_event_date",
                "historical_risk_score",
                "historical_risk_band",
            ]
        ].head(30)
        st.dataframe(fallback, hide_index=True, use_container_width=True)


# ============================================================
# FORECASTING
# ============================================================

with forecast_tab:
    st.markdown("### 7 / 14 / 30-day system forecast")

    st.success(
        f"Final model: {selected_model_name}. "
        f"It was selected from Linear Regression, Decision Tree and Random Forest "
        f"using the lowest 5-fold walk-forward mean RMSE "
        f"({selected_model['mean_RMSE']:.3f})."
    )

    st.caption(
        f"Secondary validation metric: mean MAE = "
        f"{selected_model['mean_MAE']:.3f}. "
        "The forecast remains exploratory because the source dataset contains a "
        "limited number of independent municipal events."
    )

    horizon_cols = st.columns(3)
    for i, row in horizon.iterrows():
        horizon_cols[i].metric(
            str(row["horizon"]),
            f"{float(row['forecast_affected_area_records']):.2f}",
            help="Forecast affected-area records across eThekwini, not guaranteed suburb outages.",
        )

    st.caption(
        "The 14-day and 30-day values are accumulated system-wide affected-area "
        "record forecasts. They should not be interpreted as the probability of a "
        "specific household losing water."
    )

    st.markdown("### 12-week forecast")

    fig = px.line(
        future,
        x="week_start",
        y="forecast_affected_area_records",
        markers=True,
        labels={
            "week_start": "Week",
            "forecast_affected_area_records": "Forecast affected-area records",
        },
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Historical + future context")

    history_plot = weekly[
        ["week", "affected_area_records"]
    ].tail(52).rename(
        columns={
            "week": "date",
            "affected_area_records": "value",
        }
    )
    history_plot["series"] = "Historical"

    future_plot = future.rename(
        columns={
            "week_start": "date",
            "forecast_affected_area_records": "value",
        }
    )[["date", "value"]]
    future_plot["series"] = "Forecast"

    combined = pd.concat([history_plot, future_plot], ignore_index=True)

    fig = px.line(
        combined,
        x="date",
        y="value",
        color="series",
        markers=True,
        labels={
            "date": "Date",
            "value": "Affected-area records",
            "series": "",
        },
    )
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# MODEL EVALUATION
# ============================================================

with model_tab:
    st.markdown("### Three-model walk-forward comparison")

    st.write(
        "AquaSafe compares exactly three regression models: Linear Regression, "
        "Decision Tree Regression and Random Forest Regression. All three use the "
        "same time-series features and the same 5-fold walk-forward validation."
    )

    display_board = leaderboard.copy()

    for col in ["mean_MAE", "std_MAE", "mean_RMSE", "std_RMSE", "mean_R2"]:
        if col in display_board.columns:
            display_board[col] = display_board[col].round(3)

    display_board["selected"] = np.where(
        display_board["model"] == selected_model_name,
        "✓ Final model",
        ""
    )

    st.dataframe(
        display_board[
            ["model", "mean_MAE", "mean_RMSE", "mean_R2", "selected"]
        ],
        hide_index=True,
        use_container_width=True,
        column_config={
            "model": "Model",
            "mean_MAE": st.column_config.NumberColumn(
                "Mean MAE",
                format="%.3f",
                help="Average absolute forecasting error. Lower is better."
            ),
            "mean_RMSE": st.column_config.NumberColumn(
                "Mean RMSE",
                format="%.3f",
                help="Primary selection metric. Lower is better."
            ),
            "mean_R2": st.column_config.NumberColumn(
                "Mean R²",
                format="%.3f",
                help="Additional diagnostic metric."
            ),
            "selected": "Selection",
        },
    )

    st.markdown("### RMSE comparison")

    fig = px.bar(
        leaderboard.sort_values("mean_RMSE", ascending=True),
        x="mean_RMSE",
        y="model",
        orientation="h",
        text="mean_RMSE",
        labels={
            "mean_RMSE": "Mean RMSE (lower is better)",
            "model": "Model",
        },
    )
    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside"
    )
    fig.update_layout(
        height=390,
        yaxis={"categoryorder": "total descending"},
        margin=dict(l=10, r=60, t=20, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    runner_up = leaderboard.iloc[1]

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Selected model",
        selected_model_name,
        f"RMSE {selected_model['mean_RMSE']:.3f}",
    )

    m2.metric(
        "Mean MAE",
        f"{selected_model['mean_MAE']:.3f}",
    )

    m3.metric(
        "Next-best RMSE",
        f"{runner_up['mean_RMSE']:.3f}",
        runner_up["model"],
    )

    st.markdown(
        f"""
        **Why {selected_model_name}?**

        Linear Regression, Decision Tree and Random Forest were evaluated using the
        same chronological validation folds.

        The final model is selected automatically using **mean RMSE**. RMSE gives
        more weight to large forecasting errors, which is useful when large
        water-disruption spikes are important.

        **{selected_model_name} achieved the lowest mean RMSE of
        {selected_model['mean_RMSE']:.3f}.**

        The mean R² values are negative, which shows that this sparse public-notice
        dataset is difficult to forecast. AquaSafe should therefore be treated as an
        educational / exploratory forecasting system, not an operational municipal
        outage-warning service.
        """
    )

# ============================================================
# EVENTS AND OFFICIAL SOURCES
# ============================================================

with events_tab:
    st.markdown("### Community water events")

    event_count = filtered_events["event_id"].nunique()
    area_count = filtered_events["affected_area_normalized"].nunique()

    a, b, c = st.columns(3)
    a.metric("Filtered independent events", f"{event_count:,}")
    b.metric("Filtered areas", f"{area_count:,}")
    c.metric("Filtered affected-area rows", f"{len(filtered_events):,}")

    event_summary = (
        filtered_events.groupby(
            [
                "event_id",
                "event_date",
                "event_category",
                "planned_flag",
                "source_title",
                "source_url",
            ],
            as_index=False,
        )
        .agg(
            affected_areas=(
                "affected_area_normalized",
                lambda s: ", ".join(sorted(set(s))),
            ),
            affected_area_count=(
                "affected_area_normalized",
                "nunique",
            ),
        )
        .sort_values("event_date", ascending=False)
    )

    if event_summary.empty:
        st.info("No events match the current sidebar filters.")
    else:
        event_summary["planned"] = event_summary["planned_flag"].map(
            {0: "Unplanned / emergency", 1: "Planned"}
        )

        st.dataframe(
            event_summary[
                [
                    "event_date",
                    "event_category",
                    "planned",
                    "affected_area_count",
                    "source_title",
                    "source_url",
                ]
            ],
            hide_index=True,
            use_container_width=True,
            column_config={
                "event_date": st.column_config.DateColumn(
                    "Date",
                    format="DD MMM YYYY",
                ),
                "event_category": "Category",
                "planned": "Type",
                "affected_area_count": "Affected areas",
                "source_title": "Official notice",
                "source_url": st.column_config.LinkColumn(
                    "Source",
                    display_text="Open source",
                ),
            },
        )

        st.markdown("### Inspect an event")

        selected_event = st.selectbox(
            "Choose an event",
            event_summary["event_id"].tolist(),
            format_func=lambda event_id: (
                f"{event_id} — "
                f"{event_summary.loc[event_summary['event_id'] == event_id, 'source_title'].iloc[0]}"
            ),
        )

        row = event_summary[
            event_summary["event_id"] == selected_event
        ].iloc[0]

        st.markdown(
            f"""
            <div class="aquasafe-card">
            <b>{row['source_title']}</b><br/>
            Date: {pd.to_datetime(row['event_date']).strftime('%d %B %Y')}<br/>
            Category: {row['event_category']}<br/>
            Type: {row['planned']}<br/>
            Number of affected areas: {int(row['affected_area_count'])}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("**Affected communities / areas:**")
        st.write(row["affected_areas"])

        st.link_button(
            "Open official municipal source",
            row["source_url"],
        )


# ============================================================
# DATA NOTES
# ============================================================

with data_tab:
    st.markdown("### What this dataset is")

    st.success(
        "The source events are factual community water events documented in "
        "official eThekwini Municipality public notices."
    )

    st.markdown(
        f"""
        - **Affected-area rows:** {len(events):,}
        - **Independent municipal events:** {events['event_id'].nunique():,}
        - **Unique community / area labels:** {events['affected_area_normalized'].nunique():,}
        - **Coverage:** {events['event_date'].min().strftime('%d %b %Y')} to
          {events['event_date'].max().strftime('%d %b %Y')}
        - **Municipality:** eThekwini Metropolitan Municipality
        - **Province:** KwaZulu-Natal
        - **Country:** South Africa
        """
    )

    st.markdown("### What this dataset is not")

    st.warning(
        "It is not the complete raw eThekwini operational fault-log database. "
        "The CSV was compiled from factual official municipal notices. "
        "One event can affect many communities, so affected-area rows must not be "
        "treated as independent incidents."
    )

    st.markdown("### Historical risk formula")

    st.code(
        "Historical risk score = "
        "50% frequency percentile + "
        "30% recency score + "
        "20% unplanned-event share"
    )

    st.markdown("### Forecasting scope")

    st.write(
        "The **Random Forest** forecast estimates future **system-wide affected-area activity** "
        "from the historical public-notice series. It does not claim that a "
        "particular suburb will definitely experience an interruption."
    )

    st.markdown("### Privacy")

    st.write(
        "The AquaSafe project does not use resident names, telephone numbers "
        "or email addresses."
    )


st.divider()
st.caption(
    "AquaSafe Durban • Linear Regression vs Decision Tree vs Random Forest • "
    "Final model: Random Forest • Factual eThekwini community water-event data."
)
