import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import Dense


st.set_page_config(page_title="PrognosAI Dashboard", layout="wide")

SENSOR_COLS = [
    "op1",
    "op2",
    "op3",
    "sensor2",
    "sensor3",
    "sensor4",
    "sensor7",
    "sensor8",
    "sensor9",
    "sensor11",
    "sensor12",
    "sensor13",
    "sensor14",
    "sensor15",
    "sensor17",
    "sensor20",
    "sensor21",
]
WINDOW_SIZE = 30
CRITICAL_THRESHOLD = 50
WARNING_THRESHOLD = 85
TRUTH_PATH = r"cmaps combined datasets\CMAPSS_rul(actual).csv"


class SafeDense(Dense):
    def __init__(self, **kwargs):
        kwargs.pop("quantization_config", None)
        super().__init__(**kwargs)


@st.cache_resource
def load_assets():
    try:
        custom_objects = {"Dense": SafeDense}
        lstm = tf.keras.models.load_model(
            "models/lstm_rul_model.keras", custom_objects=custom_objects
        )
        gru = tf.keras.models.load_model(
            "models/gru_rul_model.keras", custom_objects=custom_objects
        )
        feature_scaler = joblib.load("models/feature_scaler.pkl")
        target_scaler = joblib.load("models/target_scaler.pkl")
        return lstm, gru, feature_scaler, target_scaler
    except Exception as exc:
        st.error(f"Unable to load saved models and scalers: {exc}")
        return None, None, None, None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean.columns = clean.columns.str.strip().str.lower()
    return clean


def find_id_column(df: pd.DataFrame) -> str | None:
    return next((col for col in df.columns if col in ["id", "unit", "unit_id"]), None)


def find_cycle_column(df: pd.DataFrame) -> str | None:
    return next((col for col in df.columns if col in ["cycle", "time", "timestep"]), None)


def classify_alert(rul_value: float) -> str:
    if rul_value <= CRITICAL_THRESHOLD:
        return "Critical"
    if rul_value <= WARNING_THRESHOLD:
        return "Warning"
    return "Healthy"


def alert_color(alert: str) -> str:
    return {
        "Critical": "#D7263D",
        "Warning": "#F4A259",
        "Healthy": "#2A9D8F",
    }.get(alert, "#457B9D")


def inverse_predict(sequence: np.ndarray, lstm_model, gru_model, target_scaler) -> tuple[float, float, float]:
    reshaped = sequence.reshape(1, WINDOW_SIZE, len(SENSOR_COLS))
    lstm_pred = target_scaler.inverse_transform(lstm_model.predict(reshaped, verbose=0))[0][0]
    gru_pred = target_scaler.inverse_transform(gru_model.predict(reshaped, verbose=0))[0][0]
    ensemble = (0.4 * lstm_pred) + (0.6 * gru_pred)
    return max(0.0, lstm_pred), max(0.0, gru_pred), max(0.0, ensemble)


def load_truth_values(unit_count: int) -> np.ndarray:
    if not os.path.exists(TRUTH_PATH):
        return np.array([])
    truth_raw = pd.read_csv(TRUTH_PATH, header=None).iloc[:, 0]
    return pd.to_numeric(truth_raw, errors="coerce").dropna().values[:unit_count]


def build_prediction_table(
    test_df: pd.DataFrame,
    id_col: str,
    cycle_col: str | None,
    lstm_model,
    gru_model,
    feature_scaler,
    target_scaler,
) -> pd.DataFrame:
    units = sorted(test_df[id_col].unique())
    actual_values = load_truth_values(len(units))
    records = []

    for idx, unit in enumerate(units):
        unit_data = test_df[test_df[id_col] == unit].copy()
        if cycle_col:
            unit_data = unit_data.sort_values(cycle_col)
        tail = unit_data.tail(WINDOW_SIZE)

        if len(tail) == WINDOW_SIZE:
            scaled = feature_scaler.transform(tail[SENSOR_COLS])
            pred_lstm, pred_gru, pred_ensemble = inverse_predict(
                scaled, lstm_model, gru_model, target_scaler
            )
        else:
            pred_lstm, pred_gru, pred_ensemble = 0.0, 0.0, 0.0

        actual_rul = float(actual_values[idx]) if idx < len(actual_values) else np.nan
        current_cycle = int(unit_data[cycle_col].max()) if cycle_col else len(unit_data)
        error = abs(actual_rul - pred_ensemble) if not np.isnan(actual_rul) else np.nan

        records.append(
            {
                "Engine": int(unit),
                "Current Cycle": current_cycle,
                "Actual RUL": actual_rul,
                "Predicted RUL": pred_ensemble,
                "LSTM RUL": pred_lstm,
                "GRU RUL": pred_gru,
                "Absolute Error": error,
                "Alert Zone": classify_alert(pred_ensemble),
            }
        )

    prediction_df = pd.DataFrame(records)
    prediction_df["Alert Rank"] = prediction_df["Alert Zone"].map(
        {"Critical": 0, "Warning": 1, "Healthy": 2}
    )
    prediction_df = prediction_df.sort_values(
        ["Alert Rank", "Predicted RUL", "Engine"], ascending=[True, True, True]
    ).reset_index(drop=True)
    return prediction_df


def build_engine_trend(
    test_df: pd.DataFrame,
    unit_id: int,
    id_col: str,
    cycle_col: str | None,
    lstm_model,
    gru_model,
    feature_scaler,
    target_scaler,
) -> pd.DataFrame:
    unit_data = test_df[test_df[id_col] == unit_id].copy()
    if cycle_col:
        unit_data = unit_data.sort_values(cycle_col).reset_index(drop=True)
    else:
        unit_data = unit_data.reset_index(drop=True)
        unit_data["cycle"] = np.arange(1, len(unit_data) + 1)
        cycle_col = "cycle"

    records = []
    for end_idx in range(WINDOW_SIZE, len(unit_data) + 1):
        window = unit_data.iloc[end_idx - WINDOW_SIZE : end_idx]
        scaled = feature_scaler.transform(window[SENSOR_COLS])
        _, _, pred_ensemble = inverse_predict(scaled, lstm_model, gru_model, target_scaler)
        records.append(
            {
                "Cycle": int(window[cycle_col].iloc[-1]),
                "Predicted RUL": pred_ensemble,
                "Alert Zone": classify_alert(pred_ensemble),
            }
        )

    return pd.DataFrame(records)


def render_header():
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(34, 197, 94, 0.12), transparent 26%),
                    radial-gradient(circle at top right, rgba(59, 130, 246, 0.16), transparent 24%),
                    linear-gradient(180deg, #F3F7FB 0%, #EDF2F7 100%);
            }
            .block-container {padding-top: 1.2rem; padding-bottom: 1.8rem;}
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #0F172A 0%, #172554 100%);
            }
            [data-testid="stSidebar"] * {
                color: #E5EEF8;
            }
            .dashboard-shell {
                position: relative;
                overflow: hidden;
                padding: 1.5rem 1.6rem;
                border-radius: 24px;
                background: linear-gradient(135deg, rgba(15, 23, 42, 0.98), rgba(14, 116, 144, 0.95));
                color: white;
                margin-bottom: 1.2rem;
                border: 1px solid rgba(255, 255, 255, 0.12);
                box-shadow: 0 18px 45px rgba(15, 23, 42, 0.18);
            }
            .dashboard-shell::after {
                content: "";
                position: absolute;
                inset: auto -60px -90px auto;
                width: 240px;
                height: 240px;
                border-radius: 999px;
                background: rgba(125, 211, 252, 0.14);
                filter: blur(4px);
            }
            .hero-grid {
                display: grid;
                grid-template-columns: 1.8fr 1fr;
                gap: 1rem;
                align-items: center;
            }
            .hero-kicker {
                display: inline-block;
                padding: 0.35rem 0.7rem;
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.12);
                font-size: 0.8rem;
                letter-spacing: 0.04em;
                text-transform: uppercase;
            }
            .hero-title {
                margin: 0.85rem 0 0;
                font-size: 2.15rem;
                line-height: 1.1;
                font-weight: 700;
            }
            .hero-copy {
                margin: 0.75rem 0 0;
                font-size: 1rem;
                max-width: 48rem;
                color: rgba(255, 255, 255, 0.82);
            }
            .signal-panel {
                padding: 1rem 1.1rem;
                border-radius: 20px;
                background: rgba(255, 255, 255, 0.08);
                backdrop-filter: blur(8px);
                border: 1px solid rgba(255, 255, 255, 0.12);
            }
            .signal-label {
                font-size: 0.85rem;
                color: rgba(255,255,255,0.7);
                margin-bottom: 0.35rem;
            }
            .signal-value {
                font-size: 1.9rem;
                font-weight: 700;
                margin: 0;
            }
            .signal-subtext {
                margin: 0.45rem 0 0;
                color: rgba(255,255,255,0.78);
                font-size: 0.92rem;
            }
            .section-card {
                background: rgba(255, 255, 255, 0.86);
                border: 1px solid rgba(148, 163, 184, 0.18);
                box-shadow: 0 16px 38px rgba(15, 23, 42, 0.08);
                border-radius: 22px;
                padding: 1rem 1rem 0.4rem;
                margin-bottom: 1rem;
            }
            .metric-band {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 0.9rem;
                margin: 0.25rem 0 1rem;
            }
            .metric-tile {
                padding: 1rem 1rem 0.9rem;
                border-radius: 20px;
                color: #0F172A;
                border: 1px solid rgba(148, 163, 184, 0.18);
                box-shadow: inset 0 1px 0 rgba(255,255,255,0.35);
            }
            .metric-tile.navy { background: linear-gradient(180deg, #E0F2FE 0%, #F8FBFF 100%); }
            .metric-tile.sand { background: linear-gradient(180deg, #FEF3C7 0%, #FFFBEB 100%); }
            .metric-tile.mint { background: linear-gradient(180deg, #D1FAE5 0%, #F0FDF4 100%); }
            .metric-tile.rose { background: linear-gradient(180deg, #FFE4E6 0%, #FFF1F2 100%); }
            .metric-label {
                font-size: 0.84rem;
                color: #475569;
                margin-bottom: 0.35rem;
            }
            .metric-value {
                font-size: 1.7rem;
                font-weight: 700;
                margin: 0;
            }
            .metric-note {
                font-size: 0.87rem;
                color: #64748B;
                margin-top: 0.35rem;
            }
            .subtle-heading {
                margin: 0 0 0.8rem;
                color: #0F172A;
                font-size: 1.08rem;
                font-weight: 700;
            }
            @media (max-width: 900px) {
                .hero-grid, .metric-band {
                    grid-template-columns: 1fr;
                }
            }
        </style>
        <div class="dashboard-shell">
            <div class="hero-grid">
                <div>
                    <span class="hero-kicker">Predictive Maintenance Console</span>
                    <h1 class="hero-title">PrognosAI Fleet RUL Monitor</h1>
                    <p class="hero-copy">
                        Explore remaining useful life trends, spot alert-zone drift early, and inspect engine behavior through a cleaner, more decision-focused monitoring interface.
                    </p>
                </div>
                <div class="signal-panel">
                    <div class="signal-label">Live dashboard mode</div>
                    <p class="signal-value">Monitoring</p>
                    <p class="signal-subtext">Upload a CMAPSS-style file to turn raw telemetry into health signals, alert segmentation, and engine-level trend views.</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview_metrics(prediction_df: pd.DataFrame):
    avg_rul = prediction_df["Predicted RUL"].mean()
    critical_count = int((prediction_df["Alert Zone"] == "Critical").sum())
    warning_count = int((prediction_df["Alert Zone"] == "Warning").sum())
    mae = prediction_df["Absolute Error"].dropna().mean()
    health_index = max(0, min(100, int((avg_rul / 200) * 100)))
    mae_text = f"{mae:.2f} cycles" if not np.isnan(mae) else "Unavailable"

    st.markdown(
        f"""
        <div class="metric-band">
            <div class="metric-tile navy">
                <div class="metric-label">Average Predicted RUL</div>
                <p class="metric-value">{avg_rul:.1f} cycles</p>
                <div class="metric-note">Fleet-wide projected life remaining</div>
            </div>
            <div class="metric-tile rose">
                <div class="metric-label">Critical Engines</div>
                <p class="metric-value">{critical_count}</p>
                <div class="metric-note">Immediate action candidates</div>
            </div>
            <div class="metric-tile sand">
                <div class="metric-label">Warning Engines</div>
                <p class="metric-value">{warning_count}</p>
                <div class="metric-note">Maintenance watchlist</div>
            </div>
            <div class="metric-tile mint">
                <div class="metric-label">Fleet Health Index</div>
                <p class="metric-value">{health_index}/100</p>
                <div class="metric-note">MAE: {mae_text}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_fleet_status_banner(prediction_df: pd.DataFrame):
    avg_rul = prediction_df["Predicted RUL"].mean()
    critical_count = int((prediction_df["Alert Zone"] == "Critical").sum())

    if critical_count > 0:
        st.error(
            f"{critical_count} engine(s) are in the critical zone. Immediate maintenance planning is recommended."
        )
    elif avg_rul <= WARNING_THRESHOLD:
        st.warning(
            f"Fleet health is trending down. Average predicted RUL is {avg_rul:.1f} cycles."
        )
    else:
        st.success(
            f"Fleet is operating in a healthy range with an average predicted RUL of {avg_rul:.1f} cycles."
        )


def render_fleet_charts(prediction_df: pd.DataFrame):
    tab1, tab2, tab3 = st.tabs(["Fleet Trend", "Alert Zones", "Prediction Accuracy"])

    sorted_by_engine = prediction_df.sort_values("Engine")

    with tab1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<p class="subtle-heading">Fleet RUL trend across engine units</p>', unsafe_allow_html=True)
        trend_fig = go.Figure()
        trend_fig.add_trace(
            go.Scatter(
                x=sorted_by_engine["Engine"],
                y=sorted_by_engine["Predicted RUL"],
                mode="lines+markers",
                name="Predicted RUL",
                line=dict(color="#1D4ED8", width=3),
                marker=dict(
                    size=10,
                    color=[alert_color(alert) for alert in sorted_by_engine["Alert Zone"]],
                    line=dict(width=1.5, color="#FFFFFF"),
                ),
                customdata=np.stack(
                    [
                        sorted_by_engine["Current Cycle"],
                        sorted_by_engine["Alert Zone"],
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "Engine %{x}<br>"
                    "Predicted RUL: %{y:.1f} cycles<br>"
                    "Current Cycle: %{customdata[0]}<br>"
                    "Alert Zone: %{customdata[1]}<extra></extra>"
                ),
            )
        )
        trend_fig.add_hrect(y0=0, y1=CRITICAL_THRESHOLD, fillcolor="rgba(215, 38, 61, 0.15)", line_width=0)
        trend_fig.add_hrect(y0=CRITICAL_THRESHOLD, y1=WARNING_THRESHOLD, fillcolor="rgba(244, 162, 89, 0.18)", line_width=0)
        trend_fig.add_hrect(
            y0=WARNING_THRESHOLD,
            y1=max(sorted_by_engine["Predicted RUL"].max() + 20, WARNING_THRESHOLD + 20),
            fillcolor="rgba(42, 157, 143, 0.14)",
            line_width=0,
        )
        trend_fig.update_layout(
            xaxis_title="Engine Unit",
            yaxis_title="Predicted RUL (cycles)",
            legend_title="",
            height=430,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.92)",
        )
        st.plotly_chart(trend_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        chart_left, chart_right = st.columns((1, 1))
        zone_counts = (
            prediction_df["Alert Zone"]
            .value_counts()
            .reindex(["Critical", "Warning", "Healthy"], fill_value=0)
            .reset_index()
        )
        zone_counts.columns = ["Alert Zone", "Engines"]

        with chart_left:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<p class="subtle-heading">Alert-zone distribution</p>', unsafe_allow_html=True)
            zone_fig = px.bar(
                zone_counts,
                x="Alert Zone",
                y="Engines",
                color="Alert Zone",
                color_discrete_map={
                    "Critical": "#D7263D",
                    "Warning": "#F4A259",
                    "Healthy": "#2A9D8F",
                },
                text="Engines",
            )
            zone_fig.update_layout(
                showlegend=False,
                height=390,
                xaxis_title="",
                yaxis_title="Count",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.92)",
            )
            st.plotly_chart(zone_fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with chart_right:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<p class="subtle-heading">Alert-zone spread by engine</p>', unsafe_allow_html=True)
            spread_fig = px.scatter(
                sorted_by_engine,
                x="Engine",
                y="Predicted RUL",
                color="Alert Zone",
                size="Predicted RUL",
                size_max=24,
                color_discrete_map={
                    "Critical": "#D7263D",
                    "Warning": "#F4A259",
                    "Healthy": "#2A9D8F",
                },
                hover_data=["Current Cycle"],
            )
            spread_fig.update_layout(
                height=390,
                xaxis_title="Engine Unit",
                yaxis_title="Predicted RUL (cycles)",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.92)",
            )
            st.plotly_chart(spread_fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        compare_df = prediction_df.dropna(subset=["Actual RUL"]).sort_values("Engine")
        if not compare_df.empty:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<p class="subtle-heading">Actual vs predicted RUL performance</p>', unsafe_allow_html=True)
            compare_fig = go.Figure()
            compare_fig.add_trace(
                go.Scatter(
                    x=compare_df["Engine"],
                    y=compare_df["Actual RUL"],
                    mode="lines+markers",
                    name="Actual RUL",
                    line=dict(color="#111827", width=2),
                )
            )
            compare_fig.add_trace(
                go.Scatter(
                    x=compare_df["Engine"],
                    y=compare_df["Predicted RUL"],
                    mode="lines+markers",
                    name="Predicted RUL",
                    line=dict(color="#C1121F", width=3, dash="dash"),
                )
            )
            compare_fig.update_layout(
                xaxis_title="Engine Unit",
                yaxis_title="RUL (cycles)",
                height=420,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.92)",
            )
            st.plotly_chart(compare_fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)


def render_engine_detail(
    test_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    selected_engine: int,
    id_col: str,
    cycle_col: str | None,
    lstm_model,
    gru_model,
    feature_scaler,
    target_scaler,
):
    engine_row = prediction_df[prediction_df["Engine"] == selected_engine].iloc[0]
    trend_df = build_engine_trend(
        test_df,
        selected_engine,
        id_col,
        cycle_col,
        lstm_model,
        gru_model,
        feature_scaler,
        target_scaler,
    )

    st.subheader(f"Engine {selected_engine} Detail")
    info1, info2, info3 = st.columns(3)
    info1.metric("Current Predicted RUL", f"{engine_row['Predicted RUL']:.1f} cycles")
    info2.metric("Current Alert Zone", engine_row["Alert Zone"])
    info3.metric("Observed Cycle", int(engine_row["Current Cycle"]))

    unit_data = test_df[test_df[id_col] == selected_engine].copy()
    if cycle_col:
        unit_data = unit_data.sort_values(cycle_col)
        x_axis = cycle_col
    else:
        unit_data = unit_data.reset_index(drop=True)
        unit_data["cycle"] = np.arange(1, len(unit_data) + 1)
        x_axis = "cycle"

    detail_tab1, detail_tab2 = st.tabs(["RUL Trend", "Sensor Story"])

    with detail_tab1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        trend_fig = go.Figure()
        trend_fig.add_trace(
            go.Scatter(
                x=trend_df["Cycle"],
                y=trend_df["Predicted RUL"],
                mode="lines+markers",
                line=dict(color="#0F766E", width=3),
                marker=dict(
                    size=9,
                    color=[alert_color(alert) for alert in trend_df["Alert Zone"]],
                    line=dict(width=1.2, color="#FFFFFF"),
                ),
                fill="tozeroy",
                fillcolor="rgba(15, 118, 110, 0.12)",
                name="Predicted RUL",
                hovertemplate="Cycle %{x}<br>Predicted RUL: %{y:.1f}<extra></extra>",
            )
        )
        trend_fig.add_hline(y=CRITICAL_THRESHOLD, line_color="#D7263D", line_dash="dot")
        trend_fig.add_hline(y=WARNING_THRESHOLD, line_color="#F4A259", line_dash="dot")
        trend_fig.update_layout(
            title="RUL Trend Over Time",
            xaxis_title="Cycle",
            yaxis_title="Predicted RUL (cycles)",
            height=430,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.92)",
        )
        st.plotly_chart(trend_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with detail_tab2:
        sensor_options = ["sensor11", "sensor12", "sensor15", "sensor20", "sensor21"]
        available_sensors = [sensor for sensor in sensor_options if sensor in unit_data.columns]
        chosen_sensor = st.selectbox(
            "Sensor trend",
            available_sensors,
            index=0,
            key=f"sensor_select_{selected_engine}",
        )
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        sensor_fig = px.line(
            unit_data,
            x=x_axis,
            y=chosen_sensor,
            title=f"{chosen_sensor.upper()} progression for Engine {selected_engine}",
            markers=True,
        )
        sensor_fig.update_traces(line=dict(color="#7C3AED", width=3))
        sensor_fig.update_layout(
            height=340,
            xaxis_title="Cycle",
            yaxis_title=chosen_sensor.upper(),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.92)",
        )
        st.plotly_chart(sensor_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


def main():
    lstm_model, gru_model, feature_scaler, target_scaler = load_assets()
    render_header()

    st.sidebar.header("Data Input")
    uploaded_test = st.sidebar.file_uploader(
        "Upload test sensor data", type="csv", help="Use a CMAPSS-style test CSV."
    )

    st.sidebar.header("Alert Zones")
    st.sidebar.caption(f"Critical: 0 to {CRITICAL_THRESHOLD} cycles")
    st.sidebar.caption(
        f"Warning: {CRITICAL_THRESHOLD + 1} to {WARNING_THRESHOLD} cycles"
    )
    st.sidebar.caption(f"Healthy: above {WARNING_THRESHOLD} cycles")

    if not uploaded_test:
        st.info("Upload a test CSV to open the RUL monitoring dashboard.")
        return

    if not all([lstm_model, gru_model, feature_scaler, target_scaler]):
        st.error("Model assets are not available. Please check the `models/` directory.")
        return

    test_df = normalize_columns(pd.read_csv(uploaded_test))
    id_col = find_id_column(test_df)
    cycle_col = find_cycle_column(test_df)

    if not id_col:
        st.error("The uploaded file must contain an engine identifier column such as `unit`, `id`, or `unit_id`.")
        return

    missing_cols = [col for col in SENSOR_COLS if col not in test_df.columns]
    if missing_cols:
        st.error(f"Missing required feature columns: {', '.join(missing_cols)}")
        return

    with st.spinner("Generating fleet predictions and dashboard visuals..."):
        prediction_df = build_prediction_table(
            test_df,
            id_col,
            cycle_col,
            lstm_model,
            gru_model,
            feature_scaler,
            target_scaler,
        )

    render_overview_metrics(prediction_df)
    render_fleet_status_banner(prediction_df)
    render_fleet_charts(prediction_df)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Current RUL Predictions")
    display_df = prediction_df.drop(columns=["Alert Rank"]).copy()
    st.dataframe(
        display_df.style.format(
            {
                "Actual RUL": "{:.1f}",
                "Predicted RUL": "{:.1f}",
                "LSTM RUL": "{:.1f}",
                "GRU RUL": "{:.1f}",
                "Absolute Error": "{:.1f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    selected_engine = st.selectbox("Inspect an engine", prediction_df["Engine"].tolist(), index=0)
    render_engine_detail(
        test_df,
        prediction_df,
        selected_engine,
        id_col,
        cycle_col,
        lstm_model,
        gru_model,
        feature_scaler,
        target_scaler,
    )
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
