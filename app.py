import os
from io import StringIO

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


def load_default_dataset() -> pd.DataFrame:
    return normalize_columns(pd.read_csv("sample_engine_test_50.csv"))


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


def predict_from_manual_sequence(
    sequence_df: pd.DataFrame,
    lstm_model,
    gru_model,
    feature_scaler,
    target_scaler,
) -> tuple[float, float, float]:
    scaled = feature_scaler.transform(sequence_df[SENSOR_COLS])
    return inverse_predict(scaled, lstm_model, gru_model, target_scaler)


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


def render_header(theme_mode: str = "Sunrise"):
    if theme_mode == "Midnight":
        app_background = """
                    radial-gradient(circle at 10% 5%, rgba(249, 115, 22, 0.12), transparent 22%),
                    radial-gradient(circle at 88% 8%, rgba(139, 92, 246, 0.18), transparent 22%),
                    radial-gradient(circle at 50% 30%, rgba(6, 182, 212, 0.12), transparent 26%),
                    linear-gradient(180deg, #08111f 0%, #111827 46%, #1e1037 100%)
        """
        sidebar_background = """
                    radial-gradient(circle at top, rgba(56, 189, 248, 0.16), transparent 34%),
                    radial-gradient(circle at bottom right, rgba(236, 72, 153, 0.14), transparent 30%),
                    linear-gradient(180deg, #08111f 0%, #0f172a 48%, #1a1033 100%)
        """
        section_background = "linear-gradient(180deg, rgba(15, 23, 42, 0.92), rgba(30, 41, 59, 0.92))"
        section_border = "rgba(148, 163, 184, 0.18)"
        text_color = "#E2E8F0"
        muted_color = "#94A3B8"
        tab_background = "linear-gradient(135deg, rgba(15,23,42,0.92), rgba(30,41,59,0.92))"
        metric_background = "linear-gradient(135deg, rgba(15,23,42,0.92), rgba(30,41,59,0.96))"
    else:
        app_background = """
                    radial-gradient(circle at 10% 5%, rgba(249, 115, 22, 0.18), transparent 22%),
                    radial-gradient(circle at 88% 8%, rgba(139, 92, 246, 0.16), transparent 22%),
                    radial-gradient(circle at 50% 30%, rgba(6, 182, 212, 0.10), transparent 26%),
                    linear-gradient(180deg, #f8fcff 0%, #eef5fb 46%, #fdf7ff 100%)
        """
        sidebar_background = """
                    radial-gradient(circle at top, rgba(56, 189, 248, 0.20), transparent 34%),
                    radial-gradient(circle at bottom right, rgba(236, 72, 153, 0.12), transparent 30%),
                    linear-gradient(180deg, #10233A 0%, #16324F 48%, #25133d 100%)
        """
        section_background = "linear-gradient(180deg, rgba(255, 255, 255, 0.95), rgba(248, 250, 252, 0.95))"
        section_border = "rgba(148, 163, 184, 0.16)"
        text_color = "#10233a"
        muted_color = "#5f7085"
        tab_background = "linear-gradient(135deg, rgba(255,255,255,0.82), rgba(245,247,255,0.92))"
        metric_background = "linear-gradient(135deg, rgba(255,255,255,0.96), rgba(245,248,255,0.98))"

    header_markup = """
        <style>
            :root {
                --bg-shell: #f6fbff;
                --ink: __TEXT_COLOR__;
                --muted: __MUTED_COLOR__;
                --cyan: #06b6d4;
                --teal: #14b8a6;
                --amber: #f59e0b;
                --coral: #f97316;
                --pink: #ec4899;
                --violet: #8b5cf6;
                --card-border: rgba(148, 163, 184, 0.16);
            }
            .stApp {
                font-family: "Aptos Display", "Aptos", "Trebuchet MS", "Segoe UI", sans-serif;
                background: __APP_BACKGROUND__;
                color: var(--ink);
            }
            .block-container {
                padding-top: 1.2rem;
                padding-bottom: 1.8rem;
                max-width: 1380px;
            }
            [data-testid="stSidebar"] {
                background: __SIDEBAR_BACKGROUND__;
            }
            [data-testid="stSidebar"] * {
                color: #E5EEF8;
            }
            [data-testid="stSidebar"] > div:first-child {
                border-right: 1px solid rgba(255,255,255,0.08);
            }
            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
                line-height: 1.45;
            }
            .dashboard-shell {
                position: relative;
                overflow: hidden;
                padding: 1.65rem 1.7rem;
                border-radius: 30px;
                background:
                    radial-gradient(circle at top right, rgba(251, 146, 60, 0.24), transparent 22%),
                    radial-gradient(circle at bottom left, rgba(34, 211, 238, 0.16), transparent 30%),
                    linear-gradient(135deg, #10233a 0%, #0b6b89 52%, #5b21b6 100%);
                color: white;
                margin-bottom: 1.35rem;
                border: 1px solid rgba(255, 255, 255, 0.14);
                box-shadow: 0 24px 60px rgba(36, 22, 84, 0.22);
            }
            .dashboard-shell::after {
                content: "";
                position: absolute;
                inset: auto -60px -90px auto;
                width: 280px;
                height: 280px;
                border-radius: 999px;
                background: rgba(236, 72, 153, 0.18);
                filter: blur(8px);
            }
            .dashboard-shell::before {
                content: "";
                position: absolute;
                inset: -80px auto auto -80px;
                width: 220px;
                height: 220px;
                border-radius: 999px;
                background: rgba(34, 211, 238, 0.14);
                filter: blur(8px);
            }
            .hero-grid {
                display: grid;
                grid-template-columns: 1.8fr 1fr;
                gap: 1.2rem;
                align-items: center;
            }
            .hero-kicker {
                display: inline-block;
                padding: 0.42rem 0.82rem;
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.14);
                border: 1px solid rgba(255,255,255,0.14);
                font-size: 0.78rem;
                letter-spacing: 0.04em;
                text-transform: uppercase;
            }
            .hero-title {
                margin: 0.9rem 0 0;
                font-size: 2.6rem;
                line-height: 1.1;
                font-weight: 800;
                max-width: 14ch;
            }
            .hero-copy {
                margin: 0.75rem 0 0;
                font-size: 1.02rem;
                max-width: 48rem;
                color: rgba(255, 255, 255, 0.82);
            }
            .hero-badges {
                display: flex;
                flex-wrap: wrap;
                gap: 0.55rem;
                margin-top: 1rem;
            }
            .hero-badge {
                padding: 0.48rem 0.8rem;
                border-radius: 999px;
                background: rgba(255,255,255,0.10);
                border: 1px solid rgba(255,255,255,0.12);
                font-size: 0.86rem;
                color: rgba(255,255,255,0.88);
            }
            .signal-panel {
                padding: 1.05rem 1.15rem;
                border-radius: 24px;
                background: linear-gradient(180deg, rgba(255, 255, 255, 0.14), rgba(255, 255, 255, 0.08));
                backdrop-filter: blur(8px);
                border: 1px solid rgba(255, 255, 255, 0.14);
                box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
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
            .pulse-row {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 0.75rem;
                margin-top: 1rem;
            }
            .pulse-card {
                padding: 0.88rem 0.9rem;
                border-radius: 18px;
                background: rgba(255,255,255,0.08);
                border: 1px solid rgba(255,255,255,0.10);
            }
            .pulse-label {
                font-size: 0.78rem;
                color: rgba(255,255,255,0.7);
                text-transform: uppercase;
                letter-spacing: 0.03em;
            }
            .pulse-value {
                margin: 0.2rem 0 0;
                font-size: 1.15rem;
                font-weight: 800;
            }
            .section-card {
                background: __SECTION_BACKGROUND__;
                border: 1px solid __SECTION_BORDER__;
                box-shadow: 0 20px 40px rgba(15, 23, 42, 0.08);
                border-radius: 24px;
                padding: 1rem 1rem 0.5rem;
                margin-bottom: 1.05rem;
            }
            .metric-band {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 0.9rem;
                margin: 0.25rem 0 1rem;
            }
            .metric-tile {
                padding: 1rem 1rem 0.9rem;
                border-radius: 24px;
                color: #0F172A;
                border: 1px solid rgba(255,255,255,0.45);
                box-shadow: inset 0 1px 0 rgba(255,255,255,0.35), 0 16px 32px rgba(15,23,42,0.06);
            }
            .metric-tile.navy { background: linear-gradient(135deg, #c8f3ff 0%, #ebfbff 48%, #f8fbff 100%); }
            .metric-tile.sand { background: linear-gradient(135deg, #ffe0a3 0%, #fff0cf 50%, #fffaf0 100%); }
            .metric-tile.mint { background: linear-gradient(135deg, #bcf6dd 0%, #e5fff2 50%, #f5fff9 100%); }
            .metric-tile.rose { background: linear-gradient(135deg, #ffd2ec 0%, #ffe8f4 48%, #fff6fb 100%); }
            .metric-label {
                font-size: 0.84rem;
                color: #475569;
                margin-bottom: 0.35rem;
                text-transform: uppercase;
                letter-spacing: 0.04em;
            }
            .metric-value {
                font-size: 1.9rem;
                font-weight: 800;
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
                font-weight: 800;
            }
            .action-strip {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 1rem;
                margin: 0.45rem 0 1.2rem;
            }
            .action-tile {
                position: relative;
                overflow: hidden;
                padding: 1.05rem 1rem 0.95rem;
                border-radius: 22px;
                background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(245,248,255,0.98));
                border: 1px solid var(--card-border);
                box-shadow: 0 16px 34px rgba(15, 23, 42, 0.06);
            }
            .action-tile::after {
                content: "";
                position: absolute;
                inset: auto -30px -40px auto;
                width: 120px;
                height: 120px;
                border-radius: 999px;
                opacity: 0.18;
            }
            .action-tile.teal::after { background: var(--teal); }
            .action-tile.orange::after { background: var(--coral); }
            .action-tile.violet::after { background: var(--violet); }
            .action-title {
                margin: 0 0 0.25rem;
                color: #0F172A;
                font-size: 1rem;
                font-weight: 800;
            }
            .action-copy {
                margin: 0;
                color: var(--muted);
                font-size: 0.9rem;
            }
            .panel-shell {
                padding: 1rem 1rem 0.9rem;
                border-radius: 22px;
                background: __SECTION_BACKGROUND__;
                border: 1px solid __SECTION_BORDER__;
                margin-bottom: 1rem;
            }
            .panel-title {
                margin: 0;
                color: var(--ink);
                font-size: 1rem;
                font-weight: 800;
            }
            .panel-copy {
                margin: 0.35rem 0 0;
                color: var(--muted);
                font-size: 0.92rem;
            }
            .status-pill {
                display: inline-block;
                padding: 0.32rem 0.7rem;
                border-radius: 999px;
                background: rgba(14, 165, 233, 0.12);
                color: #075985;
                font-size: 0.82rem;
                font-weight: 700;
                margin-bottom: 0.6rem;
            }
            .story-grid {
                display: grid;
                grid-template-columns: 1.15fr 0.85fr;
                gap: 1rem;
                align-items: stretch;
                margin-bottom: 1rem;
            }
            .story-card {
                padding: 1.1rem 1.1rem 1rem;
                border-radius: 26px;
                background: __SECTION_BACKGROUND__;
                border: 1px solid __SECTION_BORDER__;
                box-shadow: 0 16px 34px rgba(15, 23, 42, 0.06);
            }
            .story-title {
                margin: 0;
                color: var(--ink);
                font-size: 1.24rem;
                font-weight: 800;
            }
            .story-copy {
                margin: 0.65rem 0 0;
                color: var(--muted);
                font-size: 0.96rem;
                line-height: 1.6;
            }
            .mini-grid {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 0.8rem;
                margin-top: 0.95rem;
            }
            .mini-card {
                padding: 0.95rem;
                border-radius: 18px;
                background: linear-gradient(135deg, #ecfeff 0%, #eff6ff 100%);
                border: 1px solid rgba(148, 163, 184, 0.16);
            }
            .mini-title {
                margin: 0;
                color: var(--ink);
                font-size: 0.92rem;
                font-weight: 800;
            }
            .mini-copy {
                margin: 0.35rem 0 0;
                color: var(--muted);
                font-size: 0.87rem;
                line-height: 1.45;
            }
            .highlight-band {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 0.9rem;
                margin-top: 1rem;
            }
            .highlight-tile {
                padding: 1rem;
                border-radius: 20px;
                color: white;
                box-shadow: 0 16px 34px rgba(15,23,42,0.12);
            }
            .highlight-tile.cyan { background: linear-gradient(135deg, #0891b2 0%, #22d3ee 100%); }
            .highlight-tile.orange { background: linear-gradient(135deg, #ea580c 0%, #fb7185 100%); }
            .highlight-tile.violet { background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%); }
            .highlight-title {
                margin: 0;
                font-size: 0.9rem;
                font-weight: 800;
            }
            .highlight-copy {
                margin: 0.35rem 0 0;
                font-size: 0.86rem;
                line-height: 1.45;
                color: rgba(255,255,255,0.86);
            }
            div[data-testid="stButton"] > button,
            div[data-testid="stDownloadButton"] > button,
            div[data-testid="stFormSubmitButton"] > button {
                width: 100%;
                border-radius: 16px;
                border: 0;
                background: linear-gradient(135deg, #f97316 0%, #ec4899 48%, #8b5cf6 100%);
                color: white;
                font-weight: 800;
                padding: 0.7rem 0.95rem;
                box-shadow: 0 16px 28px rgba(139, 92, 246, 0.22);
                transition: transform 0.18s ease, box-shadow 0.18s ease;
            }
            div[data-testid="stButton"] > button:hover,
            div[data-testid="stDownloadButton"] > button:hover,
            div[data-testid="stFormSubmitButton"] > button:hover {
                transform: translateY(-2px);
                box-shadow: 0 20px 34px rgba(236, 72, 153, 0.24);
            }
            div[data-baseweb="tab-list"] {
                gap: 0.45rem;
                background: __TAB_BACKGROUND__;
                border-radius: 18px;
                padding: 0.34rem;
                border: 1px solid rgba(148, 163, 184, 0.14);
                box-shadow: 0 10px 22px rgba(15,23,42,0.04);
            }
            button[data-baseweb="tab"] {
                border-radius: 14px;
                font-weight: 800;
                color: #334155;
                padding: 0.62rem 0.96rem;
            }
            button[data-baseweb="tab"][aria-selected="true"] {
                background: linear-gradient(135deg, rgba(249, 115, 22, 0.16), rgba(168, 85, 247, 0.16));
                color: var(--ink);
                box-shadow: inset 0 1px 0 rgba(255,255,255,0.4);
            }
            div[data-testid="stMetric"] {
                background: __METRIC_BACKGROUND__;
                border: 1px solid rgba(148, 163, 184, 0.16);
                padding: 0.82rem 0.9rem;
                border-radius: 20px;
                box-shadow: 0 14px 26px rgba(15, 23, 42, 0.06);
            }
            div[data-testid="stDataFrame"] {
                border-radius: 18px;
                overflow: hidden;
                border: 1px solid rgba(148,163,184,0.12);
                box-shadow: 0 14px 30px rgba(15,23,42,0.06);
            }
            div[data-testid="stFileUploader"] section {
                border-radius: 18px;
                background: rgba(255,255,255,0.08);
            }
            [data-testid="stMetricValue"], [data-testid="stMetricLabel"], .subtle-heading, h1, h2, h3, label, .stMarkdown, .stCaption {
                color: var(--ink);
            }
            .action-tile, .story-card, .section-card, .panel-shell, div[data-testid="stMetric"] {
                animation: riseIn 0.45s ease;
            }
            .action-tile:hover, .story-card:hover, .section-card:hover, .panel-shell:hover {
                transform: translateY(-3px);
                transition: transform 0.18s ease, box-shadow 0.18s ease;
                box-shadow: 0 24px 42px rgba(15, 23, 42, 0.10);
            }
            @keyframes riseIn {
                from {
                    opacity: 0;
                    transform: translateY(12px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            @media (max-width: 900px) {
                .hero-grid, .metric-band, .action-strip, .story-grid, .mini-grid, .highlight-band, .pulse-row {
                    grid-template-columns: 1fr;
                }
                .hero-title {
                    font-size: 2rem;
                }
            }
        </style>
        <div class="dashboard-shell">
            <div class="hero-grid">
                <div>
                    <span class="hero-kicker">Predictive Maintenance Console</span>
                    <h1 class="hero-title">PrognosAI Fleet RUL Monitor</h1>
                    <p class="hero-copy">
                        Explore remaining useful life trends, spot alert-zone drift early, and inspect engine behavior through a vivid, decision-ready monitoring interface designed for demos and real analysis.
                    </p>
                    <div class="hero-badges">
                        <span class="hero-badge">Color-coded fleet health</span>
                        <span class="hero-badge">Manual GRU predictor</span>
                        <span class="hero-badge">One-click report export</span>
                    </div>
                </div>
                <div class="signal-panel">
                    <div class="signal-label">Live dashboard mode</div>
                    <p class="signal-value">Monitoring</p>
                    <p class="signal-subtext">Upload a CMAPSS-style file to turn raw telemetry into health signals, alert segmentation, and engine-level trend views.</p>
                    <div class="pulse-row">
                        <div class="pulse-card">
                            <div class="pulse-label">Models</div>
                            <p class="pulse-value">LSTM + GRU</p>
                        </div>
                        <div class="pulse-card">
                            <div class="pulse-label">Views</div>
                            <p class="pulse-value">Fleet to Engine</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
    header_markup = (
        header_markup.replace("__TEXT_COLOR__", text_color)
        .replace("__MUTED_COLOR__", muted_color)
        .replace("__APP_BACKGROUND__", app_background.strip())
        .replace("__SIDEBAR_BACKGROUND__", sidebar_background.strip())
        .replace("__SECTION_BACKGROUND__", section_background)
        .replace("__SECTION_BORDER__", section_border)
        .replace("__TAB_BACKGROUND__", tab_background)
        .replace("__METRIC_BACKGROUND__", metric_background)
    )
    st.markdown(header_markup, unsafe_allow_html=True)


def render_action_strip():
    st.markdown(
        """
        <div class="action-strip">
            <div class="action-tile teal">
                <p class="action-title">Generate Reports</p>
                <p class="action-copy">Export the current fleet prediction table and a management-ready text summary from the same screen.</p>
            </div>
            <div class="action-tile orange">
                <p class="action-title">Manual GRU Response</p>
                <p class="action-copy">Enter sensor values directly or paste a 30-row sequence to get GRU, LSTM, and ensemble RUL outputs.</p>
            </div>
            <div class="action-tile violet">
                <p class="action-title">Operator Controls</p>
                <p class="action-copy">Switch between uploaded data and the bundled sample set, then inspect engine-level trends without leaving the page.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_home_tab():
    st.markdown(
        """
        <div class="story-grid">
            <div class="story-card">
                <div class="status-pill">Project Overview</div>
                <p class="story-title">Predictive maintenance, explained clearly</p>
                <p class="story-copy">
                    PrognosAI estimates Remaining Useful Life (RUL) for aircraft engines using sequential sensor data and deep learning models.
                    The dashboard is designed for both technical exploration and project demonstration, so users can move from raw telemetry to maintenance decisions in one place.
                </p>
                <div class="mini-grid">
                    <div class="mini-card">
                        <p class="mini-title">What it uses</p>
                        <p class="mini-copy">LSTM and GRU models, scaled sensor windows, and alert-based fleet ranking.</p>
                    </div>
                    <div class="mini-card">
                        <p class="mini-title">What it shows</p>
                        <p class="mini-copy">Fleet health, engine-level drift, downloadable reports, and manual GRU responses.</p>
                    </div>
                    <div class="mini-card">
                        <p class="mini-title">How to start</p>
                        <p class="mini-copy">Use the sample dataset or upload a CMAPSS-style CSV from the sidebar.</p>
                    </div>
                </div>
            </div>
            <div class="story-card">
                <div class="status-pill">Quick Start</div>
                <p class="story-title">Recommended app flow</p>
                <p class="story-copy">
                    1. Load sample or uploaded engine telemetry.
                    <br>2. Review fleet RUL trends and alert-zone distribution.
                    <br>3. Inspect a specific engine for sensor behavior.
                    <br>4. Export reports for review.
                    <br>5. Use the manual predictor to test GRU responses.
                </p>
            </div>
        </div>
        <div class="highlight-band">
            <div class="highlight-tile cyan">
                <p class="highlight-title">Fleet Awareness</p>
                <p class="highlight-copy">Surface engines that need attention first with colorful alert zoning and sortable prediction output.</p>
            </div>
            <div class="highlight-tile orange">
                <p class="highlight-title">Demo Friendly</p>
                <p class="highlight-copy">Switch to the sample dataset instantly, show architecture, and export a clean summary for reviewers.</p>
            </div>
            <div class="highlight-tile violet">
                <p class="highlight-title">Hands-On AI</p>
                <p class="highlight-copy">Let users enter GRU sensor values themselves and watch model responses update in the same app.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    arch_path = "project-arch.png"
    if os.path.exists(arch_path):
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<p class="subtle-heading">System Architecture</p>', unsafe_allow_html=True)
        st.image(arch_path, caption="PrognosAI pipeline and modeling architecture", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<p class="subtle-heading">Why this UI helps</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.info("Decision makers can download concise reports without digging through raw tables.")
    col2.info("Operators can inspect engine alerts, trends, and sensor changes from a single interface.")
    col3.info("Reviewers can test manual GRU inputs to understand how the model responds to telemetry patterns.")
    st.markdown("</div>", unsafe_allow_html=True)


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


def build_report_summary(prediction_df: pd.DataFrame) -> str:
    healthiest_engine = prediction_df.sort_values("Predicted RUL", ascending=False).iloc[0]
    riskiest_engine = prediction_df.sort_values("Predicted RUL", ascending=True).iloc[0]
    critical_count = int((prediction_df["Alert Zone"] == "Critical").sum())
    warning_count = int((prediction_df["Alert Zone"] == "Warning").sum())
    avg_rul = prediction_df["Predicted RUL"].mean()
    mae = prediction_df["Absolute Error"].dropna().mean()
    mae_text = f"{mae:.2f} cycles" if not np.isnan(mae) else "Unavailable"

    lines = [
        "PrognosAI Fleet RUL Report",
        "",
        f"Engines analysed: {len(prediction_df)}",
        f"Average predicted RUL: {avg_rul:.2f} cycles",
        f"Critical engines: {critical_count}",
        f"Warning engines: {warning_count}",
        f"Mean absolute error: {mae_text}",
        "",
        f"Highest projected RUL: Engine {int(healthiest_engine['Engine'])} at {healthiest_engine['Predicted RUL']:.2f} cycles",
        f"Lowest projected RUL: Engine {int(riskiest_engine['Engine'])} at {riskiest_engine['Predicted RUL']:.2f} cycles",
        "",
        "Top 5 engines needing attention:",
    ]

    focus_df = prediction_df.sort_values(["Alert Rank", "Predicted RUL"]).head(5)
    for _, row in focus_df.iterrows():
        lines.append(
            f"- Engine {int(row['Engine'])}: {row['Predicted RUL']:.2f} cycles remaining ({row['Alert Zone']})"
        )

    return "\n".join(lines)


def render_report_center(prediction_df: pd.DataFrame):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<p class="subtle-heading">Report Center</p>', unsafe_allow_html=True)
    report_text = build_report_summary(prediction_df)
    col1, col2 = st.columns((1.1, 1))

    with col1:
        st.markdown(
            """
            <div class="panel-shell">
                <div class="status-pill">Auto-generated Summary</div>
                <p class="panel-title">Fleet executive snapshot</p>
                <p class="panel-copy">This summary is designed for quick handoff to project reviewers, maintenance leads, or faculty evaluators.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.text_area(
            "Generated fleet summary",
            report_text,
            height=240,
            disabled=True,
            key="report_preview",
        )
        st.download_button(
            "Download fleet report (.txt)",
            data=report_text,
            file_name="prognosai_fleet_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with col2:
        st.markdown(
            """
            <div class="panel-shell">
                <div class="status-pill">Export Actions</div>
                <p class="panel-title">One-click report downloads</p>
                <p class="panel-copy">Export the current prediction state in plain text, CSV, or JSON depending on who needs the output.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        export_df = prediction_df.drop(columns=["Alert Rank"]).copy()
        st.download_button(
            "Download predictions (.csv)",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name="prognosai_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "Download predictions (.json)",
            data=export_df.to_json(orient="records", indent=2).encode("utf-8"),
            file_name="prognosai_predictions.json",
            mime="application/json",
            use_container_width=True,
        )
        st.caption("Use these buttons to hand predictions to maintenance, analytics, or management teams.")

    st.markdown("</div>", unsafe_allow_html=True)


def build_manual_sequence_from_inputs(input_values: dict[str, float]) -> pd.DataFrame:
    repeated_rows = pd.DataFrame([input_values] * WINDOW_SIZE)
    return repeated_rows[SENSOR_COLS]


def parse_manual_sequence(sequence_text: str) -> pd.DataFrame:
    parsed = normalize_columns(pd.read_csv(StringIO(sequence_text.strip())))
    missing_cols = [col for col in SENSOR_COLS if col not in parsed.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in pasted sequence: {', '.join(missing_cols)}")
    if len(parsed) != WINDOW_SIZE:
        raise ValueError(f"Sequence must contain exactly {WINDOW_SIZE} rows.")
    return parsed[SENSOR_COLS]


def render_manual_gru_workspace(
    lstm_model,
    gru_model,
    feature_scaler,
    target_scaler,
):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<p class="subtle-heading">Manual GRU Prediction Workspace</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="panel-shell">
            <div class="status-pill">Interactive Prediction Controls</div>
            <p class="panel-title">Ask the model for a direct response</p>
            <p class="panel-copy">Choose quick entry for a fast estimate using one sensor snapshot repeated across the window, or paste a full 30-cycle sequence for a more realistic GRU-driven result.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_quick, tab_sequence = st.tabs(["Quick Sensor Entry", "Paste 30-Row Sequence"])

    with tab_quick:
        with st.form("quick_manual_gru_form"):
            input_cols = st.columns(3)
            manual_inputs = {}
            defaults = {
                "op1": 0.0,
                "op2": 0.0,
                "op3": 100.0,
                "sensor2": 642.0,
                "sensor3": 1585.0,
                "sensor4": 1400.0,
                "sensor7": 554.0,
                "sensor8": 2388.0,
                "sensor9": 9050.0,
                "sensor11": 47.3,
                "sensor12": 522.0,
                "sensor13": 2388.0,
                "sensor14": 8132.0,
                "sensor15": 8.42,
                "sensor17": 392.0,
                "sensor20": 39.0,
                "sensor21": 23.4,
            }

            for idx, sensor in enumerate(SENSOR_COLS):
                with input_cols[idx % 3]:
                    manual_inputs[sensor] = st.number_input(
                        sensor,
                        value=float(defaults.get(sensor, 0.0)),
                        format="%.5f",
                        key=f"quick_{sensor}",
                    )

            submitted_quick = st.form_submit_button("Get GRU Response", use_container_width=True)

        if submitted_quick:
            manual_sequence = build_manual_sequence_from_inputs(manual_inputs)
            lstm_pred, gru_pred, ensemble_pred = predict_from_manual_sequence(
                manual_sequence,
                lstm_model,
                gru_model,
                feature_scaler,
                target_scaler,
            )
            result_cols = st.columns(3)
            result_cols[0].metric("GRU RUL", f"{gru_pred:.1f} cycles")
            result_cols[1].metric("LSTM RUL", f"{lstm_pred:.1f} cycles")
            result_cols[2].metric("Ensemble RUL", f"{ensemble_pred:.1f} cycles")
            st.success(f"Alert assessment: {classify_alert(ensemble_pred)}")
            response_df = pd.DataFrame(
                [
                    {"Model": "GRU", "Predicted RUL": round(gru_pred, 2)},
                    {"Model": "LSTM", "Predicted RUL": round(lstm_pred, 2)},
                    {"Model": "Ensemble", "Predicted RUL": round(ensemble_pred, 2)},
                ]
            )
            response_fig = px.bar(
                response_df,
                x="Model",
                y="Predicted RUL",
                color="Model",
                color_discrete_map={
                    "GRU": "#0F766E",
                    "LSTM": "#1D4ED8",
                    "Ensemble": "#C2410C",
                },
                text="Predicted RUL",
            )
            response_fig.update_layout(
                height=300,
                showlegend=False,
                xaxis_title="",
                yaxis_title="Predicted RUL (cycles)",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.92)",
            )
            st.plotly_chart(response_fig, use_container_width=True)

    with tab_sequence:
        st.markdown(
            """
            <div class="panel-shell">
                <p class="panel-title">Required sequence format</p>
                <p class="panel-copy">Paste CSV text containing exactly 30 rows and the feature columns shown below. This matches the model window size used for fleet predictions.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.code(
            ",".join(SENSOR_COLS),
            language="text",
        )
        sequence_text = st.text_area(
            "Paste CSV data with exactly 30 rows and these columns",
            height=220,
            key="manual_sequence_text",
        )
        if st.button("Run Full Sequence Prediction", use_container_width=True):
            try:
                manual_sequence = parse_manual_sequence(sequence_text)
                lstm_pred, gru_pred, ensemble_pred = predict_from_manual_sequence(
                    manual_sequence,
                    lstm_model,
                    gru_model,
                    feature_scaler,
                    target_scaler,
                )
                result_cols = st.columns(3)
                result_cols[0].metric("GRU RUL", f"{gru_pred:.1f} cycles")
                result_cols[1].metric("LSTM RUL", f"{lstm_pred:.1f} cycles")
                result_cols[2].metric("Ensemble RUL", f"{ensemble_pred:.1f} cycles")
                st.info(f"Predicted maintenance state: {classify_alert(ensemble_pred)}")
                response_df = pd.DataFrame(
                    [
                        {"Model": "GRU", "Predicted RUL": round(gru_pred, 2)},
                        {"Model": "LSTM", "Predicted RUL": round(lstm_pred, 2)},
                        {"Model": "Ensemble", "Predicted RUL": round(ensemble_pred, 2)},
                    ]
                )
                response_fig = px.line(
                    response_df,
                    x="Model",
                    y="Predicted RUL",
                    markers=True,
                )
                response_fig.update_traces(line=dict(color="#0F766E", width=3), marker=dict(size=11))
                response_fig.update_layout(
                    height=300,
                    xaxis_title="",
                    yaxis_title="Predicted RUL (cycles)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(255,255,255,0.92)",
                )
                st.plotly_chart(response_fig, use_container_width=True)
            except Exception as exc:
                st.error(f"Could not process the pasted sequence: {exc}")

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

    if "use_sample_data" not in st.session_state:
        st.session_state.use_sample_data = False

    st.sidebar.header("Data Input")
    theme_mode = st.sidebar.selectbox(
        "Theme",
        ["Sunrise", "Midnight"],
        index=0,
        help="Switch between a bright colorful dashboard and a darker presentation mode.",
    )
    render_header(theme_mode)
    render_action_strip()
    if st.sidebar.button("Use bundled sample dataset", use_container_width=True):
        st.session_state.use_sample_data = True
    if st.sidebar.button("Clear sample dataset", use_container_width=True):
        st.session_state.use_sample_data = False
    uploaded_test = st.sidebar.file_uploader(
        "Upload test sensor data", type="csv", help="Use a CMAPSS-style test CSV."
    )

    st.sidebar.header("Alert Zones")
    st.sidebar.caption(f"Critical: 0 to {CRITICAL_THRESHOLD} cycles")
    st.sidebar.caption(
        f"Warning: {CRITICAL_THRESHOLD + 1} to {WARNING_THRESHOLD} cycles"
    )
    st.sidebar.caption(f"Healthy: above {WARNING_THRESHOLD} cycles")
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **Control Center**

        Use the sample dataset for a fast demo, or upload your own CMAPSS-style CSV to generate fleet predictions, reports, and GRU responses.
        """
    )

    if not uploaded_test and not st.session_state.use_sample_data:
        st.info("Upload a test CSV or click the sample-data button in the sidebar to open the full dashboard.")
        return

    if not all([lstm_model, gru_model, feature_scaler, target_scaler]):
        st.error("Model assets are not available. Please check the `models/` directory.")
        return

    if uploaded_test is not None:
        test_df = normalize_columns(pd.read_csv(uploaded_test))
        st.sidebar.success("Using uploaded dataset")
    else:
        test_df = load_default_dataset()
        st.sidebar.success("Using bundled sample dataset")

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
    home_tab, overview_tab, engine_tab, report_tab, manual_tab = st.tabs(
        ["Home", "Fleet Dashboard", "Engine Inspector", "Reports", "Manual Predictor"]
    )

    with home_tab:
        render_home_tab()

    with overview_tab:
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

    with engine_tab:
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

    with report_tab:
        render_report_center(prediction_df)

    with manual_tab:
        render_manual_gru_workspace(
            lstm_model,
            gru_model,
            feature_scaler,
            target_scaler,
        )


if __name__ == "__main__":
    main()