import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="LLWS Escalation Monitoring Prototype",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT BASE DIRECTORY
# ============================================================
# File location for public deployment:
# C:\LLWS_DASHBOARD_PUBLIC\app.py
#
# Public folder contains:
# - app.py
# - df_eval_demo.parquet
# - feature_columns_final.pkl
# - alert_config_final.pkl
# - model_metadata_final.pkl

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: "Inter", "Segoe UI", sans-serif;
    }

    .stApp {
        background: #f5f7fb;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    section[data-testid="stSidebar"] {
        background-color: #0f172a;
    }

    section[data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    .hero-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #334155 100%);
        border-radius: 22px;
        padding: 28px 32px;
        color: white;
        box-shadow: 0 10px 26px rgba(15, 23, 42, 0.22);
        margin-bottom: 18px;
    }

    .hero-title {
        font-size: 34px;
        font-weight: 850;
        margin-bottom: 6px;
        letter-spacing: -0.5px;
    }

    .hero-subtitle {
        font-size: 15px;
        color: #cbd5e1;
        line-height: 1.5;
        max-width: 980px;
    }

    .hero-badge {
        display: inline-block;
        background: rgba(239, 68, 68, 0.16);
        color: #fecaca;
        border: 1px solid rgba(248, 113, 113, 0.5);
        padding: 6px 12px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 700;
        margin-top: 12px;
    }

    .kpi-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.07);
        padding: 18px 18px;
        min-height: 132px;
    }

    .kpi-label {
        font-size: 13px;
        color: #64748b;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .kpi-value {
        font-size: 32px;
        font-weight: 850;
        color: #0f172a;
        line-height: 1.1;
    }

    .kpi-caption {
        font-size: 12px;
        color: #94a3b8;
        margin-top: 8px;
        line-height: 1.35;
    }

    .alert-active {
        background: #fff1f2;
        border: 1.5px solid #ef4444;
    }

    .alert-off {
        background: #f8fafc;
        border: 1px solid #cbd5e1;
    }

    .status-pill {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 800;
    }

    .pill-high {
        background: #fee2e2;
        color: #dc2626;
    }

    .pill-moderate {
        background: #ffedd5;
        color: #ea580c;
    }

    .pill-low {
        background: #dcfce7;
        color: #16a34a;
    }

    .pill-active {
        background: #fee2e2;
        color: #dc2626;
    }

    .pill-off {
        background: #e2e8f0;
        color: #334155;
    }

    .info-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 9px 0px;
        border-bottom: 1px solid #f1f5f9;
        gap: 12px;
    }

    .info-label {
        color: #64748b;
        font-size: 13px;
        font-weight: 650;
    }

    .info-value {
        color: #0f172a;
        font-size: 14px;
        font-weight: 800;
        text-align: right;
    }

    .metric-explain {
        background: #f8fafc;
        border: 1px dashed #cbd5e1;
        border-radius: 14px;
        padding: 13px 15px;
        color: #475569;
        font-size: 13px;
        line-height: 1.45;
    }

    .footer-note {
        text-align: center;
        color: #64748b;
        font-size: 13px;
        margin-top: 18px;
        font-style: italic;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# REQUIRED FILE CHECK
# ============================================================
required_files = {
    "df_eval_demo": BASE_DIR / "df_eval_demo.parquet",
    "feature_columns": BASE_DIR / "feature_columns_final.pkl",
    "alert_config": BASE_DIR / "alert_config_final.pkl",
    "model_metadata": BASE_DIR / "model_metadata_final.pkl",
}

missing_files = [name for name, path in required_files.items() if not path.exists()]

if missing_files:
    st.error("Beberapa file publik tidak ditemukan.")
    st.write("BASE_DIR yang terbaca:")
    st.code(str(BASE_DIR))
    st.write("File yang belum ditemukan:")
    st.write(missing_files)
    st.stop()


# ============================================================
# LOAD DATA & ARTIFACTS
# ============================================================
@st.cache_data
def load_data(base_dir: Path):
    df_eval = pd.read_parquet(base_dir / "df_eval_demo.parquet")
    return df_eval


@st.cache_resource
def load_artifacts(base_dir: Path):
    feature_cols = joblib.load(base_dir / "feature_columns_final.pkl")
    alert_config = joblib.load(base_dir / "alert_config_final.pkl")
    model_metadata = joblib.load(base_dir / "model_metadata_final.pkl")
    return feature_cols, alert_config, model_metadata


df_eval = load_data(BASE_DIR)
feature_cols, alert_config, model_metadata = load_artifacts(BASE_DIR)


# ============================================================
# TIME COLUMN HANDLING
# ============================================================
def prepare_time_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    possible_time_cols = ["date and time", "time", "datetime", "timestamp"]

    time_col = None
    for col in possible_time_cols:
        if col in df.columns:
            time_col = col
            break

    if time_col is not None:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col)
        df["monitor_time"] = df[time_col]
    else:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.sort_index()
            df["monitor_time"] = df.index
        else:
            df["monitor_time"] = pd.RangeIndex(start=0, stop=len(df), step=1)

    return df


df_eval = prepare_time_column(df_eval)


# ============================================================
# COLUMN SAFETY CHECK
# ============================================================
required_cols = ["prob", "prob_smooth", "actual", "alert_hold"]
missing_cols = [col for col in required_cols if col not in df_eval.columns]

if missing_cols:
    st.error(f"Kolom berikut tidak ditemukan di df_eval: {missing_cols}")
    st.write("Kolom tersedia:")
    st.write(df_eval.columns.tolist())
    st.stop()


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def risk_level(prob: float) -> str:
    if prob < 0.3:
        return "Low"
    elif prob < 0.7:
        return "Moderate"
    else:
        return "High"


def risk_color(level: str) -> str:
    if level == "High":
        return "#dc2626"
    elif level == "Moderate":
        return "#ea580c"
    else:
        return "#16a34a"


def risk_pill_class(level: str) -> str:
    if level == "High":
        return "pill-high"
    elif level == "Moderate":
        return "pill-moderate"
    else:
        return "pill-low"


def alert_status(value) -> str:
    return "ACTIVE" if int(value) == 1 else "OFF"


def alert_pill_class(value) -> str:
    return "pill-active" if int(value) == 1 else "pill-off"


def format_time_for_display(value):
    try:
        return pd.to_datetime(value).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(value)


def build_alert_segments(df: pd.DataFrame):
    """Return continuous alert_hold == 1 segments for shaded alert region."""
    if df.empty or "alert_hold" not in df.columns:
        return []

    temp = df[["monitor_time", "alert_hold"]].copy()
    temp["alert_hold"] = temp["alert_hold"].astype(int)

    segments = []
    in_segment = False
    start_time = None
    previous_time = None

    for _, row in temp.iterrows():
        current_time = row["monitor_time"]
        active = row["alert_hold"] == 1

        if active and not in_segment:
            start_time = current_time
            in_segment = True

        if not active and in_segment:
            segments.append((start_time, previous_time))
            in_segment = False

        previous_time = current_time

    if in_segment:
        segments.append((start_time, previous_time))

    return segments


# ============================================================
# SIDEBAR CONTROL
# ============================================================
st.sidebar.title("LLWS Control Panel")
st.sidebar.caption("Demo configuration and checkpoint status")

st.sidebar.success("Public demo data loaded")

threshold = alert_config.get("threshold", 0.7)
smoothing_window = alert_config.get("smoothing_window", 5)
hold_minutes = alert_config.get("hold_minutes", 5)

with st.sidebar.expander("Project path", expanded=False):
    st.code(str(BASE_DIR))

with st.sidebar.expander("Checkpoint summary", expanded=True):
    st.write(
        {
            "df_eval_demo": df_eval.shape,
            "feature_count": len(feature_cols),
        }
    )

with st.sidebar.expander("Alert configuration", expanded=True):
    st.write(
        {
            "threshold": threshold,
            "smoothing_window": smoothing_window,
            "persistence_rule": "2 of 3",
            "hold_minutes": hold_minutes,
        }
    )

st.sidebar.markdown("### Demo window")

if pd.api.types.is_datetime64_any_dtype(df_eval["monitor_time"]):
    min_time = df_eval["monitor_time"].min()
    max_time = df_eval["monitor_time"].max()

    default_start = pd.Timestamp("2025-10-19 02:20:00")
    default_end = pd.Timestamp("2025-10-19 03:10:00")

    if default_start < min_time or default_end > max_time:
        default_start = min_time
        default_end = min_time + pd.Timedelta(minutes=60)

    start_time = st.sidebar.text_input(
        "Start time",
        value=str(default_start)
    )

    end_time = st.sidebar.text_input(
        "End time",
        value=str(default_end)
    )

    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)

    demo_df = df_eval[
        (df_eval["monitor_time"] >= start_time) &
        (df_eval["monitor_time"] <= end_time)
    ].copy()

else:
    start_idx = st.sidebar.number_input("Start index", 0, len(df_eval) - 1, 0)
    window_size = st.sidebar.slider("Window size", 100, 3000, 500)
    demo_df = df_eval.iloc[start_idx:start_idx + window_size].copy()


if demo_df.empty:
    st.warning("Demo window kosong. Ubah start time dan end time.")
    st.stop()


selected_idx = st.sidebar.slider(
    "Current monitoring step",
    min_value=0,
    max_value=len(demo_df) - 1,
    value=min(25, len(demo_df) - 1),
    step=1
)

current_row = demo_df.iloc[selected_idx]

current_prob = float(current_row["prob"])
current_prob_smooth = float(current_row["prob_smooth"])
current_risk = risk_level(current_prob_smooth)
current_alert = alert_status(current_row["alert_hold"])
current_actual = "Escalation" if int(current_row["actual"]) == 1 else "No Escalation"
current_time = current_row["monitor_time"]


# ============================================================
# HEADER
# ============================================================
st.markdown(
    f"""
    <div class="hero-card">
        <div class="hero-title">LLWS Escalation Monitoring Prototype</div>
        <div class="hero-subtitle">
            A decision-support dashboard for monitoring probabilistic Low-Level Wind Shear escalation risk
            using multi-runway AWOS surface observations.
        </div>
        <div class="hero-badge">● Live demo mode &nbsp; | &nbsp; Current time: {format_time_for_display(current_time)}</div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# KPI CARDS
# ============================================================
k1, k2, k3, k4, k5, k6 = st.columns(6)

with k1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Raw probability</div>
            <div class="kpi-value" style="color:#2563eb;">{current_prob:.2f}</div>
            <div class="kpi-caption">Direct model output before smoothing.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Smoothed probability</div>
            <div class="kpi-value" style="color:#ea580c;">{current_prob_smooth:.2f}</div>
            <div class="kpi-caption">{smoothing_window}-minute rolling average for stable monitoring.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Risk level</div>
            <div class="kpi-value" style="color:{risk_color(current_risk)};">{current_risk}</div>
            <div class="kpi-caption">Low, Moderate, or High escalation risk.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k4:
    st.markdown(
        f"""
        <div class="kpi-card {'alert-active' if current_alert == 'ACTIVE' else 'alert-off'}">
            <div class="kpi-label">Operational alert</div>
            <div class="kpi-value" style="color:{'#dc2626' if current_alert == 'ACTIVE' else '#334155'};">{current_alert}</div>
            <div class="kpi-caption">After smoothing, persistence, and hold logic.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k5:
    st.markdown(
        """
        <div class="kpi-card">
            <div class="kpi-label">Forecast horizon</div>
            <div class="kpi-value" style="color:#7c3aed;">+3 min</div>
            <div class="kpi-caption">Escalation target horizon.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k6:
    st.markdown(
        """
        <div class="kpi-card">
            <div class="kpi-label">Data source</div>
            <div class="kpi-value" style="font-size:25px; color:#15803d;">RWY 15/33</div>
            <div class="kpi-caption">Multi-runway AWOS surface observations.</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MAIN MONITORING AREA
# ============================================================
left_col, right_col = st.columns([3.2, 1.15])

with left_col:
    with st.container(border=True):
        st.markdown("### Escalation Probability Timeline")
        st.caption(
            "Raw probability, smoothed probability, alert threshold, active alert regions, and actual escalation events."
        )

        fig = go.Figure()

        alert_segments = build_alert_segments(demo_df)
        for start, end in alert_segments:
            fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor="rgba(239, 68, 68, 0.12)",
                line_width=0,
                layer="below"
            )

        fig.add_trace(
            go.Scatter(
                x=demo_df["monitor_time"],
                y=demo_df["prob"],
                mode="lines",
                name="Raw probability",
                line=dict(width=1.5, color="#2563eb"),
                hovertemplate="Time=%{x}<br>Raw prob=%{y:.3f}<extra></extra>"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=demo_df["monitor_time"],
                y=demo_df["prob_smooth"],
                mode="lines",
                name="Smoothed probability",
                line=dict(width=4, color="#ea580c"),
                hovertemplate="Time=%{x}<br>Smoothed prob=%{y:.3f}<extra></extra>"
            )
        )

        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="#dc2626",
            annotation_text=f"Threshold {threshold}",
            annotation_position="top left"
        )

        actual_df = demo_df[demo_df["actual"] == 1]
        fig.add_trace(
            go.Scatter(
                x=actual_df["monitor_time"],
                y=actual_df["prob_smooth"],
                mode="markers",
                name="Actual escalation",
                marker=dict(size=9, color="#111827", symbol="circle"),
                hovertemplate="Time=%{x}<br>Actual escalation<extra></extra>"
            )
        )

        fig.add_vline(
            x=current_time,
            line_width=2,
            line_dash="dot",
            line_color="#111827",
            annotation_text="Current",
            annotation_position="top"
        )

        fig.update_layout(
            height=520,
            margin=dict(l=10, r=10, t=10, b=10),
            template="plotly_white",
            xaxis_title="Time",
            yaxis_title="Probability",
            yaxis=dict(range=[0, 1.05]),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.03,
                xanchor="left",
                x=0
            ),
            hovermode="x unified"
        )

        st.plotly_chart(fig, use_container_width=True)


with right_col:
    with st.container(border=True):
        st.markdown("### Current Operational Status")

        st.markdown(
            f"""
            <div style="margin-bottom:14px;">
                <div class="info-label">Risk level</div>
                <span class="status-pill {risk_pill_class(current_risk)}">{current_risk.upper()}</span>
            </div>

            <div style="margin-bottom:14px;">
                <div class="info-label">Alert status</div>
                <span class="status-pill {alert_pill_class(current_row['alert_hold'])}">{current_alert}</span>
            </div>

            <div class="info-row">
                <div class="info-label">Actual condition</div>
                <div class="info-value">{current_actual}</div>
            </div>

            <div class="info-row">
                <div class="info-label">Raw probability</div>
                <div class="info-value">{current_prob:.3f}</div>
            </div>

            <div class="info-row">
                <div class="info-label">Smoothed probability</div>
                <div class="info-value">{current_prob_smooth:.3f}</div>
            </div>

            <div class="info-row">
                <div class="info-label">Threshold</div>
                <div class="info-value">{threshold}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with st.container(border=True):
        st.markdown("### Alert Logic")

        st.markdown(
            f"""
            <div class="info-row">
                <div class="info-label">Smoothing window</div>
                <div class="info-value">{smoothing_window} min</div>
            </div>
            <div class="info-row">
                <div class="info-label">Persistence rule</div>
                <div class="info-value">2 of 3</div>
            </div>
            <div class="info-row">
                <div class="info-label">Hold logic</div>
                <div class="info-value">{hold_minutes} min</div>
            </div>
            <br>
            <div class="metric-explain">
                The alert is designed to reduce noisy one-minute probability spikes by combining smoothing,
                persistence confirmation, and temporary hold after activation.
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# OPERATIONAL SNAPSHOT
# ============================================================
st.markdown("## Operational Snapshot")

snapshot_left, snapshot_right = st.columns([1.2, 1.8])

with snapshot_left:
    with st.container(border=True):
        st.markdown("### Runway Surface Condition Snapshot")
        st.caption("Current surface-condition context used for operational interpretation.")

        runway_15 = {
            "Wind speed": "18 kt",
            "Wind direction": "150°",
            "Temperature": "17.3 °C",
            "Dew point": "12.4 °C",
            "QFE": "1009.2 hPa",
            "Relative humidity": "72%",
        }

        runway_33 = {
            "Wind speed": "22 kt",
            "Wind direction": "330°",
            "Temperature": "17.1 °C",
            "Dew point": "12.1 °C",
            "QFE": "1009.1 hPa",
            "Relative humidity": "71%",
        }

        col15, col33 = st.columns(2)

        with col15:
            st.markdown("#### Runway 15")
            for k, v in runway_15.items():
                st.markdown(
                    f"""
                    <div class="info-row">
                        <div class="info-label">{k}</div>
                        <div class="info-value">{v}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with col33:
            st.markdown("#### Runway 33")
            for k, v in runway_33.items():
                st.markdown(
                    f"""
                    <div class="info-row">
                        <div class="info-label">{k}</div>
                        <div class="info-value">{v}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


with snapshot_right:
    with st.container(border=True):
        st.markdown("### Model Output Snapshot")
        st.caption("Recent monitoring records around the selected current step.")

        snapshot_cols = ["monitor_time", "prob", "prob_smooth", "actual", "alert_hold"]

        snapshot = demo_df.iloc[
            max(0, selected_idx - 6): selected_idx + 1
        ][snapshot_cols].copy()

        snapshot["Risk Level"] = snapshot["prob_smooth"].apply(risk_level)
        snapshot["Alert Status"] = snapshot["alert_hold"].apply(
            lambda x: "ACTIVE" if int(x) == 1 else "OFF"
        )
        snapshot["Actual Condition"] = snapshot["actual"].apply(
            lambda x: "Escalation" if int(x) == 1 else "No Escalation"
        )

        snapshot = snapshot.rename(
            columns={
                "monitor_time": "Time",
                "prob": "Raw Prob",
                "prob_smooth": "Smoothed Prob"
            }
        )

        snapshot = snapshot[
            ["Time", "Raw Prob", "Smoothed Prob", "Risk Level", "Alert Status", "Actual Condition"]
        ]

        snapshot["Time"] = snapshot["Time"].apply(format_time_for_display)
        snapshot["Raw Prob"] = snapshot["Raw Prob"].round(3)
        snapshot["Smoothed Prob"] = snapshot["Smoothed Prob"].round(3)

        st.dataframe(snapshot, use_container_width=True, hide_index=True)


# ============================================================
# PERFORMANCE SUMMARY
# ============================================================
st.markdown("## Final Checkpoint Performance Summary")

perf_left, perf_right = st.columns([1.2, 1.8])

with perf_left:
    with st.container(border=True):
        st.markdown("### Checkpoint Integrity")

        st.metric("Evaluation rows", f"{len(df_eval):,}")
        st.metric("Demo window rows", f"{len(demo_df):,}")
        st.metric("Feature count", f"{len(feature_cols)}")

        st.markdown(
            """
            <div class="metric-explain">
            This panel confirms that the dashboard is using the final saved checkpoint files.
            It is intended for portfolio demonstration and final project verification.
            </div>
            """,
            unsafe_allow_html=True
        )


with perf_right:
    with st.container(border=True):
        st.markdown("### Operational Alert Confusion Matrix")
        st.caption("Evaluation between actual escalation condition and final alert-hold output.")

        cm = confusion_matrix(df_eval["actual"], df_eval["alert_hold"])

        cm_df = pd.DataFrame(
            cm,
            index=["Actual Normal", "Actual Escalation"],
            columns=["Alert OFF", "Alert ACTIVE"]
        )

        st.dataframe(cm_df, use_container_width=True)

        tn, fp, fn, tp = cm.ravel()
        total = tn + fp + fn + tp

        acc = (tn + tp) / total if total > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric("Accuracy", f"{acc:.2%}")

        with m2:
            st.metric("Escalation recall", f"{recall:.2%}")

        with m3:
            st.metric("Alert precision", f"{precision:.2%}")

        st.markdown(
            """
            <div class="metric-explain">
            The prototype prioritizes stable situational monitoring rather than exact-minute certified warning.
            False alarms and missed short events should be interpreted as part of the sensitivity–stability trade-off.
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div class="footer-note">
        Prototype for monitoring probabilistic LLWS escalation — not an operational certified warning system.
    </div>
    """,
    unsafe_allow_html=True
)