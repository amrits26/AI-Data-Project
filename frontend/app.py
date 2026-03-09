"""
AI Data Scientist — Professional, Google-inspired Streamlit dashboard.
Layout: Upload CSV → Overview → Executive Summary → Modeling → Visualization.
Night mode default; high-contrast; sidebar nav; interactive filters and Plotly charts.
"""
import sys
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from backend.app.agents import (
    profile_dataset,
    run_statistical_insights,
    recommend_and_run_models,
    run_anomaly_detection,
    compute_cognitive_flags,
    generate_insights,
)

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Data Scientist",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Session state
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
if "result" not in st.session_state:
    st.session_state.result = None
if "last_upload" not in st.session_state:
    st.session_state.last_upload = None
if "selected_columns" not in st.session_state:
    st.session_state.selected_columns = None
if "max_preview_rows" not in st.session_state:
    st.session_state.max_preview_rows = 500
if "current_section" not in st.session_state:
    st.session_state.current_section = "Overview"

# -----------------------------------------------------------------------------
# Google-inspired night-mode CSS (high contrast, clean sections)
# -----------------------------------------------------------------------------
def inject_theme(dark: bool):
    bg = "#0e1117" if dark else "#f8f9fa"
    card_bg = "#1a1d24" if dark else "#ffffff"
    card_border = "#2d3238" if dark else "#e5e7eb"
    text = "#fafafa" if dark else "#1a1a1a"
    text_secondary = "#9ca3af" if dark else "#5f6368"
    accent = "#4285f4"
    accent_light = "#8ab4f8"
    font = "'Google Sans', 'Segoe UI', Roboto, sans-serif"
    st.markdown(
        f"""
    <style>
    .stApp {{ background: {bg}; }}
    .main .block-container {{ padding: 1.5rem 2rem; max-width: 1600px; }}
    * {{ font-family: {font}; box-sizing: border-box; }}
    
    /* Section headers */
    .section-header {{
        font-size: 1.35rem; font-weight: 500; color: {text};
        letter-spacing: -0.01em; margin: 1.25rem 0 0.75rem 0;
        padding-bottom: 0.5rem; border-bottom: 1px solid {card_border};
    }}
    .section-sub {{ font-size: 0.9rem; color: {text_secondary}; margin-bottom: 1rem; }}
    
    /* Metric cards */
    .metric-card {{
        background: {card_bg}; border-radius: 10px; padding: 1rem 1.25rem;
        border: 1px solid {card_border}; margin: 0 0.5rem 0.5rem 0;
    }}
    .metric-label {{ font-size: 0.8rem; color: {text_secondary}; text-transform: uppercase; letter-spacing: 0.03em; }}
    .metric-value {{ font-size: 1.5rem; font-weight: 600; color: {accent}; }}
    
    /* Insight / summary boxes */
    .insight-box {{
        background: {card_bg}; border-radius: 10px; padding: 1.25rem 1.5rem;
        border-left: 4px solid {accent}; margin: 0.75rem 0;
        border: 1px solid {card_border}; border-left: 4px solid {accent};
    }}
    .insight-box.warning {{ border-left-color: #f59e0b; }}
    .insight-box.success {{ border-left-color: #10b981; }}
    
    /* Tooltips */
    [data-tooltip] {{ border-bottom: 1px dotted {text_secondary}; cursor: help; }}
    
    #MainMenu, footer {{ visibility: hidden; }}
    [data-testid="stFileUploader"] {{
        background: {card_bg}; border-radius: 10px; padding: 1rem;
        border: 2px dashed {card_border};
    }}
    div[data-testid="stExpander"] {{
        background: {card_bg}; border-radius: 8px; border: 1px solid {card_border};
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )


def _auto_detect_target(df: pd.DataFrame) -> str | None:
    if df is None or df.empty or len(df.columns) == 0:
        return None
    candidates = ["target", "label", "class", "outcome", "y", "result", "dependent"]
    cols_lower = [c.lower().strip() for c in df.columns]
    for c in candidates:
        for i, col in enumerate(df.columns):
            if cols_lower[i] == c or c in cols_lower[i]:
                return col
    last = df.columns[-1]
    if pd.api.types.is_numeric_dtype(df[last]) or df[last].nunique() <= 50:
        return last
    return None


def load_data(uploaded_file) -> pd.DataFrame | None:
    if uploaded_file is None:
        return None
    name = (uploaded_file.name or "").lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file, nrows=100_000)
        if name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file, engine="openpyxl", nrows=100_000)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return None
    st.warning("Unsupported format. Use CSV or Excel (.xlsx, .xls).")
    return None


def run_pipeline(df: pd.DataFrame, target_column: str | None) -> dict | None:
    progress = st.progress(0, text="Starting pipeline…")
    try:
        progress.progress(0.15, text="Profiling dataset…")
        profile = profile_dataset(df, target_column=target_column)
        progress.progress(0.35, text="Statistical insights…")
        statistical = run_statistical_insights(df, target_column=target_column, profile=profile)
        progress.progress(0.55, text="Modeling…")
        modeling = recommend_and_run_models(df, target_column=target_column)
        progress.progress(0.70, text="Anomaly detection…")
        anomaly = run_anomaly_detection(df)
        progress.progress(0.85, text="Cognitive flags & executive summary…")
        flags = compute_cognitive_flags(profile=profile, statistical=statistical, modeling=modeling)
        executive = generate_insights(profile, statistical, modeling, anomaly, flags, use_llm=True)
        progress.progress(1.0, text="Done")
        return {
            "profile": profile,
            "statistical": statistical,
            "modeling": modeling,
            "anomaly": anomaly,
            "cognitive_flags": flags,
            "executive_summary": executive,
        }
    except Exception as e:
        progress.progress(1.0, text="Error")
        st.error(f"Pipeline failed: {str(e)}")
        return None


def _get_filtered_df(df: pd.DataFrame, profile: dict | None) -> pd.DataFrame:
    """Apply column and row limits from session state for overview/visualization."""
    cols = st.session_state.get("selected_columns") or list(df.columns)
    if not cols:
        cols = list(df.columns)
    max_rows = st.session_state.get("max_preview_rows") or 500
    return df[cols].head(max_rows)


# =============================================================================
# Section 1: Data Overview (metrics, filters, table, charts)
# =============================================================================
def render_overview(df: pd.DataFrame, profile: dict | None):
    st.markdown('<p class="section-header">📋 Data overview</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Key metrics, column/row filters, and summary charts. Selections update tables and charts below.</p>', unsafe_allow_html=True)

    # Metrics row
    r, c = df.shape
    missing_pct = (df.isna().sum().sum() / (r * c * 1.0) * 100) if (r and c) else 0
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [x for x in df.columns if x not in numeric_cols]
    n_numeric, n_categorical = len(numeric_cols), len(cat_cols)

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Rows", f"{r:,}", help="Total number of rows in the dataset")
    with m2:
        st.metric("Columns", c, help="Total number of columns")
    with m3:
        st.metric("Missing %", f"{missing_pct:.1f}%", help="Share of cells that are missing")
    with m4:
        st.metric("Numeric", n_numeric, help="Columns with numeric type")
    with m5:
        st.metric("Categorical", n_categorical, help="Non-numeric columns")
    if profile:
        st.metric("Data health score", f"{profile.get('data_health_score', 0)}/100", help="Overall data quality score (0–100)")

    # Filters (update when changed)
    st.markdown("**Filters**")
    valid_default = [c for c in (st.session_state.selected_columns or list(df.columns)) if c in df.columns]
    if not valid_default:
        valid_default = list(df.columns)
    col_filter = st.multiselect(
        "Select columns to display",
        options=list(df.columns),
        default=valid_default,
        help="Charts and table below use only selected columns.",
    )
    st.session_state.selected_columns = col_filter if col_filter else list(df.columns)
    max_rows = st.slider("Max rows in table and charts", 50, 2000, st.session_state.get("max_preview_rows", 500), 50, help="Limit rows for performance")
    st.session_state.max_preview_rows = max_rows

    filtered = _get_filtered_df(df, profile)
    display_cols = [x for x in st.session_state.selected_columns if x in filtered.columns]
    if not display_cols:
        display_cols = list(filtered.columns)

    # Interactive table
    st.markdown("**Preview data**")
    st.dataframe(filtered[display_cols] if display_cols else filtered, use_container_width=True, hide_index=True)

    # Summary charts (bar, pie, histograms)
    st.markdown("**Summary charts**")

    # Bar: missing % per column (for displayed columns)
    miss_per_col = filtered[display_cols].isna().mean() * 100
    if len(display_cols) > 0 and len(display_cols) <= 30:
        fig_miss = px.bar(
            x=miss_per_col.index.astype(str),
            y=miss_per_col.values,
            labels={"x": "Column", "y": "Missing %"},
            title="Missing values by column",
        )
        fig_miss.update_layout(xaxis_tickangle=-45, margin=dict(b=100))
        st.plotly_chart(fig_miss, use_container_width=True)

    # Pie: numeric vs categorical
    fig_pie = go.Figure(data=[go.Pie(
        labels=["Numeric", "Categorical"],
        values=[n_numeric, n_categorical],
        hole=0.5,
        marker_colors=["#4285f4", "#34a853"],
    )])
    fig_pie.update_layout(title="Column types", height=300)
    st.plotly_chart(fig_pie, use_container_width=True)

    # Histograms: one per numeric column (top 6) in filtered set
    num_in_display = [x for x in display_cols if x in numeric_cols]
    for i, col in enumerate(num_in_display[:6]):
        try:
            fig_hist = px.histogram(
                filtered[col].dropna(),
                nbins=min(50, max(10, int(filtered[col].nunique() / 2))),
                title=f"Distribution: {col}",
                labels={"value": col, "count": "Count"},
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        except Exception:
            pass


# =============================================================================
# Section 2: Executive Summary (polished insight boxes)
# =============================================================================
def render_executive_summary(data: dict):
    st.markdown('<p class="section-header">📄 Executive summary</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Human-readable insights, trends, and recommendations for decision-making.</p>', unsafe_allow_html=True)

    ex = data.get("executive_summary") or {}
    profile = data.get("profile") or {}
    statistical = data.get("statistical") or {}
    modeling = data.get("modeling") or {}
    anomaly = data.get("anomaly") or {}

    # Main summary box
    summary_text = ex.get("summary", "Run the analysis to see the executive summary.")
    st.markdown(f'<div class="insight-box">{summary_text}</div>', unsafe_allow_html=True)

    # Trends & patterns (from profile + stats)
    bullets = []
    bullets.append(f"**Dataset:** {profile.get('rows', 0):,} rows, {profile.get('columns', 0)} columns. Data health score: {profile.get('data_health_score', 0)}/100.")
    if statistical.get("summary"):
        bullets.append(f"**Statistics:** {statistical['summary']}")
    if anomaly.get("combined_summary"):
        bullets.append(f"**Anomalies:** {anomaly['combined_summary']}")
    if modeling and not modeling.get("message"):
        bullets.append(f"**Modeling:** {modeling.get('summary', '')}")
    for b in bullets:
        st.markdown(f'<div class="insight-box success"><p style="margin:0;">{b}</p></div>', unsafe_allow_html=True)

    # Business implications
    if ex.get("business_implications"):
        st.markdown("**Business implications**")
        for imp in ex["business_implications"]:
            st.markdown(f"- {imp}")

    # Risks
    if ex.get("risks"):
        st.markdown("**Risks**")
        for r in ex["risks"]:
            st.warning(r)

    # Next steps
    if ex.get("next_steps"):
        st.markdown("**Recommended next steps**")
        for n in ex["next_steps"]:
            st.markdown(f"- {n}")

    # Cognitive flags as expandable cards
    flags = data.get("cognitive_flags") or []
    if flags:
        st.markdown("**Insight cards**")
        for f in flags:
            sev = f.get("severity", "info")
            with st.expander(f"{f.get('title', '')} — {sev}", expanded=False):
                st.write(f.get("description", ""))
                if f.get("recommendation"):
                    st.info(f"💡 {f['recommendation']}")
                if f.get("math_detail"):
                    st.caption(f"Detail: {f['math_detail']}")


# =============================================================================
# Section 3: Modeling (performance, overfitting, feature importance, skip messages)
# =============================================================================
def render_modeling(modeling: dict):
    st.markdown('<p class="section-header">🤖 Modeling</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Model performance, cross-validation, overfitting risk, and rich diagnostics. Expand sections below for more detail.</p>', unsafe_allow_html=True)

    if not modeling:
        st.info("Run the full analysis with a target column to see modeling results.")
        return

    if modeling.get("message"):
        st.markdown(
            '<div class="insight-box warning">'
            '<strong>Modeling not run</strong><br/>' + modeling["message"] +
            '</div>',
            unsafe_allow_html=True,
        )
        st.caption("Common reasons: no target column selected, too few rows (min 10), no numeric features, or target has only one class.")
        return

    # ------------------------------------------------------------------
    # Performance & overfitting
    # ------------------------------------------------------------------
    with st.expander("Performance & overfitting", expanded=True):
        st.markdown(modeling.get("summary", ""))
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Problem type", modeling.get("problem_type", ""), help="Classification or regression task")
        with c2:
            st.metric("CV mean", f"{modeling.get('cross_val_mean', 0):.4f}", help="Cross-validation mean (accuracy or R²)")
        with c3:
            st.metric("CV std", f"{modeling.get('cross_val_std', 0):.4f}", help="Stability of performance across folds")
        with c4:
            st.metric("Model", modeling.get("model_used", ""), help="Primary model used")

        overfit = modeling.get("overfitting_analysis") or {}
        train_s = overfit.get("train_score")
        val_s = overfit.get("validation_score")
        if train_s is not None and val_s is not None:
            fig_scores = px.bar(
                x=["Train", "Validation"],
                y=[train_s, val_s],
                range_y=[0, 1],
                labels={"x": "", "y": "Score"},
                title="Overfitting indicator: train vs validation",
            )
            st.plotly_chart(fig_scores, use_container_width=True)
        risk = overfit.get("overfitting_risk", "")
        if risk in ("high", "moderate"):
            st.warning(
                f"**Overfitting risk: {risk}.** Train: {train_s}, Validation: {val_s}. "
                "Consider regularization, simpler models, or more data."
            )
        else:
            st.success(f"Overfitting risk: **{risk}**.")

        cv_scores = modeling.get("cv_scores")
        if cv_scores:
            fig_cv = px.bar(
                x=list(range(1, len(cv_scores) + 1)),
                y=cv_scores,
                labels={"x": "Fold", "y": "Score"},
                title="Cross-validation scores by fold",
            )
            st.plotly_chart(fig_cv, use_container_width=True)

    # ------------------------------------------------------------------
    # Feature importance & SHAP-style explanation
    # ------------------------------------------------------------------
    with st.expander("Feature importance & SHAP-style explanation", expanded=False):
        imp = modeling.get("feature_importance") or {}
        if imp:
            imp_sorted = dict(sorted(imp.items(), key=lambda x: -x[1])[:15])
            fig = px.bar(
                x=list(imp_sorted.keys()),
                y=list(imp_sorted.values()),
                labels={"x": "Feature", "y": "Importance"},
                title="Feature importance (RandomForest)",
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        shap_imp = modeling.get("shap_importance") or {}
        if shap_imp:
            shap_sorted = dict(sorted(shap_imp.items(), key=lambda x: -x[1])[:15])
            fig_shap = px.bar(
                x=list(shap_sorted.keys()),
                y=list(shap_sorted.values()),
                labels={"x": "Feature", "y": "Mean |SHAP|"},
                title="SHAP summary (global feature impact)",
            )
            fig_shap.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_shap, use_container_width=True)
            st.caption("Higher mean |SHAP| indicates stronger contribution of a feature to the model's predictions.")

    # ------------------------------------------------------------------
    # Predictions, residuals, and target vs predictions
    # ------------------------------------------------------------------
    with st.expander("Predictions, residuals, and target vs predictions", expanded=False):
        pv = modeling.get("pred_vs_actual") or {}
        y_true, y_pred = pv.get("y_true"), pv.get("y_pred")
        if y_true and y_pred and len(y_true) == len(y_pred):
            df_pv = pd.DataFrame({"actual": y_true, "predicted": y_pred})
            if modeling.get("problem_type") == "regression":
                st.markdown("**Predicted vs actual (regression)**")
                fig = px.scatter(
                    df_pv,
                    x="actual",
                    y="predicted",
                    opacity=0.7,
                    title="Predicted vs actual",
                )
                try:
                    lo, hi = float(df_pv["actual"].min()), float(df_pv["actual"].max())
                    line = np.linspace(lo, hi, 100)
                    fig.add_trace(
                        go.Scatter(
                            x=line,
                            y=line,
                            name="Ideal",
                            line=dict(color="red", dash="dash"),
                        )
                    )
                except Exception:
                    pass
                st.plotly_chart(fig, use_container_width=True)

                # Residual histogram
                resid = modeling.get("residuals_sample") or []
                if resid:
                    fig_resid = px.histogram(
                        resid,
                        nbins=40,
                        labels={"value": "Residual"},
                        title="Residuals (prediction error)",
                    )
                    st.plotly_chart(fig_resid, use_container_width=True)

                # Target vs prediction distribution
                fig_dist = go.Figure()
                fig_dist.add_trace(
                    go.Histogram(x=df_pv["actual"], name="Actual", opacity=0.6)
                )
                fig_dist.add_trace(
                    go.Histogram(x=df_pv["predicted"], name="Predicted", opacity=0.6)
                )
                fig_dist.update_layout(
                    barmode="overlay",
                    title="Target vs prediction distribution",
                    xaxis_title="Value",
                )
                fig_dist.update_traces(marker_line_width=0)
                st.plotly_chart(fig_dist, use_container_width=True)
            else:
                # Classification: confusion matrix + class distribution
                try:
                    st.markdown("**Confusion matrix (actual vs predicted)**")
                    ct = pd.crosstab(df_pv["actual"], df_pv["predicted"])
                    fig = px.imshow(
                        ct,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale="Blues",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    st.dataframe(
                        df_pv.head(50), use_container_width=True, hide_index=True
                    )

                # Class distribution: actual vs predicted
                try:
                    actual_counts = pd.Series(y_true).value_counts().sort_index()
                    pred_counts = pd.Series(y_pred).value_counts().sort_index()
                    df_cls = pd.DataFrame(
                        {"Actual": actual_counts, "Predicted": pred_counts}
                    ).fillna(0)
                    fig_cls = px.bar(
                        df_cls,
                        barmode="group",
                        title="Target vs prediction distribution",
                    )
                except Exception:
                    df_cls = pd.DataFrame(
                        {
                            "Actual": pd.Series(y_true).value_counts(),
                            "Predicted": pd.Series(y_pred).value_counts(),
                        }
                    )
                    fig_cls = px.bar(
                        df_cls,
                        barmode="group",
                        title="Target vs prediction distribution",
                    )
                st.plotly_chart(fig_cls, use_container_width=True)

    # ------------------------------------------------------------------
    # ROC and precision–recall (classification only)
    # ------------------------------------------------------------------
    if modeling.get("problem_type") == "classification":
        roc = modeling.get("roc_curve") or {}
        pr = modeling.get("pr_curve") or {}
        if roc or pr:
            with st.expander("ROC and precision–recall curves", expanded=False):
                if roc.get("fpr") and roc.get("tpr"):
                    fig_roc = px.line(
                        x=roc["fpr"],
                        y=roc["tpr"],
                        labels={"x": "False positive rate", "y": "True positive rate"},
                        title="ROC curve",
                    )
                    fig_roc.add_shape(
                        type="line",
                        x0=0,
                        y0=0,
                        x1=1,
                        y1=1,
                        line=dict(color="gray", dash="dash"),
                    )
                    st.plotly_chart(fig_roc, use_container_width=True)
                if pr.get("precision") and pr.get("recall"):
                    fig_pr = px.line(
                        x=pr["recall"],
                        y=pr["precision"],
                        labels={"x": "Recall", "y": "Precision"},
                        title="Precision–recall curve",
                    )
                    st.plotly_chart(fig_pr, use_container_width=True)
                st.caption(
                    "ROC and PR curves help evaluate classification performance under different thresholds, especially with imbalanced data."
                )


# =============================================================================
# Section 4: Visualization (correlation, PCA, anomaly)
# =============================================================================
def render_visualization(data: dict, df: pd.DataFrame):
    st.markdown('<p class="section-header">📈 Visualization</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Correlation matrix, PCA variance, and anomaly overview.</p>', unsafe_allow_html=True)

    statistical = data.get("statistical") or {}
    anomaly = data.get("anomaly") or {}

    # Correlation heatmap
    corr = statistical.get("correlation") or {}
    if corr.get("matrix"):
        num_df = pd.DataFrame(corr["matrix"])
        fig = px.imshow(num_df, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", title="Correlation matrix")
        fig.update_layout(margin=dict(l=80, r=40))
        st.plotly_chart(fig, use_container_width=True)

    # PCA
    pca = statistical.get("pca") or {}
    if pca.get("explained_variance_ratio"):
        r = pca["explained_variance_ratio"]
        c = pca.get("cumulative", [])
        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(range(1, len(r) + 1)), y=r, name="Variance"))
        if c:
            fig.add_trace(go.Scatter(x=list(range(1, len(c) + 1)), y=c, name="Cumulative", line=dict(color="orange")))
        fig.update_layout(title="PCA variance explained", xaxis_title="Component")
        st.plotly_chart(fig, use_container_width=True)

    # Anomaly summary bar
    if anomaly:
        methods = []
        pcts = []
        for k in ["isolation_forest", "zscore", "dbscan"]:
            if k in anomaly and isinstance(anomaly[k], dict):
                methods.append(anomaly[k].get("method", k))
                pcts.append(anomaly[k].get("anomaly_pct", 0))
        if methods and pcts:
            fig_anom = px.bar(x=methods, y=pcts, labels={"x": "Method", "y": "Anomaly %"}, title="Anomaly detection by method")
            st.plotly_chart(fig_anom, use_container_width=True)
        st.markdown(anomaly.get("combined_summary", ""))

    # Distribution recommendations
    for rec in (statistical.get("distribution_recommendations") or [])[:5]:
        st.warning(f"**{rec.get('feature')}**: {rec.get('recommendation')}")


# =============================================================================
# Main
# =============================================================================
def main():
    dark = st.session_state.dark_mode
    inject_theme(dark)

    st.markdown('<p class="section-header" style="margin-top:0;">📊 AI Data Scientist</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Upload a CSV or Excel file, then run the full analysis. Use the sidebar to jump to Overview, Executive Summary, Modeling, or Visualization.</p>', unsafe_allow_html=True)

    # Sidebar: upload, target, run, theme, navigation
    with st.sidebar:
        st.header("Input")
        dark_mode = st.toggle("Dark mode", value=dark, help="Night mode on by default for readability")
        if dark_mode != dark:
            st.session_state.dark_mode = dark_mode
            st.rerun()
        uploaded = st.file_uploader(
            "Upload CSV or Excel",
            type=["csv", "xlsx", "xls"],
            help="Drag and drop or click. Max 50MB recommended.",
        )
        df = load_data(uploaded) if uploaded else None

        target_options = [""] + (df.columns.tolist() if df is not None else [])
        target_suggestion = _auto_detect_target(df) if df is not None else None
        target_index = target_options.index(target_suggestion) if (target_suggestion and target_suggestion in target_options) else 0
        target_column = st.selectbox("Target column (optional)", options=target_options, index=target_index, help="For modeling. Auto-detected when possible.")
        target_column = (target_column or None) if isinstance(target_column, str) else None

        run = st.button("Run full analysis", type="primary")

        st.divider()
        st.header("Navigate")
        section = st.radio(
            "Go to section",
            ["Overview", "Executive Summary", "Modeling", "Visualization"],
            index=["Overview", "Executive Summary", "Modeling", "Visualization"].index(st.session_state.get("current_section", "Overview")),
            help="Jump to a section after running analysis.",
        )
        st.session_state.current_section = section

    if df is None and not run:
        st.info("👆 Upload a CSV or Excel file in the sidebar, then click **Run full analysis**.")
        return
    if df is None:
        st.warning("Please upload a file first.")
        return
    if df.empty:
        st.error("The file is empty.")
        return

    if len(df) < 100:
        st.warning("⚠️ Small dataset (< 100 rows). Results may have high variance.")

    if target_column and target_column not in df.columns:
        st.warning(f"Target '{target_column}' not found. Proceeding without target.")
        target_column = None

    if run:
        st.session_state.last_upload = uploaded.name if uploaded else None
        st.session_state.result = run_pipeline(df, target_column)

    data = st.session_state.result
    profile = data.get("profile") if data else None

    # Show section based on sidebar nav
    if section == "Overview":
        render_overview(df, profile)
    elif section == "Executive Summary":
        if data is None:
            st.info("Run the full analysis to see the executive summary.")
        else:
            render_executive_summary(data)
    elif section == "Modeling":
        render_modeling(data.get("modeling") if data else None)
    elif section == "Visualization":
        if data is None:
            st.info("Run the full analysis to see visualizations.")
        else:
            render_visualization(data, df)

    # Export (show when we have results)
    if data:
        st.divider()
        buf = json.dumps({
            "rows": (data.get("profile") or {}).get("rows"),
            "columns": (data.get("profile") or {}).get("columns"),
            "data_health_score": (data.get("profile") or {}).get("data_health_score"),
            "executive_summary": (data.get("executive_summary") or {}).get("summary"),
        }, indent=2).encode("utf-8")
        st.download_button("Download summary (JSON)", data=buf, file_name="analysis_summary.json", mime="application/json")
        txt = (data.get("executive_summary") or {}).get("summary", "") or "No summary."
        st.download_button("Download summary (TXT)", data=txt, file_name="summary.txt", mime="text/plain")


if __name__ == "__main__":
    main()
