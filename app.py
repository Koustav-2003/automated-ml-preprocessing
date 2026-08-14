import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
from pathlib import Path
from datetime import datetime

from pipeline import (
    SupervisedPreprocessor,
    UnsupervisedPreprocessor,
    process_supervised_dataset,
    process_unsupervised_dataset,
)


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Automated ML Data Pipeline",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Automated ML Data Pipeline")
st.caption(
    "EDA → preprocessing → downloadable processed data"
)


# ==========================================================
# HELPERS
# ==========================================================

def make_download_link(data: bytes, filename: str, label: str):
    b64 = base64.b64encode(data).decode()
    return (
        f'<a href="data:application/octet-stream;base64,{b64}" '
        f'download="{filename}">{label}</a>'
    )


def dataframe_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def create_eda_document(
    df,
    target_col=None,
    task=None,
    title="Exploratory Data Analysis Report"
):
    """
    Creates a self-contained HTML document for download.

    This intentionally contains tables/statistics rather than web-page
    visualizations. The user can download/open it as a document.
    """

    rows, cols = df.shape

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_cols = df.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    missing = (
        df.isna()
        .sum()
        .to_frame("Missing Count")
    )

    missing["Missing %"] = (
        missing["Missing Count"] / len(df) * 100
    )

    missing = missing.sort_values(
        "Missing %",
        ascending=False
    )

    missing = missing[
        missing["Missing Count"] > 0
    ]

    duplicates = int(df.duplicated().sum())

    numeric_summary = pd.DataFrame()

    if numeric_cols:
        numeric_summary = (
            df[numeric_cols]
            .describe()
            .T
        )

        numeric_summary["skewness"] = (
            df[numeric_cols]
            .skew()
        )

    categorical_summary = pd.DataFrame()

    if categorical_cols:
        categorical_summary = pd.DataFrame({
            "Feature": categorical_cols,
            "Unique Values": [
                df[col].nunique(dropna=True)
                for col in categorical_cols
            ],
            "Missing Values": [
                int(df[col].isna().sum())
                for col in categorical_cols
            ]
        }).sort_values(
            "Unique Values",
            ascending=False
        )

    target_section = ""

    if target_col and target_col in df.columns:

        target = df[target_col]

        target_section = f"""
        <h2>Target Analysis</h2>
        <p><b>Target:</b> {target_col}</p>
        <p><b>Detected task:</b> {task or "Not specified"}</p>
        {target.describe().to_frame("Value").to_html()}
        """

    def table_or_message(frame, message):
        if frame is None or frame.empty:
            return f"<p>{message}</p>"
        return frame.to_html(
            classes="data-table",
            border=0
        )

    generated = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>

<style>
body {{
    font-family: Arial, Helvetica, sans-serif;
    margin: 40px;
    line-height: 1.5;
    color: #222;
}}

h1 {{
    margin-bottom: 4px;
}}

h2 {{
    margin-top: 32px;
    border-bottom: 1px solid #ddd;
    padding-bottom: 6px;
}}

.meta {{
    color: #666;
    margin-bottom: 24px;
}}

.data-table {{
    border-collapse: collapse;
    width: 100%;
    margin: 12px 0 24px 0;
    font-size: 13px;
}}

.data-table th,
.data-table td {{
    border: 1px solid #ddd;
    padding: 7px;
    text-align: left;
}}

.data-table th {{
    font-weight: bold;
}}

.summary {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
}}

.card {{
    border: 1px solid #ddd;
    padding: 14px;
    border-radius: 8px;
}}

code {{
    background: #f4f4f4;
    padding: 2px 5px;
}}
</style>
</head>

<body>

<h1>{title}</h1>

<div class="meta">
Generated: {generated}
</div>

<h2>Dataset Overview</h2>

<div class="summary">

<div class="card">
<b>Rows</b><br>
{rows:,}
</div>

<div class="card">
<b>Columns</b><br>
{cols:,}
</div>

<div class="card">
<b>Numerical Features</b><br>
{len(numeric_cols):,}
</div>

<div class="card">
<b>Categorical Features</b><br>
{len(categorical_cols):,}
</div>

</div>

<p><b>Duplicate rows:</b> {duplicates:,}</p>

{target_section}

<h2>Numerical Feature Summary</h2>

{table_or_message(
    numeric_summary,
    "No numerical features were detected."
)}

<h2>Missing Values</h2>

{table_or_message(
    missing,
    "No missing values were detected."
)}

<h2>Categorical Feature Summary</h2>

{table_or_message(
    categorical_summary,
    "No categorical features were detected."
)}

<h2>Detected Feature Types</h2>

<h3>Numerical Features</h3>
<p>
{", ".join(numeric_cols) if numeric_cols else "None"}
</p>

<h3>Categorical Features</h3>
<p>
{", ".join(categorical_cols) if categorical_cols else "None"}
</p>

<h2>Notes</h2>

<p>
This report is generated from the uploaded dataset. It contains
tabular/statistical EDA rather than interactive web-page visualizations.
The report can be downloaded and opened independently.
</p>

</body>
</html>
"""

    return html.encode("utf-8")


def get_pipeline_info(processor):
    try:
        return processor.get_info()
    except Exception:
        return {}


def display_pipeline_info(info):
    if not info:
        return

    st.subheader("Pipeline Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Original Features",
            info.get(
                "original_feature_count",
                info.get(
                    "feature_count_before_processing",
                    "—"
                )
            )
        )

    with col2:
        st.metric(
            "After Encoding",
            info.get(
                "feature_count_after_encoding",
                "—"
            )
        )

    with col3:
        st.metric(
            "Final Features",
            info.get(
                "final_feature_count",
                info.get(
                    "selected_feature_count",
                    "—"
                )
            )
        )

    with col4:
        st.metric(
            "Task",
            str(info.get("task", "—")).title()
        )

    with st.expander("Pipeline details"):
        st.json(info)


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.header("Settings")

mode = st.sidebar.radio(
    "Pipeline mode",
    [
        "Supervised",
        "Unsupervised"
    ]
)

st.sidebar.info(
    "EDA visualizations and Sweetviz have been removed. "
    "Use the downloadable EDA report instead."
)


# ==========================================================
# UPLOAD
# ==========================================================

uploaded_file = st.file_uploader(
    "Upload CSV dataset",
    type=["csv"]
)

if uploaded_file is None:

    st.info(
        "Upload a CSV file to begin."
    )

    st.stop()


# ==========================================================
# LOAD DATA
# ==========================================================

try:

    df = pd.read_csv(
        uploaded_file
    )

except Exception as e:

    st.error(
        f"Could not read the CSV: {e}"
    )

    st.stop()


st.success(
    f"Loaded {uploaded_file.name}"
)

# Dataset preview only — no web-page visualization.
st.subheader("Dataset Preview")
st.dataframe(
    df.head(20),
    use_container_width=True
)

st.write(
    f"Rows: **{df.shape[0]:,}**  |  "
    f"Columns: **{df.shape[1]:,}**"
)


# ==========================================================
# EDA DOCUMENT
# ==========================================================

st.divider()
st.header("1. EDA Report")

target_col = None
task = None

if mode == "Supervised":

    target_col = st.selectbox(
        "Select target column",
        options=df.columns.tolist()
    )

    if target_col:
        y_preview = df[target_col]

        if (
            pd.api.types.is_object_dtype(y_preview)
            or pd.api.types.is_string_dtype(y_preview)
            or pd.api.types.is_bool_dtype(y_preview)
        ):
            task = "classification"

        elif pd.api.types.is_numeric_dtype(y_preview):

            task = (
                "classification"
                if y_preview.nunique(dropna=True) <= 20
                else "regression"
            )

        st.caption(
            f"Detected task: **{task}**"
        )


eda_bytes = create_eda_document(
    df,
    target_col=target_col,
    task=task
)

st.download_button(
    label="📄 Download EDA Report",
    data=eda_bytes,
    file_name=(
        Path(uploaded_file.name).stem
        + "_EDA_Report.html"
    ),
    mime="text/html"
)

st.caption(
    "The EDA report contains statistical tables and dataset diagnostics. "
    "No Sweetviz or interactive visualizations are embedded in the app."
)


# ==========================================================
# PROCESSING
# ==========================================================

st.divider()
st.header("2. Data Preprocessing")

if mode == "Supervised":

    if not target_col:

        st.warning(
            "Select a target column."
        )

        st.stop()

    if st.button(
        "⚙️ Run Supervised Pipeline",
        type="primary"
    ):

        try:

            with st.spinner(
                "Running supervised preprocessing..."
            ):

                result = process_supervised_dataset(
                    df=df,
                    target_col=target_col,
                    test_size=0.20,
                    random_state=42
                )

            processor = result["processor"]
            info = result["info"]

            st.success(
                "Supervised preprocessing completed."
            )

            display_pipeline_info(info)

            X_train = result["X_train"]
            X_test = result["X_test"]

            st.subheader("Processed Training Data")
            st.dataframe(
                X_train.head(20),
                use_container_width=True
            )

            st.subheader("Processed Test Data")
            st.dataframe(
                X_test.head(20),
                use_container_width=True
            )

            st.download_button(
                "⬇️ Download Processed Train CSV",
                data=dataframe_to_csv_bytes(
                    X_train
                ),
                file_name=(
                    Path(uploaded_file.name).stem
                    + "_processed_train.csv"
                ),
                mime="text/csv"
            )

            st.download_button(
                "⬇️ Download Processed Test CSV",
                data=dataframe_to_csv_bytes(
                    X_test
                ),
                file_name=(
                    Path(uploaded_file.name).stem
                    + "_processed_test.csv"
                ),
                mime="text/csv"
            )

        except Exception as e:

            st.error(
                "Supervised preprocessing failed."
            )

            st.exception(e)


else:

    if st.button(
        "⚙️ Run Unsupervised Pipeline",
        type="primary"
    ):

        try:

            with st.spinner(
                "Running unsupervised preprocessing..."
            ):

                result = process_unsupervised_dataset(
                    df=df,
                    test_size=0.20,
                    random_state=42
                )

            processor = result["processor"]
            info = result["info"]

            st.success(
                "Unsupervised preprocessing completed."
            )

            display_pipeline_info(info)

            X_train = result["X_train"]
            X_test = result["X_test"]

            st.subheader("Processed Training Data")
            st.dataframe(
                X_train.head(20),
                use_container_width=True
            )

            st.subheader("Processed Test Data")
            st.dataframe(
                X_test.head(20),
                use_container_width=True
            )

            st.download_button(
                "⬇️ Download Processed Train CSV",
                data=dataframe_to_csv_bytes(
                    X_train
                ),
                file_name=(
                    Path(uploaded_file.name).stem
                    + "_processed_train.csv"
                ),
                mime="text/csv"
            )

            st.download_button(
                "⬇️ Download Processed Test CSV",
                data=dataframe_to_csv_bytes(
                    X_test
                ),
                file_name=(
                    Path(uploaded_file.name).stem
                    + "_processed_test.csv"
                ),
                mime="text/csv"
            )

        except Exception as e:

            st.error(
                "Unsupervised preprocessing failed."
            )

            st.exception(e)


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    "Automated ML Data Pipeline • "
    "EDA is delivered as a downloadable document; "
    "web-page visualizations and Sweetviz are disabled."
)
