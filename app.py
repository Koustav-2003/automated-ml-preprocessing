import streamlit as st
import pandas as pd
import numpy as np
import io
import zipfile
import requests
from pathlib import Path
from datetime import datetime

API_URL = "https://automated-ml-preprocessing-api.onrender.com/process"


# ==========================================================
# PAGE
# ==========================================================

st.set_page_config(
    page_title="Auto ML Preprocessor",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Auto ML Preprocessor")
st.caption(
    "EDA → preprocessing → feature engineering → feature selection → download"
)


# ==========================================================
# SESSION STATE
# ==========================================================

defaults = {
    "processed": False,
    "zip_bytes": None,
    "x_train_bytes": None,
    "x_test_bytes": None,
    "pipeline_info_bytes": None,
    "processed_target": None,
    "processed_dataset_type": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def clear_results():
    st.session_state.processed = False
    st.session_state.zip_bytes = None
    st.session_state.x_train_bytes = None
    st.session_state.x_test_bytes = None
    st.session_state.pipeline_info_bytes = None
    st.session_state.processed_target = None
    st.session_state.processed_dataset_type = None


# ==========================================================
# EDA DOCUMENT
# ==========================================================

def create_eda_document(
    df,
    target_col=None,
    task=None,
    title="Exploratory Data Analysis Report"
):
    numeric = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical = df.select_dtypes(
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

    missing = missing[
        missing["Missing Count"] > 0
    ]

    numeric_summary = (
        df[numeric].describe().T
        if numeric
        else pd.DataFrame()
    )

    if numeric:
        numeric_summary["skewness"] = (
            df[numeric].skew()
        )

    categorical_summary = pd.DataFrame()

    if categorical:
        categorical_summary = pd.DataFrame({
            "Feature": categorical,
            "Unique Values": [
                df[c].nunique(dropna=True)
                for c in categorical
            ],
            "Missing Values": [
                int(df[c].isna().sum())
                for c in categorical
            ]
        }).sort_values(
            "Unique Values",
            ascending=False
        )

    target_html = ""

    if target_col and target_col in df.columns:
        target_html = f"""
        <h2>Target Analysis</h2>
        <p><b>Target:</b> {target_col}</p>
        <p><b>Task:</b> {task or "Not specified"}</p>
        {df[target_col].describe().to_frame("Value").to_html(
            border=0,
            classes="table"
        )}
        """

    def table(frame, empty_message):
        if frame.empty:
            return f"<p>{empty_message}</p>"
        return frame.to_html(
            border=0,
            classes="table"
        )

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body {{
    font-family: Arial, sans-serif;
    margin: 40px;
    color: #222;
}}
h1 {{ margin-bottom: 4px; }}
h2 {{
    margin-top: 30px;
    border-bottom: 1px solid #ddd;
    padding-bottom: 5px;
}}
.table {{
    border-collapse: collapse;
    width: 100%;
    margin: 12px 0 24px;
    font-size: 13px;
}}
.table th, .table td {{
    border: 1px solid #ddd;
    padding: 7px;
}}
.table th {{
    font-weight: bold;
}}
.meta {{
    color: #666;
}}
</style>
</head>
<body>

<h1>{title}</h1>
<p class="meta">
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
</p>

<h2>Dataset Overview</h2>
<p><b>Rows:</b> {len(df):,}</p>
<p><b>Columns:</b> {df.shape[1]:,}</p>
<p><b>Numerical features:</b> {len(numeric):,}</p>
<p><b>Categorical features:</b> {len(categorical):,}</p>
<p><b>Duplicate rows:</b> {int(df.duplicated().sum()):,}</p>

{target_html}

<h2>Numerical Feature Summary</h2>
{table(numeric_summary, "No numerical features detected.")}

<h2>Missing Values</h2>
{table(missing, "No missing values detected.")}

<h2>Categorical Feature Summary</h2>
{table(categorical_summary, "No categorical features detected.")}

<h2>Feature Lists</h2>
<h3>Numerical</h3>
<p>{", ".join(numeric) if numeric else "None"}</p>

<h3>Categorical</h3>
<p>{", ".join(categorical) if categorical else "None"}</p>

</body>
</html>
"""

    return html.encode("utf-8")


# ==========================================================
# LEARNING TYPE
# ==========================================================

learning_type = st.radio(
    "Select learning type:",
    [
        "Supervised Learning",
        "Unsupervised Learning"
    ],
    horizontal=True
)

st.divider()


# ==========================================================
# DATASET WORKFLOW
# ==========================================================

dataset_type = st.radio(
    "Select dataset type:",
    [
        "Entire Dataset",
        "Training Dataset",
        "Test Dataset"
    ],
    horizontal=True
)

if dataset_type == "Entire Dataset":

    st.info(
        "Upload the complete dataset. The pipeline will "
        "automatically create training and test data."
    )

    test_size_percent = st.number_input(
        "Test dataset size (%)",
        min_value=1,
        max_value=99,
        value=20,
        step=1
    )

    uploaded_file = st.file_uploader(
        "📁 Upload complete dataset",
        type=["csv"],
        key=f"entire_{learning_type}"
    )

    train_file = None
    test_file = None

elif dataset_type == "Training Dataset":

    st.info(
        "Upload only the training dataset. "
        "No additional train/test split will be performed."
    )

    uploaded_file = st.file_uploader(
        "📁 Upload training dataset",
        type=["csv"],
        key=f"train_{learning_type}"
    )

    train_file = None
    test_file = None
    test_size_percent = 20

else:

    st.info(
        "Upload BOTH the training and test datasets. "
        "The pipeline will fit only on training data and "
        "then transform the test data."
    )

    uploaded_file = None

    train_file = st.file_uploader(
        "🏋️ Upload training dataset",
        type=["csv"],
        key=f"paired_train_{learning_type}"
    )

    test_file = st.file_uploader(
        "🧪 Upload test dataset",
        type=["csv"],
        key=f"paired_test_{learning_type}"
    )

    test_size_percent = 20


# ==========================================================
# LOAD DATA FOR EDA
# ==========================================================

eda_df = None
target_col = None
task = None

if dataset_type in [
    "Entire Dataset",
    "Training Dataset"
] and uploaded_file is not None:

    try:
        uploaded_file.seek(0)
        eda_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

elif (
    dataset_type == "Test Dataset"
    and train_file is not None
):

    try:
        train_file.seek(0)
        eda_df = pd.read_csv(train_file)
    except Exception as e:
        st.error(f"Could not read training CSV: {e}")
        st.stop()


# ==========================================================
# DATASET PREVIEW + TARGET
# ==========================================================

if eda_df is not None:

    st.divider()
    st.subheader("👀 Dataset Preview")

    st.dataframe(
        eda_df.head(20),
        use_container_width=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Rows", f"{len(eda_df):,}")

    with c2:
        st.metric("Columns", f"{eda_df.shape[1]:,}")

    with c3:
        st.metric(
            "Numerical",
            len(
                eda_df.select_dtypes(
                    include=np.number
                ).columns
            )
        )

    with c4:
        st.metric(
            "Missing Values",
            f"{int(eda_df.isna().sum().sum()):,}"
        )

    if learning_type == "Supervised Learning":

        target_col = st.selectbox(
            "🎯 Select target column",
            eda_df.columns.tolist(),
            index=len(eda_df.columns) - 1
        )

        y = eda_df[target_col]

        if (
            pd.api.types.is_object_dtype(y)
            or pd.api.types.is_string_dtype(y)
            or pd.api.types.is_bool_dtype(y)
        ):
            task = "classification"
        elif pd.api.types.is_numeric_dtype(y):
            task = (
                "classification"
                if y.nunique(dropna=True) <= 20
                else "regression"
            )

        st.caption(
            f"Detected task: **{task}**"
        )


# ==========================================================
# EDA DOWNLOAD
# ==========================================================

if eda_df is not None:

    st.divider()
    st.subheader("📄 EDA Report")

    if dataset_type == "Test Dataset":
        st.caption(
            "EDA is generated from the training dataset only. "
            "The test dataset remains unseen."
        )

    eda_bytes = create_eda_document(
        eda_df,
        target_col=target_col,
        task=task
    )

    st.download_button(
        "⬇️ Download EDA Report",
        data=eda_bytes,
        file_name=(
            Path(
                train_file.name
                if dataset_type == "Test Dataset"
                else uploaded_file.name
            ).stem
            + "_EDA_Report.html"
        ),
        mime="text/html",
        use_container_width=True
    )


# ==========================================================
# PROCESS
# ==========================================================

ready = (
    (
        dataset_type in [
            "Entire Dataset",
            "Training Dataset"
        ]
        and uploaded_file is not None
    )
    or
    (
        dataset_type == "Test Dataset"
        and train_file is not None
        and test_file is not None
    )
)

if ready:

    st.divider()
    st.subheader("⚙️ Automated Processing")

    if st.button(
        "🚀 Process Dataset",
        type="primary",
        use_container_width=True
    ):

        try:

            if dataset_type == "Test Dataset":

                train_file.seek(0)
                test_file.seek(0)

                files = {
                    "train_file": (
                        train_file.name,
                        train_file,
                        "text/csv"
                    ),
                    "test_file": (
                        test_file.name,
                        test_file,
                        "text/csv"
                    )
                }

            else:

                uploaded_file.seek(0)

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file,
                        "text/csv"
                    )
                }

            data = {
                "ml_task": learning_type,
                "dataset_type": dataset_type,
                "test_size": test_size_percent / 100,
                "random_state": 42
            }

            if learning_type == "Supervised Learning":
                data["target"] = target_col

            with st.spinner(
                "Running preprocessing pipeline..."
            ):

                response = requests.post(
                    API_URL,
                    files=files,
                    data=data,
                    timeout=300
                )

            if response.status_code != 200:

                try:
                    detail = response.json().get(
                        "detail",
                        "Unknown API error"
                    )
                except Exception:
                    detail = response.text

                st.error(
                    f"Processing failed: {detail}"
                )

            else:

                st.session_state.zip_bytes = (
                    response.content
                )

                with zipfile.ZipFile(
                    io.BytesIO(response.content),
                    "r"
                ) as z:

                    names = z.namelist()

                    st.session_state.x_train_bytes = (
                        z.read("X_train.csv")
                        if "X_train.csv" in names
                        else None
                    )

                    st.session_state.x_test_bytes = (
                        z.read("X_test.csv")
                        if "X_test.csv" in names
                        else None
                    )

                    st.session_state.pipeline_info_bytes = (
                        z.read("pipeline_info.txt")
                        if "pipeline_info.txt" in names
                        else None
                    )

                st.session_state.processed = True
                st.session_state.processed_target = (
                    target_col
                )
                st.session_state.processed_dataset_type = (
                    dataset_type
                )

                st.success(
                    "✅ Dataset processed successfully!"
                )

        except requests.exceptions.ConnectionError:

            st.error(
                "Could not connect to the preprocessing API. "
                "Start main.py first."
            )

        except requests.exceptions.Timeout:

            st.error(
                "The preprocessing request timed out."
            )

        except zipfile.BadZipFile:

            st.error(
                "The API returned an invalid ZIP file."
            )

        except Exception as e:

            st.exception(e)


# ==========================================================
# DOWNLOADS
# ==========================================================

if st.session_state.processed:

    st.divider()
    st.subheader("📥 Download Processed Data")

    x_train = st.session_state.x_train_bytes
    x_test = st.session_state.x_test_bytes

    if x_train is not None:

        st.download_button(
            "⬇️ Download X_train.csv",
            data=x_train,
            file_name="X_train.csv",
            mime="text/csv",
            use_container_width=True
        )

    if x_test is not None:

        st.download_button(
            "⬇️ Download X_test.csv",
            data=x_test,
            file_name="X_test.csv",
            mime="text/csv",
            use_container_width=True
        )

    if st.session_state.pipeline_info_bytes:

        st.download_button(
            "📄 Download Pipeline Information",
            data=st.session_state.pipeline_info_bytes,
            file_name="pipeline_info.txt",
            mime="text/plain",
            use_container_width=True
        )

    if st.session_state.zip_bytes:

        st.download_button(
            "📦 Download Complete Package",
            data=st.session_state.zip_bytes,
            file_name="processed_dataset.zip",
            mime="application/zip",
            use_container_width=True
        )

    st.subheader("🔍 Processed Data Preview")

    if x_train is not None:
        train_preview = pd.read_csv(
            io.BytesIO(x_train)
        )

        st.write("**X_train.csv**")
        st.dataframe(
            train_preview.head(20),
            use_container_width=True
        )

    if x_test is not None:
        test_preview = pd.read_csv(
            io.BytesIO(x_test)
        )

        st.write("**X_test.csv**")
        st.dataframe(
            test_preview.head(20),
            use_container_width=True
        )
