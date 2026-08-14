import streamlit as st
import pandas as pd
import requests
import io
import zipfile
import textwrap
import base64
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time


# ==========================================================
# CONFIGURATION
# ==========================================================

API_URL = "https://automated-ml-preprocessing-api.onrender.com/process"
EDA_API_URL = "https://automated-ml-preprocessing-api.onrender.com/eda"


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Auto ML Preprocessor",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown(
    textwrap.dedent(
        """
        <style>

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }

        .hero {
            padding: 2.4rem 2.6rem;
            border-radius: 18px;
            margin-bottom: 2rem;
            border: 1px solid rgba(128,128,128,0.20);
            background: linear-gradient(
                135deg,
                rgba(70,70,70,0.10),
                rgba(100,100,100,0.04)
            );
        }

        .hero-title {
            font-size: 2.65rem;
            font-weight: 750;
            margin-bottom: 0.4rem;
            line-height: 1.15;
        }

        .hero-subtitle {
            font-size: 1.18rem;
            font-weight: 600;
            margin-bottom: 0.7rem;
        }

        .hero-description {
            font-size: 1rem;
            opacity: 0.78;
            max-width: 900px;
            line-height: 1.6;
            margin-bottom: 1.25rem;
        }

        .workflow {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            align-items: center;
            font-size: 0.95rem;
            font-weight: 600;
        }

        .workflow-step {
            padding: 0.45rem 0.8rem;
            border-radius: 999px;
            border: 1px solid rgba(128,128,128,0.25);
            background: rgba(128,128,128,0.08);
        }

        .workflow-arrow {
            opacity: 0.5;
        }

        .section-card {
            padding: 1.25rem 1.5rem;
            border-radius: 14px;
            border: 1px solid rgba(128,128,128,0.20);
            margin-bottom: 1rem;
        }

        .section-card-title {
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }

        .section-card-description {
            opacity: 0.7;
            font-size: 0.92rem;
            line-height: 1.5;
        }

        .download-card {
            padding: 1.2rem 1.4rem;
            border-radius: 14px;
            border: 1px solid rgba(128,128,128,0.20);
            margin-bottom: 1rem;
        }

        .download-title {
            font-size: 1.2rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        .download-description {
            opacity: 0.7;
            font-size: 0.9rem;
        }

        .footer {
            text-align: center;
            opacity: 0.5;
            font-size: 0.85rem;
            padding-top: 2rem;
        }

        div.stButton > button {
            border-radius: 10px;
            font-weight: 650;
            min-height: 2.7rem;
        }

        div.stDownloadButton > button {
            border-radius: 10px;
            font-weight: 600;
            min-height: 2.6rem;
        }

        [data-testid="stFileUploader"] {
            border-radius: 14px;
        }

        [data-testid="stMetric"] {
            padding: 0.8rem;
            border-radius: 12px;
            border: 1px solid rgba(128,128,128,0.15);
        }

        [data-testid="stExpander"] {
            border-radius: 10px;
        }

        </style>
        """
    ),
    unsafe_allow_html=True
)


# ==========================================================
# SESSION STATE
# ==========================================================

defaults = {
    "processed": False,

    "zip_bytes": None,

    # Supervised
    "x_train_bytes": None,
    "x_test_bytes": None,

    # Unsupervised
    "processed_bytes": None,

    "pipeline_info_bytes": None,

    "eda_report_bytes": None,
    "eda_generated": False,

    # Locks
    "processing_running": False,
    "eda_running": False,
    "operation_process": None,
    "operation_result_path": None,
    "operation_request_path": None,
    "operation_tmp_dir": None,
    "operation_kind": None,
    "operation_metadata": {},
    "operation_message": None,
    "operation_error": None,
    "active_input_signature": None,

    # Metadata
    "processed_target": None,
    "processed_dataset_type": None,

    "previous_ml_task": None,
    "previous_dataset_type": None,
    "previous_unsupervised_dataset_type": None,
}


for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ==========================================================
# CLEAR RESULTS
# ==========================================================

def clear_results(clear_uploads=False):

    st.session_state.processed = False
    st.session_state.zip_bytes = None
    st.session_state.x_train_bytes = None
    st.session_state.x_test_bytes = None
    st.session_state.processed_bytes = None
    st.session_state.pipeline_info_bytes = None
    st.session_state.eda_report_bytes = None
    st.session_state.eda_generated = False
    st.session_state.processed_target = None
    st.session_state.processed_dataset_type = None
    st.session_state.operation_error = None
    st.session_state.operation_message = None

    if clear_uploads:
        for key in [
            "entire_dataset_upload", "training_dataset_upload",
            "test_mode_train_upload", "test_mode_test_upload",
            "unsupervised_entire_dataset_upload",
            "unsupervised_training_dataset_upload",
            "unsupervised_test_workflow_train_upload",
            "unsupervised_test_workflow_test_upload",
            "single_dataset_target", "test_dataset_target"
        ]:
            st.session_state.pop(key, None)

        st.session_state.active_input_signature = None


def operation_is_running():
    return (
        st.session_state.processing_running
        or st.session_state.eda_running
    )


def file_payload(uploaded_file):
    uploaded_file.seek(0)
    content = uploaded_file.getvalue()
    return {
        "name": uploaded_file.name,
        "content": base64.b64encode(content).decode("ascii"),
        "mime": "text/csv",
    }


def input_signature(*parts):
    digest = hashlib.sha256()
    for part in parts:
        if part is None:
            digest.update(b"<NONE>")
        elif isinstance(part, bytes):
            digest.update(part)
        else:
            digest.update(str(part).encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def sync_input_signature(signature):
    """Clear old outputs when the actual input/target changes."""
    previous = st.session_state.active_input_signature
    if previous is not None and previous != signature:
        clear_results(clear_uploads=False)
    st.session_state.active_input_signature = signature


def launch_http_operation(kind, endpoint, files, data, metadata=None):
    """Start API work in a separate process so the UI can be cancelled."""

    tmp_dir = tempfile.mkdtemp(prefix="auto_ml_operation_")
    request_path = os.path.join(tmp_dir, "request.json")
    result_path = os.path.join(tmp_dir, "result.json")
    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "operation_worker.py")

    # Convert Streamlit UploadedFile / requests-style file tuples into
    # JSON-safe payloads before handing the request to the worker process.
    # The previous version passed UploadedFile objects directly for EDA,
    # which caused: Object of type UploadedFile is not JSON serializable.
    serialized_files = {}
    for field, value in (files or {}).items():
        if isinstance(value, dict) and "content" in value:
            serialized_files[field] = value
            continue

        if isinstance(value, tuple) and len(value) >= 2:
            name = value[0]
            file_obj = value[1]
            mime = value[2] if len(value) >= 3 else "text/csv"

            if hasattr(file_obj, "getvalue"):
                content = file_obj.getvalue()
            elif isinstance(file_obj, (bytes, bytearray)):
                content = bytes(file_obj)
            elif hasattr(file_obj, "read"):
                current_pos = file_obj.tell() if hasattr(file_obj, "tell") else None
                content = file_obj.read()
                if current_pos is not None and hasattr(file_obj, "seek"):
                    file_obj.seek(current_pos)
            else:
                raise TypeError(
                    f"Unsupported file object for '{field}': "
                    f"{type(file_obj).__name__}"
                )

            serialized_files[field] = {
                "name": str(name),
                "content": base64.b64encode(content).decode("ascii"),
                "mime": str(mime),
            }
            continue

        raise TypeError(
            f"Unsupported file payload for '{field}': "
            f"{type(value).__name__}"
        )

    payload = {
        "endpoint": endpoint,
        "files": serialized_files,
        "data": data or {},
        "timeout": 300,
    }

    with open(request_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    process = subprocess.Popen([
        sys.executable,
        worker,
        request_path,
        result_path,
    ])

    st.session_state.operation_process = process
    st.session_state.operation_result_path = result_path
    st.session_state.operation_request_path = request_path
    st.session_state.operation_tmp_dir = tmp_dir
    st.session_state.operation_kind = kind
    st.session_state.operation_metadata = metadata or {}
    st.session_state.operation_error = None
    st.session_state.operation_message = (
        "Generating the EDA report..."
        if kind == "eda"
        else "Preprocessing the dataset..."
    )
    st.session_state.eda_running = kind == "eda"
    st.session_state.processing_running = kind == "process"


def cleanup_operation_files():
    tmp_dir = st.session_state.get("operation_tmp_dir")
    if tmp_dir and os.path.isdir(tmp_dir):
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    st.session_state.operation_process = None
    st.session_state.operation_result_path = None
    st.session_state.operation_request_path = None
    st.session_state.operation_tmp_dir = None
    st.session_state.operation_kind = None
    st.session_state.operation_metadata = {}


def cancel_current_operation():
    process = st.session_state.get("operation_process")
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3)
        except Exception:
            process.kill()

    kind = st.session_state.get("operation_kind")
    st.session_state.processing_running = False
    st.session_state.eda_running = False
    st.session_state.operation_error = None
    st.session_state.operation_message = None
    cleanup_operation_files()

    st.session_state.operation_error = (
        "EDA generation cancelled."
        if kind == "eda"
        else "Preprocessing cancelled."
    )


def finish_background_operation():
    """Collect a finished child-process request and update the UI state."""

    process = st.session_state.get("operation_process")
    if process is None:
        return False

    if process.poll() is None:
        return False

    kind = st.session_state.operation_kind
    result_path = st.session_state.operation_result_path

    result = None
    if result_path and os.path.exists(result_path):
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                result = json.load(f)
        except Exception as exc:
            result = {"ok": False, "error": str(exc), "content": ""}

    if not result:
        result = {
            "ok": False,
            "error": "The operation ended without a response.",
            "content": "",
        }

    st.session_state.processing_running = False
    st.session_state.eda_running = False

    if not result.get("ok"):
        st.session_state.operation_error = result.get(
            "error", "Unknown API error"
        )
        cleanup_operation_files()
        return True

    content = base64.b64decode(result.get("content", ""))

    if kind == "eda":
        st.session_state.eda_report_bytes = content
        st.session_state.eda_generated = True
        st.session_state.operation_message = (
            "EDA report generated successfully."
        )

    else:
        try:
            with zipfile.ZipFile(io.BytesIO(content), "r") as zip_file:
                names = zip_file.namelist()

                st.session_state.zip_bytes = content
                st.session_state.x_train_bytes = (
                    zip_file.read("X_train.csv")
                    if "X_train.csv" in names else None
                )
                st.session_state.x_test_bytes = (
                    zip_file.read("X_test.csv")
                    if "X_test.csv" in names else None
                )
                st.session_state.pipeline_info_bytes = (
                    zip_file.read("pipeline_info.txt")
                    if "pipeline_info.txt" in names else None
                )

            st.session_state.processed = True
            st.session_state.processed_target = st.session_state.operation_metadata.get(
                "target"
            )
            st.session_state.processed_dataset_type = st.session_state.operation_metadata.get(
                "dataset_type"
            )
            st.session_state.operation_message = (
                "Dataset processed successfully."
            )

        except zipfile.BadZipFile:
            st.session_state.operation_error = (
                "The API returned an invalid ZIP file."
            )

    cleanup_operation_files()
    return True


def render_operation_lock():
    """Show status/cancel while the background operation is running."""
    if not operation_is_running():
        return

    kind = st.session_state.operation_kind
    label = (
        "EDA report generation is in progress..."
        if kind == "eda"
        else "Dataset preprocessing is in progress..."
    )

    st.warning(label)
    st.caption(
        "All other controls are temporarily disabled. "
        "You can cancel the current operation below."
    )

    if st.button(
        "✖ Cancel Operation",
        use_container_width=True,
        key="cancel_current_operation"
    ):
        cancel_current_operation()
        st.rerun()

    if finish_background_operation():
        st.rerun()

    time.sleep(0.7)
    st.rerun()


def generate_eda_report_download(files, data, key):
    """Start the notebook-based EDA report as a cancellable operation."""

    disabled = (
        operation_is_running()
        or st.session_state.eda_generated
        or st.session_state.processed
    )

    if st.button(
        "📊 Generate EDA Report",
        use_container_width=True,
        key=key,
        disabled=disabled
    ):
        launch_http_operation(
            "eda",
            EDA_API_URL,
            files,
            data,
            metadata={"eda_key": key}
        )
        st.rerun()

    if (
        st.session_state.eda_generated
        and st.session_state.eda_report_bytes is not None
    ):
        st.download_button(
            "⬇️ Download EDA Report (HTML)",
            data=st.session_state.eda_report_bytes,
            file_name="EDA_report.html",
            mime="text/html",
            use_container_width=True,
            key=f"download_{key}"
        )


render_operation_lock()

if st.session_state.operation_error:
    st.error(st.session_state.operation_error)
    st.session_state.operation_error = None

if st.session_state.operation_message:
    st.success(st.session_state.operation_message)
    st.session_state.operation_message = None


# ==========================================================
# HERO
# ==========================================================

st.title("⚙️ Auto ML Preprocessor")

st.markdown(
    "### No more manual EDA. No more repetitive preprocessing."
)

st.write(
    "Upload your dataset and automatically perform "
    "exploratory data analysis, preprocessing, "
    "feature engineering and scaling. "
    "all in one workflow."
)

st.markdown(
    "**📁 Upload** → **📊 EDA** → **⚙️ Preprocess** → "
    "**🎯 Feature Processing** → **📥 Download**"
)

st.divider()



# ==========================================================
# LEARNING TYPE CARD
# ==========================================================

with st.container(border=True):
    st.subheader("🧠 Choose your learning type")
    st.caption(
        "Choose supervised learning when your dataset contains "
        "a target variable. Choose unsupervised learning when "
        "there is no target variable."
    )


# ==========================================================
# LEARNING TYPE
# ==========================================================

ml_task = st.radio(
    "Select learning type:",
    [
        "Supervised Learning",
        "Unsupervised Learning"
    ],
    horizontal=True,
    key="learning_type"
)


# ==========================================================
# LEARNING TYPE CHANGE
# ==========================================================

if (
    st.session_state.previous_ml_task is not None
    and
    st.session_state.previous_ml_task != ml_task
):

    clear_results(clear_uploads=True)
    st.session_state.previous_ml_task = ml_task
    st.rerun()

st.session_state.previous_ml_task = ml_task


# ==========================================================
# SUPERVISED LEARNING
# ==========================================================

if ml_task == "Supervised Learning":

    with st.container(border=True):
        st.subheader("📂 Choose your dataset workflow")
        st.caption(
            "Select how your supervised dataset is structured."
        )


    dataset_type = st.radio(
        "Select dataset type:",
        [
            "Entire Dataset",
            "Training Dataset",
            "Test Dataset"
        ],
        horizontal=True,
        key="supervised_dataset_type"
    )

    if (
        st.session_state.previous_dataset_type is not None
        and
        st.session_state.previous_dataset_type != dataset_type
    ):

        clear_results(clear_uploads=True)
        st.session_state.previous_dataset_type = dataset_type
        st.rerun()

    st.session_state.previous_dataset_type = dataset_type

    # Safe default used by Training/Test workflows too.
    supervised_test_size_percent = 20

    # ======================================================
    # ENTIRE DATASET
    # ======================================================

    if dataset_type == "Entire Dataset":

        st.info(
            "Your complete dataset will be automatically split "
            "into training and testing sets. The preprocessing "
            "pipeline will be fitted only on the training data."
        )

        supervised_test_size_percent = st.number_input(
            "Test dataset size (%)",
            min_value=1,
            max_value=99,
            value=20,
            step=1,
            help=(
                "Percentage of the complete dataset to reserve "
                "for testing. Default is 20%."
            ),
            key="supervised_test_size_percent"
        )

        uploaded_file = st.file_uploader(
            "📁 Upload your complete dataset",
            type=["csv"],
            key="entire_dataset_upload"
        )

        train_file = None
        test_file = None

    # ======================================================
    # TRAINING DATASET
    # ======================================================

    elif dataset_type == "Training Dataset":

        st.info(
            "Your uploaded file will be treated entirely "
            "as training data. No train/test split will "
            "be performed."
        )

        uploaded_file = st.file_uploader(
            "📁 Upload your training dataset",
            type=["csv"],
            key="training_dataset_upload"
        )

        train_file = None
        test_file = None

    # ======================================================
    # TEST DATASET
    # ======================================================

    else:

        st.info(
            "Upload both the training and test datasets. "
            "The preprocessing pipeline will be fitted on "
            "the training dataset and then applied to the "
            "test dataset."
        )

        uploaded_file = None

        train_file = st.file_uploader(
            "📁 Upload your training dataset",
            type=["csv"],
            key="test_mode_train_upload"
        )

        test_file = st.file_uploader(
            "📁 Upload your test dataset",
            type=["csv"],
            key="test_mode_test_upload"
        )

    # ======================================================
    # SINGLE DATASET WORKFLOW
    # ======================================================

    if (
        dataset_type in [
            "Entire Dataset",
            "Training Dataset"
        ]
        and
        uploaded_file is not None
    ):

        try:

            uploaded_file.seek(0)

            df = pd.read_csv(
                uploaded_file
            )

        except Exception as e:

            st.error(
                f"Could not read CSV file: {str(e)}"
            )

            st.stop()

        st.success(
            f"Dataset loaded successfully — "
            f"{df.shape[0]:,} rows × "
            f"{df.shape[1]:,} columns"
        )

        st.subheader(
            "👀 Dataset Preview"
        )

        st.dataframe(
            df.head(20),
            use_container_width=True
        )

        numerical_features = [
            column
            for column in df.columns
            if pd.api.types.is_numeric_dtype(
                df[column]
            )
        ]

        categorical_features = [
            column
            for column in df.columns
            if not pd.api.types.is_numeric_dtype(
                df[column]
            )
        ]

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:

            st.metric(
                "Rows",
                f"{df.shape[0]:,}"
            )

        with col2:

            st.metric(
                "Columns",
                f"{df.shape[1]:,}"
            )

        with col3:

            st.metric(
                "Numerical",
                len(numerical_features)
            )

        with col4:

            st.metric(
                "Categorical",
                len(categorical_features)
            )

        with col5:

            st.metric(
                "Missing Values",
                f"{int(df.isnull().sum().sum()):,}"
            )

        st.caption(
            f"Duplicate rows: "
            f"**{int(df.duplicated().sum()):,}**"
        )

        # ==================================================
        # TARGET
        # ==================================================

        st.subheader(
            "🎯 Target Selection"
        )

        target_column = st.selectbox(
            "Select Target Column",
            options=df.columns,
            index=len(df.columns) - 1,
            key="single_dataset_target"
        )

        sync_input_signature(
            input_signature(
                "supervised-single",
                dataset_type,
                target_column,
                uploaded_file.getvalue()
            )
        )

        st.warning(
            "EDA is optional and may take some time to generate. "
            "It does not affect the preprocessing process or its results."
        )

        st.caption(
            "EDA is available only as a downloadable HTML report generated "
            "from the supervised EDA notebook."
        )

        uploaded_file.seek(0)
        generate_eda_report_download(
            files={
                "file": (
                    uploaded_file.name,
                    uploaded_file,
                    "text/csv"
                )
            },
            data={
                "ml_task": "Supervised Learning",
                "dataset_type": dataset_type,
                "target": target_column
            },
            key="generate_eda_supervised_single"
        )
        uploaded_file.seek(0)


        # ==================================================
        # EDA
        # ==================================================


        # ==================================================
        # ==================================================


        # ==================================================
        # PROCESSING
        # ==================================================

        st.divider()

        st.subheader(
            "⚙️ Automated Processing"
        )

        st.caption(
            "The pipeline will automatically preprocess the data, "
            "engineer features."
        )

        processing_disabled = (
            operation_is_running()
            or st.session_state.eda_generated
            or st.session_state.processed
        )

        if st.button(
            "🚀 Process Dataset",
            use_container_width=True,
            key="process_single_dataset",
            disabled=processing_disabled
        ):
            uploaded_file.seek(0)
            launch_http_operation(
                "process",
                API_URL,
                {
                    "file": file_payload(uploaded_file)
                },
                {
                    "ml_task": "Supervised Learning",
                    "dataset_type": dataset_type,
                    "target": target_column,
                    "test_size": supervised_test_size_percent / 100
                },
                metadata={
                    "target": target_column,
                    "dataset_type": dataset_type
                }
            )
            st.rerun()

    # ======================================================
    # TEST DATASET WORKFLOW
    # ======================================================

    if (
        dataset_type == "Test Dataset"
        and
        train_file is not None
        and
        test_file is not None
    ):

        try:

            train_file.seek(0)

            train_df = pd.read_csv(
                train_file
            )

        except Exception as e:

            st.error(
                f"Could not read training dataset: {str(e)}"
            )

            st.stop()

        try:

            test_file.seek(0)

            test_df = pd.read_csv(
                test_file
            )

        except Exception as e:

            st.error(
                f"Could not read test dataset: {str(e)}"
            )

            st.stop()

        st.success(
            f"Training dataset loaded — "
            f"{train_df.shape[0]:,} rows × "
            f"{train_df.shape[1]:,} columns"
        )

        st.success(
            f"Test dataset loaded — "
            f"{test_df.shape[0]:,} rows × "
            f"{test_df.shape[1]:,} columns"
        )

        st.subheader(
            "👀 Dataset Preview"
        )

        train_tab, test_tab = st.tabs(
            [
                "🏋️ Training Dataset",
                "🧪 Test Dataset"
            ]
        )

        with train_tab:

            st.caption(
                "First 20 rows of the training dataset."
            )

            st.dataframe(
                train_df.head(20),
                use_container_width=True
            )

        with test_tab:

            st.caption(
                "First 20 rows of the test dataset."
            )

            st.dataframe(
                test_df.head(20),
                use_container_width=True
            )

        st.subheader(
            "🎯 Target Selection"
        )

        target_column = st.selectbox(
            "Select Target Column",
            options=train_df.columns,
            index=len(train_df.columns) - 1,
            help=(
                "The target is selected from the "
                "training dataset."
            ),
            key="test_dataset_target"
        )

        sync_input_signature(
            input_signature(
                "supervised-test",
                dataset_type,
                target_column,
                train_file.getvalue(),
                test_file.getvalue()
            )
        )

        st.warning(
            "EDA is optional and may take some time to generate. "
            "It does not affect preprocessing. For this workflow, "
            "EDA uses the training dataset only."
        )

        train_file.seek(0)
        generate_eda_report_download(
            files={
                "train_file": (
                    train_file.name,
                    train_file,
                    "text/csv"
                )
            },
            data={
                "ml_task": "Supervised Learning",
                "dataset_type": "Test Dataset",
                "target": target_column
            },
            key="generate_eda_supervised_test"
        )
        train_file.seek(0)


        st.info(
            "ℹ️ EDA is performed on the **training dataset only**. "
            "The test dataset is kept unseen because it should "
            "not influence preprocessing or preprocessing decisions."
        )



        st.divider()

        st.subheader(
            "⚙️ Automated Processing"
        )

        st.caption(
            "The preprocessing pipeline will be fitted on the "
            "training dataset and the learned parameters will "
            "then be applied to the test dataset."
        )

        process_disabled = (
            operation_is_running()
            or st.session_state.eda_generated
            or st.session_state.processed
        )

        if st.button(
            "🚀 Process Test Dataset",
            use_container_width=True,
            key="process_test_dataset",
            disabled=process_disabled
        ):
            train_file.seek(0)
            test_file.seek(0)
            launch_http_operation(
                "process",
                API_URL,
                {
                    "train_file": file_payload(train_file),
                    "test_file": file_payload(test_file)
                },
                {
                    "ml_task": "Supervised Learning",
                    "dataset_type": "Test Dataset",
                    "target": target_column
                },
                metadata={
                    "target": target_column,
                    "dataset_type": "Test Dataset"
                }
            )
            st.rerun()


# ==========================================================
# UNSUPERVISED LEARNING
# ==========================================================

else:

    with st.container(border=True):

        st.subheader(
            "🔬 Unsupervised Processing"
        )

        st.caption(
            "No target variable is used. Choose how your "
            "unsupervised dataset is structured."
        )


    # ======================================================
    # UNSUPERVISED DATASET WORKFLOW
    # ======================================================

    unsupervised_dataset_type = st.radio(
        "Select dataset type:",
        [
            "Entire Dataset",
            "Training Dataset",
            "Test Dataset"
        ],
        horizontal=True,
        key="unsupervised_dataset_type"
    )


    if (
        st.session_state.get(
            "previous_unsupervised_dataset_type"
        )
        is not None
        and
        st.session_state.previous_unsupervised_dataset_type
        != unsupervised_dataset_type
    ):

        clear_results(clear_uploads=True)
        st.session_state.previous_unsupervised_dataset_type = (
            unsupervised_dataset_type
        )
        st.rerun()


    st.session_state.previous_unsupervised_dataset_type = (
        unsupervised_dataset_type
    )


    # Safe default used by Training/Test workflows too.
    unsupervised_test_size_percent = 20

    # ======================================================
    # ENTIRE DATASET
    # ======================================================

    # Always initialize both upload variables so the selected
    # workflow can safely build the API request below.
    unsupervised_file = None
    unsupervised_train_file = None
    unsupervised_test_file = None


    # ======================================================
    # ENTIRE DATASET
    # ======================================================

    if (
        unsupervised_dataset_type
        == "Entire Dataset"
    ):

        st.info(
            "Your complete dataset will be automatically split "
            "into training and testing sets. The unsupervised "
            "preprocessing pipeline will be fitted only on "
            "the training data."
        )

        unsupervised_test_size_percent = st.number_input(
            "Test dataset size (%)",
            min_value=1,
            max_value=99,
            value=20,
            step=1,
            help=(
                "Percentage of the complete dataset to reserve "
                "for testing. Default is 20%."
            ),
            key="unsupervised_test_size_percent"
        )

        unsupervised_file = st.file_uploader(
            "📁 Upload your complete dataset",
            type=["csv"],
            key="unsupervised_entire_dataset_upload"
        )


    # ======================================================
    # TRAINING DATASET
    # ======================================================

    elif (
        unsupervised_dataset_type
        == "Training Dataset"
    ):

        st.info(
            "Upload your training dataset. The pipeline will "
            "fit the unsupervised preprocessing on this dataset "
            "without performing another train/test split."
        )

        unsupervised_file = st.file_uploader(
            "📁 Upload your training dataset",
            type=["csv"],
            key="unsupervised_training_dataset_upload"
        )


    # ======================================================
    # TEST DATASET
    # ======================================================

    else:

        st.info(
            "Upload both your training and test datasets. "
            "The preprocessing pipeline will be fitted only "
            "on the training dataset and then applied to the "
            "test dataset."
        )

        unsupervised_train_file = st.file_uploader(
            "📁 Upload your training dataset",
            type=["csv"],
            key="unsupervised_test_workflow_train_upload"
        )

        unsupervised_test_file = st.file_uploader(
            "📁 Upload your test dataset",
            type=["csv"],
            key="unsupervised_test_workflow_test_upload"
        )

        # Use the training dataset as the dataset displayed in
        # the existing EDA section. The test dataset is still
        # separately uploaded and sent to the backend.
        unsupervised_file = unsupervised_train_file


    if unsupervised_file is not None:

        try:

            unsupervised_file.seek(0)

            unsupervised_df = pd.read_csv(
                unsupervised_file
            )

        except Exception as e:

            st.error(
                f"Could not read CSV file: {str(e)}"
            )

            st.stop()


        st.success(
            f"Dataset loaded successfully — "
            f"{unsupervised_df.shape[0]:,} rows × "
            f"{unsupervised_df.shape[1]:,} columns"
        )

        sync_input_signature(
            input_signature(
                "unsupervised",
                unsupervised_dataset_type,
                unsupervised_file.getvalue(),
                (
                    unsupervised_test_file.getvalue()
                    if unsupervised_dataset_type == "Test Dataset"
                    and unsupervised_test_file is not None
                    else b""
                )
            )
        )


        st.subheader(
            "👀 Dataset Preview"
        )

        st.dataframe(
            unsupervised_df.head(20),
            use_container_width=True
        )

        if (
            unsupervised_dataset_type
            == "Test Dataset"
            and
            unsupervised_test_file is not None
        ):

            try:

                unsupervised_test_file.seek(0)

                unsupervised_test_df = pd.read_csv(
                    unsupervised_test_file
                )

                st.subheader(
                    "👀 Test Dataset Preview"
                )

                st.dataframe(
                    unsupervised_test_df.head(20),
                    use_container_width=True
                )

            except Exception as e:

                st.error(
                    f"Could not read test dataset: {str(e)}"
                )

                st.stop()



        numerical_features = [
            column
            for column in unsupervised_df.columns
            if pd.api.types.is_numeric_dtype(
                unsupervised_df[column]
            )
        ]


        categorical_features = [
            column
            for column in unsupervised_df.columns
            if not pd.api.types.is_numeric_dtype(
                unsupervised_df[column]
            )
        ]


        col1, col2, col3, col4, col5 = st.columns(5)


        with col1:

            st.metric(
                "Rows",
                f"{unsupervised_df.shape[0]:,}"
            )


        with col2:

            st.metric(
                "Columns",
                f"{unsupervised_df.shape[1]:,}"
            )


        with col3:

            st.metric(
                "Numerical",
                len(numerical_features)
            )


        with col4:

            st.metric(
                "Categorical",
                len(categorical_features)
            )


        with col5:

            st.metric(
                "Missing Values",
                f"{int(unsupervised_df.isnull().sum().sum()):,}"
            )


        st.caption(
            f"Duplicate rows: "
            f"**{int(unsupervised_df.duplicated().sum()):,}**"
        )

        st.warning(
            "EDA is optional and may take some time to generate. "
            "It does not affect the preprocessing process or its results."
        )

        st.caption(
            "EDA is available only as a downloadable HTML report generated "
            "from the unsupervised EDA notebook."
        )

        eda_file = (
            unsupervised_train_file
            if unsupervised_dataset_type == "Test Dataset"
            else unsupervised_file
        )

        eda_file.seek(0)
        eda_files = (
            {
                "train_file": (
                    eda_file.name,
                    eda_file,
                    "text/csv"
                )
            }
            if unsupervised_dataset_type == "Test Dataset"
            else {
                "file": (
                    eda_file.name,
                    eda_file,
                    "text/csv"
                )
            }
        )

        generate_eda_report_download(
            files=eda_files,
            data={
                "ml_task": "Unsupervised Learning",
                "dataset_type": unsupervised_dataset_type
            },
            key="generate_eda_unsupervised_single"
        )
        eda_file.seek(0)



        # PROCESSING
        # ==================================================

        st.divider()

        st.subheader(
            "⚙️ Automated Processing"
        )

        st.caption(
            "No target variable is used. The pipeline will "
            "detect ID-like columns, handle missing values, "
            "process categorical variables, transform skewed "
            "features and scale the resulting feature matrix."
        )

        processing_disabled = (
            operation_is_running()
            or st.session_state.eda_generated
            or st.session_state.processed
        )

        if st.button(
            "🚀 Process Unsupervised Dataset",
            use_container_width=True,
            key="process_unsupervised_dataset",
            disabled=processing_disabled
        ):
            if unsupervised_dataset_type == "Test Dataset":
                unsupervised_train_file.seek(0)
                unsupervised_test_file.seek(0)
                request_files = {
                    "train_file": file_payload(unsupervised_train_file),
                    "test_file": file_payload(unsupervised_test_file)
                }
            else:
                unsupervised_file.seek(0)
                request_files = {
                    "file": file_payload(unsupervised_file)
                }

            launch_http_operation(
                "process",
                API_URL,
                request_files,
                {
                    "ml_task": "Unsupervised Learning",
                    "dataset_type": unsupervised_dataset_type,
                    "test_size": (
                        unsupervised_test_size_percent / 100
                        if unsupervised_dataset_type == "Entire Dataset"
                        else 0.20
                    )
                },
                metadata={
                    "target": None,
                    "dataset_type": "Unsupervised Dataset"
                }
            )
            st.rerun()


# ==========================================================
# DOWNLOAD RESULTS
# ==========================================================

if (
    st.session_state.processed
    and
    st.session_state.zip_bytes is not None
):

    st.divider()

    with st.container(border=True):
        st.subheader("📥 Processed Dataset Ready")
        st.caption(
            "Your processed outputs are ready to download."
        )


    # ======================================================
    # UNSUPERVISED DOWNLOAD
    # ======================================================

    if (
        st.session_state.processed_dataset_type
        == "Unsupervised Dataset"
    ):

        # Unsupervised preprocessing now produces the same train/test
        # structure as supervised preprocessing.
        if (
            st.session_state.x_train_bytes
            is not None
        ):

            st.download_button(
                label="⬇️ Download X_train.csv",
                data=st.session_state.x_train_bytes,
                file_name="X_train.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_unsupervised_x_train"
            )

        if (
            st.session_state.x_test_bytes
            is not None
        ):

            st.download_button(
                label="⬇️ Download X_test.csv",
                data=st.session_state.x_test_bytes,
                file_name="X_test.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_unsupervised_x_test"
            )

        if (
            st.session_state.pipeline_info_bytes
            is not None
        ):

            st.download_button(
                label="📄 Download Pipeline Information",
                data=st.session_state.pipeline_info_bytes,
                file_name="pipeline_info.txt",
                mime="text/plain",
                use_container_width=True,
                key="download_unsupervised_pipeline_info"
            )

        st.download_button(
            label="📦 Download Complete Package",
            data=st.session_state.zip_bytes,
            file_name="processed_unsupervised_dataset.zip",
            mime="application/zip",
            use_container_width=True,
            key="download_unsupervised_zip"
        )

    # ======================================================
    # SUPERVISED DOWNLOAD
    # ======================================================

    else:

        if (
            st.session_state.x_train_bytes
            is not None
        ):

            st.download_button(
                label="⬇️ Download X_train.csv",
                data=st.session_state.x_train_bytes,
                file_name="X_train.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_x_train"
            )

        if (
            st.session_state.x_test_bytes
            is not None
        ):

            st.download_button(
                label="⬇️ Download X_test.csv",
                data=st.session_state.x_test_bytes,
                file_name="X_test.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_x_test"
            )

        if (
            st.session_state.pipeline_info_bytes
            is not None
        ):

            st.download_button(
                label="📄 Download Pipeline Information",
                data=st.session_state.pipeline_info_bytes,
                file_name="pipeline_info.txt",
                mime="text/plain",
                use_container_width=True,
                key="download_supervised_pipeline_info"
            )

        st.download_button(
            label="📦 Download Complete Package",
            data=st.session_state.zip_bytes,
            file_name="processed_dataset.zip",
            mime="application/zip",
            use_container_width=True,
            key="download_supervised_zip"
        )

        if st.session_state.processed_target:

            st.info(
                f"Processed target: "
                f"**{st.session_state.processed_target}**"
            )


# ==========================================================
# PROCESSED DATA PREVIEW
# ==========================================================

if st.session_state.processed:

    # ======================================================
    # UNSUPERVISED PREVIEW
    # ======================================================

    if (
        st.session_state.processed_dataset_type
        == "Unsupervised Dataset"
    ):

        if (
            st.session_state.x_train_bytes is not None
            or
            st.session_state.x_test_bytes is not None
        ):

            try:

                unsup_train_preview = None
                unsup_test_preview = None

                if (
                    st.session_state.x_train_bytes
                    is not None
                ):

                    unsup_train_preview = pd.read_csv(
                        io.BytesIO(
                            st.session_state.x_train_bytes
                        )
                    )

                if (
                    st.session_state.x_test_bytes
                    is not None
                ):

                    unsup_test_preview = pd.read_csv(
                        io.BytesIO(
                            st.session_state.x_test_bytes
                        )
                    )

                st.divider()

                st.subheader(
                    "🔍 Processed Dataset Preview"
                )

                # Training-only workflows should show ONLY X_train.
                # Test workflows can show both X_train and X_test.
                # This also handles Entire Dataset correctly.
                if (
                    unsup_train_preview is not None
                    and
                    unsup_test_preview is None
                ):

                    st.caption(
                        "Preview of the processed training output."
                    )

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(
                            "Training Rows",
                            f"{len(unsup_train_preview):,}"
                        )

                    with col2:

                        st.metric(
                            "Output Features",
                            f"{unsup_train_preview.shape[1]:,}"
                        )

                    with col3:

                        st.metric(
                            "Missing Values",
                            f"{int(unsup_train_preview.isnull().sum().sum()):,}"
                        )

                    st.dataframe(
                        unsup_train_preview.head(20),
                        use_container_width=True
                    )

                elif (
                    unsup_train_preview is None
                    and
                    unsup_test_preview is not None
                ):

                    st.caption(
                        "Preview of the processed test output."
                    )

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(
                            "Test Rows",
                            f"{len(unsup_test_preview):,}"
                        )

                    with col2:

                        st.metric(
                            "Output Features",
                            f"{unsup_test_preview.shape[1]:,}"
                        )

                    with col3:

                        st.metric(
                            "Missing Values",
                            f"{int(unsup_test_preview.isnull().sum().sum()):,}"
                        )

                    st.dataframe(
                        unsup_test_preview.head(20),
                        use_container_width=True
                    )

                else:

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:

                        st.metric(
                            "Training Rows",
                            f"{len(unsup_train_preview):,}"
                        )

                    with col2:

                        st.metric(
                            "Test Rows",
                            f"{len(unsup_test_preview):,}"
                        )

                    with col3:

                        st.metric(
                            "Output Features",
                            f"{unsup_train_preview.shape[1]:,}"
                        )

                    with col4:

                        st.metric(
                            "Missing Values",
                            f"{int(unsup_train_preview.isnull().sum().sum()):,}"
                        )

                    train_tab, test_tab = st.tabs(
                        [
                            "X_train.csv",
                            "X_test.csv"
                        ]
                    )

                    with train_tab:

                        st.dataframe(
                            unsup_train_preview.head(20),
                            use_container_width=True
                        )

                    with test_tab:

                        st.dataframe(
                            unsup_test_preview.head(20),
                            use_container_width=True
                        )

            except Exception as e:

                st.warning(
                    "Could not display processed unsupervised "
                    f"output: {str(e)}"
                )

    # ======================================================
    # SUPERVISED PREVIEW
    # ======================================================

    else:

        if (
            st.session_state.x_train_bytes
            is not None
            or
            st.session_state.x_test_bytes
            is not None
        ):

            try:

                x_train_preview = None
                x_test_preview = None

                if (
                    st.session_state.x_train_bytes
                    is not None
                ):

                    x_train_preview = pd.read_csv(
                        io.BytesIO(
                            st.session_state.x_train_bytes
                        )
                    )

                if (
                    st.session_state.x_test_bytes
                    is not None
                ):

                    x_test_preview = pd.read_csv(
                        io.BytesIO(
                            st.session_state.x_test_bytes
                        )
                    )

                st.divider()

                st.subheader(
                    "🔍 Processed Dataset Preview"
                )

                # --------------------------------------------------
                # TRAINING-ONLY SUPERVISED WORKFLOW
                # --------------------------------------------------

                if (
                    x_train_preview is not None
                    and
                    x_test_preview is None
                ):

                    st.caption(
                        "Preview of the processed training output."
                    )

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(
                            "Training Rows",
                            f"{x_train_preview.shape[0]:,}"
                        )

                    with col2:

                        st.metric(
                            "Output Features",
                            x_train_preview.shape[1]
                        )

                    with col3:

                        st.metric(
                            "Missing Values",
                            int(
                                x_train_preview
                                .isnull()
                                .sum()
                                .sum()
                            )
                        )

                    st.dataframe(
                        x_train_preview.head(20),
                        use_container_width=True
                    )

                # --------------------------------------------------
                # TEST-ONLY SUPERVISED WORKFLOW
                # --------------------------------------------------

                elif (
                    x_train_preview is None
                    and
                    x_test_preview is not None
                ):

                    st.caption(
                        "Preview of the processed test output."
                    )

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(
                            "Test Rows",
                            f"{x_test_preview.shape[0]:,}"
                        )

                    with col2:

                        st.metric(
                            "Output Features",
                            x_test_preview.shape[1]
                        )

                    with col3:

                        st.metric(
                            "Missing Values",
                            int(
                                x_test_preview
                                .isnull()
                                .sum()
                                .sum()
                            )
                        )

                    st.dataframe(
                        x_test_preview.head(20),
                        use_container_width=True
                    )

                # --------------------------------------------------
                # TRAIN + TEST SUPERVISED WORKFLOW
                # --------------------------------------------------

                else:

                    st.caption(
                        "Preview of the processed training "
                        "and testing outputs."
                    )

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:

                        st.metric(
                            "Training Rows",
                            f"{x_train_preview.shape[0]:,}"
                        )

                    with col2:

                        st.metric(
                            "Test Rows",
                            f"{x_test_preview.shape[0]:,}"
                        )

                    with col3:

                        st.metric(
                            "Output Features",
                            x_train_preview.shape[1]
                        )

                    with col4:

                        st.metric(
                            "Missing Values",
                            int(
                                x_train_preview
                                .isnull()
                                .sum()
                                .sum()
                            )
                        )

                    train_tab, test_tab = st.tabs(
                        [
                            "X_train.csv",
                            "X_test.csv"
                        ]
                    )

                    with train_tab:

                        st.dataframe(
                            x_train_preview.head(20),
                            use_container_width=True
                        )

                    with test_tab:

                        st.dataframe(
                            x_test_preview.head(20),
                            use_container_width=True
                        )

            except Exception as e:

                st.warning(
                    f"Could not display processed output: {str(e)}"
                )


# ==========================================================
# FOOTER
# ==========================================================

st.caption(
    "Auto ML Preprocessor · Downloadable EDA · "
    "Feature Engineering · Feature Processing"
)
