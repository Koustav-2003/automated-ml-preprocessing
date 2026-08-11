import streamlit as st
import pandas as pd
import requests
import io
import zipfile


# ==========================================================
# CONFIGURATION
# ==========================================================

API_URL = (
    "https://automated-ml-preprocessing-api.onrender.com/process"
)


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Auto ML Preprocessor",
    page_icon="⚙️",
    layout="wide"
)


# ==========================================================
# SESSION STATE
# ==========================================================

if "processed" not in st.session_state:
    st.session_state.processed = False

if "zip_bytes" not in st.session_state:
    st.session_state.zip_bytes = None

if "x_train_bytes" not in st.session_state:
    st.session_state.x_train_bytes = None

if "x_test_bytes" not in st.session_state:
    st.session_state.x_test_bytes = None

if "pipeline_info_bytes" not in st.session_state:
    st.session_state.pipeline_info_bytes = None

if "processed_target" not in st.session_state:
    st.session_state.processed_target = None


# ==========================================================
# TITLE
# ==========================================================

st.title("⚙️ Auto ML Preprocessor")

st.write(
    "Upload your CSV dataset and automatically perform "
    "data preprocessing, feature engineering, scaling "
    "and feature selection."
)


# ==========================================================
# FILE UPLOAD
# ==========================================================

uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)


# ==========================================================
# WHEN FILE IS UPLOADED
# ==========================================================

if uploaded_file is not None:

    # ------------------------------------------------------
    # Read CSV
    # ------------------------------------------------------

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

    # ------------------------------------------------------
    # Dataset information
    # ------------------------------------------------------

    st.success(
        f"Dataset loaded successfully — "
        f"{df.shape[0]} rows × {df.shape[1]} columns"
    )

    # ------------------------------------------------------
    # Preview
    # ------------------------------------------------------

    st.subheader("👀 Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    # ------------------------------------------------------
    # Dataset information
    # ------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Rows",
            df.shape[0]
        )

    with col2:

        st.metric(
            "Columns",
            df.shape[1]
        )

    with col3:

        st.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

    # ------------------------------------------------------
    # Target column
    # ------------------------------------------------------

    st.subheader("🎯 Target Selection")

    target_column = st.selectbox(
        "Select the target column",
        options=df.columns
    )

    # ======================================================
    # PROCESS DATASET
    # ======================================================

    if st.button(
        "🚀 Process Dataset",
        use_container_width=True
    ):

        with st.spinner(
            "Processing dataset... Please wait."
        ):

            try:

                # ------------------------------------------
                # Reset file pointer
                # ------------------------------------------

                uploaded_file.seek(0)

                # ------------------------------------------
                # Send request to FastAPI
                # ------------------------------------------

                response = requests.post(
                    API_URL,
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file,
                            "text/csv"
                        )
                    },
                    data={
                        "target": target_column
                    },
                    timeout=300
                )

                # ------------------------------------------
                # API error
                # ------------------------------------------

                if response.status_code != 200:

                    try:

                        error_detail = (
                            response.json()
                            .get(
                                "detail",
                                "Unknown API error"
                            )
                        )

                    except Exception:

                        error_detail = response.text

                    st.error(
                        f"Processing failed: "
                        f"{error_detail}"
                    )

                    st.stop()

                # ------------------------------------------
                # Store ZIP in session state
                # ------------------------------------------

                zip_bytes = response.content

                st.session_state.zip_bytes = (
                    zip_bytes
                )

                # ------------------------------------------
                # Extract individual files
                # ------------------------------------------

                with zipfile.ZipFile(
                    io.BytesIO(zip_bytes),
                    "r"
                ) as zip_file:

                    files_in_zip = (
                        zip_file.namelist()
                    )

                    required_files = [
                        "X_train.csv",
                        "X_test.csv",
                        "pipeline_info.txt"
                    ]

                    missing_files = [
                        file
                        for file in required_files
                        if file not in files_in_zip
                    ]

                    if missing_files:

                        st.error(
                            "The API did not return "
                            "the expected files: "
                            + ", ".join(
                                missing_files
                            )
                        )

                        st.stop()

                    # ----------------------------------
                    # Save files in session state
                    # ----------------------------------

                    st.session_state.x_train_bytes = (
                        zip_file.read(
                            "X_train.csv"
                        )
                    )

                    st.session_state.x_test_bytes = (
                        zip_file.read(
                            "X_test.csv"
                        )
                    )

                    st.session_state.pipeline_info_bytes = (
                        zip_file.read(
                            "pipeline_info.txt"
                        )
                    )

                # ------------------------------------------
                # Store processing state
                # ------------------------------------------

                st.session_state.processed = True

                st.session_state.processed_target = (
                    target_column
                )

                st.success(
                    "✅ Dataset processed successfully!"
                )

            # ----------------------------------------------
            # Connection error
            # ----------------------------------------------

            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not connect to the preprocessing API."
                )

            # ----------------------------------------------
            # Timeout
            # ----------------------------------------------

            except requests.exceptions.Timeout:

                st.error(
                    "The request timed out. "
                    "The backend may be waking up or "
                    "the dataset may be too large."
                )

            # ----------------------------------------------
            # General error
            # ----------------------------------------------

            except Exception as e:

                st.error(
                    f"An unexpected error occurred: {str(e)}"
                )


# ==========================================================
# DOWNLOAD SECTION
# ==========================================================

if (
    st.session_state.processed
    and st.session_state.zip_bytes is not None
):

    st.divider()

    st.subheader("📥 Download Results")

    st.write(
        "Your processed files are ready. "
        "You can download them individually or "
        "download everything as a ZIP."
    )

    # ======================================================
    # THREE INDIVIDUAL DOWNLOAD BUTTONS
    # ======================================================

    col1, col2, col3 = st.columns(3)

    # ------------------------------------------------------
    # X TRAIN
    # ------------------------------------------------------

    with col1:

        st.download_button(
            label="⬇️ X_train.csv",
            data=st.session_state.x_train_bytes,
            file_name="X_train.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_x_train"
        )

    # ------------------------------------------------------
    # X TEST
    # ------------------------------------------------------

    with col2:

        st.download_button(
            label="⬇️ X_test.csv",
            data=st.session_state.x_test_bytes,
            file_name="X_test.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_x_test"
        )

    # ------------------------------------------------------
    # PIPELINE INFO
    # ------------------------------------------------------

    with col3:

        st.download_button(
            label="📄 Pipeline Info",
            data=st.session_state.pipeline_info_bytes,
            file_name="pipeline_info.txt",
            mime="text/plain",
            use_container_width=True,
            key="download_pipeline_info"
        )

    # ======================================================
    # ZIP DOWNLOAD
    # ======================================================

    st.download_button(
        label="📦 Download All Files (ZIP)",
        data=st.session_state.zip_bytes,
        file_name="processed_dataset.zip",
        mime="application/zip",
        use_container_width=True,
        key="download_all_files"
    )

    # ======================================================
    # PROCESSING COMPLETE INFO
    # ======================================================

    if st.session_state.processed_target:

        st.info(
            f"Processed target: "
            f"**{st.session_state.processed_target}**"
        )
        
        # ==========================================================
# ==========================================================
# OUTPUT PREVIEW
# ==========================================================

if (
    st.session_state.processed
    and st.session_state.x_train_bytes
    and st.session_state.x_test_bytes
):

    st.divider()

    st.subheader("🔍 Processed Dataset Preview")

    try:

        # --------------------------------------------------
        # Convert processed files back to DataFrames
        # --------------------------------------------------

        x_train_preview = pd.read_csv(
            io.BytesIO(
                st.session_state.x_train_bytes
            )
        )

        x_test_preview = pd.read_csv(
            io.BytesIO(
                st.session_state.x_test_bytes
            )
        )

        # --------------------------------------------------
        # Dataset statistics
        # --------------------------------------------------

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Training Rows",
                x_train_preview.shape[0]
            )

        with col2:

            st.metric(
                "Test Rows",
                x_test_preview.shape[0]
            )

        with col3:

            st.metric(
                "Selected Features",
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

        # --------------------------------------------------
        # Train / Test tabs
        # --------------------------------------------------

        train_tab, test_tab = st.tabs(
            [
                "X_train.csv",
                "X_test.csv"
            ]
        )

        # --------------------------------------------------
        # X TRAIN PREVIEW
        # --------------------------------------------------

        with train_tab:

            st.write(
                "First 20 rows of the processed "
                "training dataset:"
            )

            st.dataframe(
                x_train_preview.head(20),
                use_container_width=True
            )

        # --------------------------------------------------
        # X TEST PREVIEW
        # --------------------------------------------------

        with test_tab:

            st.write(
                "First 20 rows of the processed "
                "test dataset:"
            )

            st.dataframe(
                x_test_preview.head(20),
                use_container_width=True
            )

    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError
    ):

        # Don't show an error/warning to the user.
        # Simply skip the preview.
        pass

    except Exception:

        # Prevent preview problems from crashing
        # the rest of the application.
        pass