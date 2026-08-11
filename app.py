import streamlit as st
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
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Auto ML Preprocessor",
    page_icon="⚙️",
    layout="wide"
)


# ==========================================================
# TITLE
# ==========================================================

st.title("⚙️ Auto ML Preprocessor")

st.write(
    "Upload your dataset and automatically perform "
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
# PROCESS DATASET
# ==========================================================

if uploaded_file is not None:

    # ------------------------------------------
    # Read uploaded dataset
    # ------------------------------------------

    try:

        df = __import__("pandas").read_csv(
            uploaded_file
        )

    except Exception as e:

        st.error(
            f"Could not read the CSV file: {str(e)}"
        )

        st.stop()

    # ------------------------------------------
    # Dataset information
    # ------------------------------------------

    st.success(
        f"Dataset loaded successfully: "
        f"{df.shape[0]} rows × {df.shape[1]} columns"
    )

    # ------------------------------------------
    # Target selection
    # ------------------------------------------

    target_column = st.selectbox(
        "Select Target Column",
        options=df.columns
    )

    # ------------------------------------------
    # Process button
    # ------------------------------------------

    process_button = st.button(
        "🚀 Process Dataset",
        use_container_width=True
    )

    if process_button:

        with st.spinner(
            "Processing dataset... This may take a moment."
        ):

            try:

                # ----------------------------------
                # Reset file pointer
                # ----------------------------------

                uploaded_file.seek(0)

                # ----------------------------------
                # Send file to FastAPI
                # ----------------------------------

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

                # ----------------------------------
                # Handle API errors
                # ----------------------------------

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

                # ----------------------------------
                # Get ZIP from API
                # ----------------------------------

                zip_bytes = response.content

                # ----------------------------------
                # Extract files from ZIP
                # ----------------------------------

                with zipfile.ZipFile(
                    io.BytesIO(zip_bytes),
                    "r"
                ) as zip_file:

                    # Check files returned by API
                    files_in_zip = zip_file.namelist()

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
                            "The API response is missing "
                            f"the following files: "
                            f"{', '.join(missing_files)}"
                        )

                        st.stop()

                    # Extract individual files
                    x_train_bytes = zip_file.read(
                        "X_train.csv"
                    )

                    x_test_bytes = zip_file.read(
                        "X_test.csv"
                    )

                    pipeline_info_bytes = zip_file.read(
                        "pipeline_info.txt"
                    )

                # ----------------------------------
                # Success message
                # ----------------------------------

                st.success(
                    "Dataset processed successfully!"
                )

                # ----------------------------------
                # Download section
                # ----------------------------------

                st.subheader(
                    "📥 Download Results"
                )

                # ----------------------------------
                # Three individual buttons
                # ----------------------------------

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.download_button(
                        label="⬇️ X_train.csv",
                        data=x_train_bytes,
                        file_name="X_train.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                with col2:

                    st.download_button(
                        label="⬇️ X_test.csv",
                        data=x_test_bytes,
                        file_name="X_test.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                with col3:

                    st.download_button(
                        label="📄 Pipeline Info",
                        data=pipeline_info_bytes,
                        file_name="pipeline_info.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

                # ----------------------------------
                # ZIP download
                # ----------------------------------

                st.download_button(
                    label="📦 Download All Files (ZIP)",
                    data=zip_bytes,
                    file_name="processed_dataset.zip",
                    mime="application/zip",
                    use_container_width=True
                )

            except requests.exceptions.Timeout:

                st.error(
                    "The processing request timed out. "
                    "The dataset may be too large or the "
                    "backend may be waking up."
                )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not connect to the preprocessing "
                    "API. Please try again."
                )

            except Exception as e:

                st.error(
                    f"An unexpected error occurred: {str(e)}"
                )