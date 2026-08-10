import streamlit as st
import pandas as pd
import requests


# =========================================================
# CONFIG
# =========================================================

API_URL = "https://automated-ml-preprocessing-api.onrender.com/process"


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="Auto Data Preprocessing",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🤖 Automated Data Preprocessing")

st.write(
    "Upload a CSV dataset and automatically perform "
    "feature engineering and feature selection."
)


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)


if uploaded_file is not None:

    # -----------------------------------------------------
    # Read dataset
    # -----------------------------------------------------

    try:

        df = pd.read_csv(
            uploaded_file
        )

    except Exception as e:

        st.error(
            f"Could not read the CSV: {e}"
        )

        st.stop()


    # -----------------------------------------------------
    # Dataset information
    # -----------------------------------------------------

    st.subheader("Dataset")

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


    # -----------------------------------------------------
    # Preview
    # -----------------------------------------------------

    st.subheader("Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


    # -----------------------------------------------------
    # Target
    # -----------------------------------------------------

    st.subheader("Configuration")

    target = st.selectbox(
        "Select the target column",
        options=df.columns
    )


    # -----------------------------------------------------
    # Process button
    # -----------------------------------------------------

    if st.button(
        "🚀 Process Dataset",
        type="primary"
    ):

        # Reset file pointer
        uploaded_file.seek(0)


        with st.spinner(
            "Processing dataset..."
        ):

            try:

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
                        "target": target
                    },
                    timeout=300
                )


            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not connect to the FastAPI server. "
                    "Make sure FastAPI is running."
                )

                st.stop()


            except requests.exceptions.Timeout:

                st.error(
                    "The request timed out."
                )

                st.stop()


        # =================================================
        # SUCCESS
        # =================================================

        if response.status_code == 200:

            st.success(
                "Dataset processed successfully! 🎉"
            )


            # ---------------------------------------------
            # Download
            # ---------------------------------------------

            st.download_button(
                label="⬇️ Download Processed Dataset",
                data=response.content,
                file_name="processed_dataset.zip",
                mime="application/zip"
            )


            st.info(
                "The ZIP contains X_train.csv, "
                "X_test.csv and pipeline_info.txt."
            )


        # =================================================
        # ERROR
        # =================================================

        else:

            try:

                error_detail = (
                    response.json()
                    .get("detail", response.text)
                )

            except Exception:

                error_detail = response.text


            st.error(
                "Processing failed."
            )

            st.code(
                error_detail
            )