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

EDA_API_URL = (
    "https://automated-ml-preprocessing-api.onrender.com/eda"
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

defaults = {
    "processed": False,
    "zip_bytes": None,
    "x_train_bytes": None,
    "x_test_bytes": None,
    "pipeline_info_bytes": None,
    "eda_report_bytes": None,
    "eda_generated": False,
    "processed_target": None,
    "processed_dataset_type": None,
    "previous_dataset_type": None,
}

for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ==========================================================
# CLEAR RESULTS
# ==========================================================

def clear_results():

    st.session_state.processed = False

    st.session_state.zip_bytes = None

    st.session_state.x_train_bytes = None

    st.session_state.x_test_bytes = None

    st.session_state.pipeline_info_bytes = None

    st.session_state.eda_report_bytes = None

    st.session_state.eda_generated = False

    st.session_state.processed_target = None

    st.session_state.processed_dataset_type = None


# ==========================================================
# FEATURE ANALYSIS
# ==========================================================

def render_feature_analysis(df, feature):

    st.markdown(
        f"### 🔎 {feature}"
    )

    data = df[feature]

    # ------------------------------------------------------
    # Numerical
    # ------------------------------------------------------

    if pd.api.types.is_numeric_dtype(data):

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Type",
                "Numerical"
            )

        with col2:

            st.metric(
                "Missing",
                int(data.isnull().sum())
            )

        with col3:

            st.metric(
                "Unique Values",
                int(data.nunique())
            )

        with col4:

            mean_value = data.mean()

            if pd.isna(mean_value):

                mean_text = "N/A"

            else:

                mean_text = f"{mean_value:.3f}"

            st.metric(
                "Mean",
                mean_text
            )

        stats_col1, stats_col2 = st.columns(2)

        with stats_col1:

            st.write("**Distribution**")

            clean_data = data.dropna()

            if not clean_data.empty:

                st.bar_chart(
                    clean_data
                    .value_counts()
                    .sort_index()
                )

            else:

                st.info(
                    "No numerical values available."
                )

        with stats_col2:

            st.write("**Summary Statistics**")

            stats = pd.DataFrame({
                "Statistic": [
                    "Mean",
                    "Median",
                    "Std Dev",
                    "Minimum",
                    "Maximum",
                    "Skewness"
                ],
                "Value": [
                    data.mean(),
                    data.median(),
                    data.std(),
                    data.min(),
                    data.max(),
                    data.skew()
                ]
            })

            st.dataframe(
                stats,
                hide_index=True,
                use_container_width=True
            )

    # ------------------------------------------------------
    # Categorical
    # ------------------------------------------------------

    else:

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Type",
                "Categorical"
            )

        with col2:

            st.metric(
                "Missing",
                int(data.isnull().sum())
            )

        with col3:

            st.metric(
                "Unique Values",
                int(data.nunique())
            )

        st.write("**Top Categories**")

        value_counts = (
            data
            .fillna("Missing")
            .astype(str)
            .value_counts()
            .head(10)
        )

        if not value_counts.empty:

            st.bar_chart(
                value_counts
            )

        else:

            st.info(
                "No categorical values available."
            )


# ==========================================================
# SELECT FIVE FEATURES
# ==========================================================

def select_eda_features(
    df,
    target=None
):

    available_features = [
        column
        for column in df.columns
        if column != target
    ]

    if not available_features:

        return []

    scored_features = []

    for feature in available_features:

        series = df[feature]

        missing_ratio = (
            series.isnull().mean()
        )

        if pd.api.types.is_numeric_dtype(series):

            unique_count = series.nunique()

            if unique_count > 1:

                try:

                    skewness = abs(
                        series.skew()
                    )

                    if pd.isna(skewness):

                        skewness = 0

                except Exception:

                    skewness = 0

            else:

                skewness = 0

            score = (
                missing_ratio * 2
                + min(skewness, 10) / 10
                + 0.2
            )

        else:

            unique_count = series.nunique()

            score = (
                missing_ratio * 2
                + min(unique_count, 50) / 50
            )

        scored_features.append(
            (
                feature,
                score
            )
        )

    scored_features.sort(
        key=lambda x: x[1],
        reverse=True
    )

    selected = [
        feature
        for feature, score
        in scored_features[:5]
    ]

    if len(selected) < 5:

        for feature in available_features:

            if feature not in selected:

                selected.append(
                    feature
                )

            if len(selected) == 5:

                break

    return selected[:5]


# ==========================================================
# GENERATE FULL EDA
# ==========================================================

def generate_full_eda(
    file,
    target
):

    try:

        file.seek(0)

        response = requests.post(
            EDA_API_URL,
            files={
                "file": (
                    file.name,
                    file,
                    "text/csv"
                )
            },
            data={
                "target": target
            },
            timeout=300
        )

        if response.status_code != 200:

            try:

                error_detail = (
                    response.json()
                    .get(
                        "detail",
                        "Unknown EDA error"
                    )
                )

            except Exception:

                error_detail = response.text

            st.error(
                f"EDA generation failed: "
                f"{error_detail}"
            )

            return False

        st.session_state.eda_report_bytes = (
            response.content
        )

        st.session_state.eda_generated = True

        return True

    except requests.exceptions.ConnectionError:

        st.error(
            "Could not connect to the EDA API."
        )

        return False

    except requests.exceptions.Timeout:

        st.error(
            "EDA generation timed out. "
            "The dataset may be too large."
        )

        return False

    except Exception as e:

        st.error(
            f"Unexpected EDA error: {str(e)}"
        )

        return False


# ==========================================================
# TITLE
# ==========================================================

st.title(
    "⚙️ Auto ML Preprocessor"
)

st.write(
    "Upload your dataset and automatically perform "
    "exploratory data analysis, preprocessing, "
    "feature engineering, scaling and feature selection."
)


# ==========================================================
# DATASET TYPE
# ==========================================================

st.subheader(
    "📂 What are you uploading?"
)

dataset_type = st.radio(
    "Select dataset type:",
    [
        "Entire Dataset",
        "Training Dataset",
        "Test Dataset"
    ],
    horizontal=True
)


# ==========================================================
# DATASET TYPE CHANGE
# ==========================================================

if (
    st.session_state.previous_dataset_type is not None
    and
    st.session_state.previous_dataset_type
    != dataset_type
):

    clear_results()


st.session_state.previous_dataset_type = (
    dataset_type
)


# ==========================================================
# INFORMATION
# ==========================================================

if dataset_type == "Entire Dataset":

    st.info(
        "Your complete dataset will be automatically "
        "split into training and testing sets. "
        "The preprocessing pipeline will be fitted only "
        "on the training portion and then applied to "
        "the test portion."
    )

elif dataset_type == "Training Dataset":

    st.info(
        "Your uploaded file will be treated entirely as "
        "training data. No train/test split will be performed."
    )

else:

    st.info(
        "To correctly process a test dataset, you must "
        "also provide its corresponding training dataset. "
        "The training data is required because preprocessing "
        "parameters such as missing-value replacements, "
        "categorical encodings, scaling parameters and "
        "selected features must be learned from the training "
        "data and then applied unchanged to the test data. "
        "This prevents data leakage."
    )


# ==========================================================
# FILE UPLOAD
# ==========================================================

if dataset_type == "Entire Dataset":

    uploaded_file = st.file_uploader(
        "📁 Upload your complete dataset",
        type=["csv"],
        key="entire_dataset_upload"
    )

    train_file = None
    test_file = None

elif dataset_type == "Training Dataset":

    uploaded_file = st.file_uploader(
        "📁 Upload your training dataset",
        type=["csv"],
        key="training_dataset_upload"
    )

    train_file = None
    test_file = None

else:

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


# ==========================================================
# SINGLE DATASET WORKFLOW
# ==========================================================

if (
    dataset_type in [
        "Entire Dataset",
        "Training Dataset"
    ]
    and uploaded_file is not None
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

    # ------------------------------------------------------
    # Dataset loaded
    # ------------------------------------------------------

    st.success(
        f"Dataset loaded successfully — "
        f"{df.shape[0]} rows × {df.shape[1]} columns"
    )

    # ------------------------------------------------------
    # Preview
    # ------------------------------------------------------

    st.subheader(
        "👀 Dataset Preview"
    )

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    # ------------------------------------------------------
    # Dataset statistics
    # ------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

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
            int(
                df.isnull()
                .sum()
                .sum()
            )
        )

    with col4:

        st.metric(
            "Duplicate Rows",
            int(
                df.duplicated()
                .sum()
            )
        )

    # ------------------------------------------------------
    # Target
    # ------------------------------------------------------

    st.subheader(
        "🎯 Target Selection"
    )

    target_column = st.selectbox(
        "Select Target Column",
        options=df.columns,
        index=len(df.columns) - 1,
        help=(
            "By default, the last column is selected "
            "as the target."
        ),
        key="single_dataset_target"
    )

    # ======================================================
    # QUICK EDA
    # ======================================================

    st.divider()

    st.subheader(
        "📊 Quick Exploratory Data Analysis"
    )

    st.write(
        "Five features are automatically selected "
        "for a quick analysis."
    )

    eda_features = select_eda_features(
        df,
        target_column
    )

    if eda_features:

        for feature in eda_features:

            with st.expander(
                f"🔎 Analyze: {feature}",
                expanded=False
            ):

                render_feature_analysis(
                    df,
                    feature
                )

    # ======================================================
    # FULL SWEETVIZ EDA
    # ======================================================

    st.divider()

    st.subheader(
        "📊 Full Automated EDA"
    )

    st.write(
        "Generate a complete interactive Sweetviz report "
        "for a deeper analysis of your dataset."
    )

    if st.button(
        "📊 Generate Full EDA Report",
        use_container_width=True,
        key="generate_full_eda_single"
    ):

        with st.spinner(
            "Generating Sweetviz report... "
            "This may take a little while."
        ):

            generate_full_eda(
                uploaded_file,
                target_column
            )

    if st.session_state.eda_generated:

        st.success(
            "✅ Full EDA report generated."
        )

        st.download_button(
            label="📄 Download Full EDA Report (HTML)",
            data=st.session_state.eda_report_bytes,
            file_name="eda_report.html",
            mime="text/html",
            use_container_width=True,
            key="download_eda_single"
        )

    # ======================================================
    # PROCESS DATASET
    # ======================================================

    st.divider()

    if st.button(
        "🚀 Process Dataset",
        use_container_width=True,
        key="process_single_dataset"
    ):

        with st.spinner(
            "Running preprocessing and feature selection..."
        ):

            try:

                uploaded_file.seek(0)

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
                        "dataset_type": dataset_type,
                        "target": target_column
                    },
                    timeout=300
                )

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

                # --------------------------------------------------
                # Store ZIP
                # --------------------------------------------------

                zip_bytes = response.content

                st.session_state.zip_bytes = (
                    zip_bytes
                )

                # --------------------------------------------------
                # Extract results
                # --------------------------------------------------

                with zipfile.ZipFile(
                    io.BytesIO(zip_bytes),
                    "r"
                ) as zip_file:

                    files_in_zip = (
                        zip_file.namelist()
                    )

                    if "X_train.csv" in files_in_zip:

                        st.session_state.x_train_bytes = (
                            zip_file.read(
                                "X_train.csv"
                            )
                        )

                    else:

                        st.session_state.x_train_bytes = None

                    if "X_test.csv" in files_in_zip:

                        st.session_state.x_test_bytes = (
                            zip_file.read(
                                "X_test.csv"
                            )
                        )

                    else:

                        st.session_state.x_test_bytes = None

                    if "pipeline_info.txt" in files_in_zip:

                        st.session_state.pipeline_info_bytes = (
                            zip_file.read(
                                "pipeline_info.txt"
                            )
                        )

                    else:

                        st.session_state.pipeline_info_bytes = None

                st.session_state.processed = True

                st.session_state.processed_target = (
                    target_column
                )

                st.session_state.processed_dataset_type = (
                    dataset_type
                )

                st.success(
                    "✅ Dataset processed successfully!"
                )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not connect to the preprocessing API."
                )

            except requests.exceptions.Timeout:

                st.error(
                    "The request timed out. "
                    "The backend may be waking up or "
                    "the dataset may be too large."
                )

            except Exception as e:

                st.error(
                    f"An unexpected error occurred: {str(e)}"
                )


# ==========================================================
# TEST DATASET WORKFLOW
# ==========================================================

if (
    dataset_type == "Test Dataset"
    and train_file is not None
    and test_file is not None
):

    # ------------------------------------------------------
    # Read train
    # ------------------------------------------------------

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

    # ------------------------------------------------------
    # Read test
    # ------------------------------------------------------

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

    # ------------------------------------------------------
    # Dataset information
    # ------------------------------------------------------

    st.success(
        f"Training dataset loaded — "
        f"{train_df.shape[0]} rows × "
        f"{train_df.shape[1]} columns"
    )

    st.success(
        f"Test dataset loaded — "
        f"{test_df.shape[0]} rows × "
        f"{test_df.shape[1]} columns"
    )

    # ------------------------------------------------------
    # Preview
    # ------------------------------------------------------

    train_preview_tab, test_preview_tab = st.tabs(
        [
            "👀 Training Dataset Preview",
            "👀 Test Dataset Preview"
        ]
    )

    with train_preview_tab:

        st.dataframe(
            train_df.head(20),
            use_container_width=True
        )

    with test_preview_tab:

        st.dataframe(
            test_df.head(20),
            use_container_width=True
        )

    # ------------------------------------------------------
    # Target
    # ------------------------------------------------------

    st.subheader(
        "🎯 Target Selection"
    )

    target_column = st.selectbox(
        "Select Target Column",
        options=train_df.columns,
        index=len(train_df.columns) - 1,
        help=(
            "The target is selected from the training "
            "dataset because the test dataset normally "
            "does not contain the target."
        ),
        key="test_dataset_target"
    )

    # ======================================================
    # QUICK EDA
    # ======================================================

    st.divider()

    st.subheader(
        "📊 Quick Exploratory Data Analysis"
    )

    st.write(
        "EDA is performed on the training dataset because "
        "it contains the target."
    )

    eda_features = select_eda_features(
        train_df,
        target_column
    )

    for feature in eda_features:

        with st.expander(
            f"🔎 Analyze: {feature}",
            expanded=False
        ):

            render_feature_analysis(
                train_df,
                feature
            )

    # ======================================================
    # FULL SWEETVIZ EDA
    # ======================================================

    st.divider()

    st.subheader(
        "📊 Full Automated EDA"
    )

    st.write(
        "The full EDA report is generated from the "
        "training dataset because the target belongs "
        "to the training data."
    )

    if st.button(
        "📊 Generate Full EDA Report",
        use_container_width=True,
        key="generate_full_eda_test"
    ):

        with st.spinner(
            "Generating Sweetviz report..."
        ):

            generate_full_eda(
                train_file,
                target_column
            )

    if st.session_state.eda_generated:

        st.success(
            "✅ Full EDA report generated."
        )

        st.download_button(
            label="📄 Download Full EDA Report (HTML)",
            data=st.session_state.eda_report_bytes,
            file_name="eda_report.html",
            mime="text/html",
            use_container_width=True,
            key="download_eda_test"
        )

    # ======================================================
    # PROCESS TEST
    # ======================================================

    st.divider()

    if st.button(
        "🚀 Process Test Dataset",
        use_container_width=True,
        key="process_test_dataset"
    ):

        with st.spinner(
            "Fitting preprocessing on training data "
            "and transforming test data..."
        ):

            try:

                train_file.seek(0)
                test_file.seek(0)

                response = requests.post(
                    API_URL,
                    files={
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
                    },
                    data={
                        "dataset_type": "Test Dataset",
                        "target": target_column
                    },
                    timeout=300
                )

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

                zip_bytes = response.content

                st.session_state.zip_bytes = (
                    zip_bytes
                )

                with zipfile.ZipFile(
                    io.BytesIO(zip_bytes),
                    "r"
                ) as zip_file:

                    files_in_zip = (
                        zip_file.namelist()
                    )

                    if "X_train.csv" in files_in_zip:

                        st.session_state.x_train_bytes = (
                            zip_file.read(
                                "X_train.csv"
                            )
                        )

                    else:

                        st.session_state.x_train_bytes = None

                    if "X_test.csv" in files_in_zip:

                        st.session_state.x_test_bytes = (
                            zip_file.read(
                                "X_test.csv"
                            )
                        )

                    else:

                        st.session_state.x_test_bytes = None

                    if "pipeline_info.txt" in files_in_zip:

                        st.session_state.pipeline_info_bytes = (
                            zip_file.read(
                                "pipeline_info.txt"
                            )
                        )

                    else:

                        st.session_state.pipeline_info_bytes = None

                st.session_state.processed = True

                st.session_state.processed_target = (
                    target_column
                )

                st.session_state.processed_dataset_type = (
                    "Test Dataset"
                )

                st.success(
                    "✅ Test dataset processed successfully!"
                )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not connect to the preprocessing API."
                )

            except requests.exceptions.Timeout:

                st.error(
                    "The request timed out."
                )

            except Exception as e:

                st.error(
                    f"An unexpected error occurred: {str(e)}"
                )


# ==========================================================
# DOWNLOAD RESULTS
# ==========================================================

if (
    st.session_state.processed
    and
    st.session_state.zip_bytes is not None
):

    st.divider()

    st.subheader(
        "📥 Download Results"
    )

    # ------------------------------------------------------
    # Individual files
    # ------------------------------------------------------

    download_items = []

    if st.session_state.x_train_bytes is not None:

        download_items.append(
            (
                "X_train.csv",
                st.session_state.x_train_bytes,
                "text/csv"
            )
        )

    if st.session_state.x_test_bytes is not None:

        download_items.append(
            (
                "X_test.csv",
                st.session_state.x_test_bytes,
                "text/csv"
            )
        )

    if st.session_state.pipeline_info_bytes is not None:

        download_items.append(
            (
                "pipeline_info.txt",
                st.session_state.pipeline_info_bytes,
                "text/plain"
            )
        )

    if download_items:

        columns = st.columns(
            len(download_items)
        )

        for column, item in zip(
            columns,
            download_items
        ):

            filename, data, mime = item

            with column:

                if filename == "pipeline_info.txt":

                    label = "📄 Pipeline Info"

                else:

                    label = f"⬇️ {filename}"

                st.download_button(
                    label=label,
                    data=data,
                    file_name=filename,
                    mime=mime,
                    use_container_width=True,
                    key=f"download_{filename}"
                )

    # ------------------------------------------------------
    # ZIP
    # ------------------------------------------------------

    st.download_button(
        label="📦 Download All Files (ZIP)",
        data=st.session_state.zip_bytes,
        file_name="processed_dataset.zip",
        mime="application/zip",
        use_container_width=True,
        key="download_all_files"
    )

    # ------------------------------------------------------
    # Information
    # ------------------------------------------------------

    if st.session_state.processed_target:

        st.info(
            f"Processed target: "
            f"**{st.session_state.processed_target}**"
        )


# ==========================================================
# OUTPUT PREVIEW
# ==========================================================

if st.session_state.processed:

    # ------------------------------------------------------
    # TRAIN PREVIEW
    # ------------------------------------------------------

    if st.session_state.x_train_bytes is not None:

        try:

            x_train_preview = pd.read_csv(
                io.BytesIO(
                    st.session_state.x_train_bytes
                )
            )

            st.divider()

            st.subheader(
                "🔍 Processed Dataset Preview"
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "Training Rows",
                    x_train_preview.shape[0]
                )

            with col2:

                if st.session_state.x_test_bytes is not None:

                    x_test_temp = pd.read_csv(
                        io.BytesIO(
                            st.session_state.x_test_bytes
                        )
                    )

                    st.metric(
                        "Test Rows",
                        x_test_temp.shape[0]
                    )

                else:

                    st.metric(
                        "Test Rows",
                        "N/A"
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

            st.write(
                "First 20 rows of the processed "
                "training dataset:"
            )

            st.dataframe(
                x_train_preview.head(20),
                use_container_width=True
            )

        except Exception:

            pass

    # ------------------------------------------------------
    # TEST PREVIEW
    # ------------------------------------------------------

    if st.session_state.x_test_bytes is not None:

        try:

            x_test_preview = pd.read_csv(
                io.BytesIO(
                    st.session_state.x_test_bytes
                )
            )

            st.subheader(
                "🔍 Processed Test Dataset Preview"
            )

            st.write(
                "First 20 rows of the processed "
                "test dataset:"
            )

            st.dataframe(
                x_test_preview.head(20),
                use_container_width=True
            )

        except Exception:

            pass