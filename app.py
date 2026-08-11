import streamlit as st
import pandas as pd
import requests
import io
import zipfile
import sweetviz as sv
import tempfile
import os
import plotly.express as px


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

defaults = {
    "processed": False,

    "zip_bytes": None,
    "x_train_bytes": None,
    "x_test_bytes": None,
    "pipeline_info_bytes": None,

    "eda_report_bytes": None,
    "eda_generated": False,

    "eda_running": False,
    "processing_running": False,

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
# NUMERICAL FEATURE ANALYSIS
# ==========================================================

def render_numerical_analysis(df, feature):

    data = df[feature]

    missing_count = int(
        data.isnull().sum()
    )

    missing_percentage = (
        data.isnull().mean() * 100
    )

    unique_count = int(
        data.nunique()
    )

    # ------------------------------------------------------
    # Metrics
    # ------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Type",
            "Numerical"
        )

    with col2:

        st.metric(
            "Missing",
            f"{missing_count} "
            f"({missing_percentage:.2f}%)"
        )

    with col3:

        st.metric(
            "Unique Values",
            unique_count
        )

    with col4:

        mean_value = data.mean()

        mean_text = (
            "N/A"
            if pd.isna(mean_value)
            else f"{mean_value:.3f}"
        )

        st.metric(
            "Mean",
            mean_text
        )

    # ------------------------------------------------------
    # Statistics
    # ------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

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
    # Distribution
    # ------------------------------------------------------

    with col2:

        st.write("**Distribution**")

        clean_data = data.dropna()

        if not clean_data.empty:

            distribution = (
                clean_data
                .value_counts()
                .sort_index()
            )

            st.bar_chart(
                distribution
            )

        else:

            st.info(
                "No values available."
            )

    # ------------------------------------------------------
    # Proper Box Plot
    # ------------------------------------------------------

    st.write("**Box Plot**")

    clean_data = data.dropna()

    if not clean_data.empty:

        fig = px.box(
            clean_data,
            y=feature,
            points="outliers",
            title=f"Box Plot - {feature}"
        )

        fig.update_layout(
            height=400,
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.info(
            "No values available for box plot."
        )


# ==========================================================
# CATEGORICAL FEATURE ANALYSIS
# ==========================================================

def render_categorical_analysis(df, feature):

    data = df[feature]

    missing_count = int(
        data.isnull().sum()
    )

    missing_percentage = (
        data.isnull().mean() * 100
    )

    unique_count = int(
        data.nunique()
    )

    # ------------------------------------------------------
    # Metrics
    # ------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Type",
            "Categorical"
        )

    with col2:

        st.metric(
            "Missing",
            f"{missing_count} "
            f"({missing_percentage:.2f}%)"
        )

    with col3:

        st.metric(
            "Unique Values",
            unique_count
        )

    # ------------------------------------------------------
    # Category Distribution
    # ------------------------------------------------------

    st.write(
        "**Category Distribution**"
    )

    value_counts = (
        data
        .fillna("Missing")
        .astype(str)
        .value_counts()
        .head(15)
    )

    if not value_counts.empty:

        st.bar_chart(
            value_counts
        )

        category_table = pd.DataFrame({
            "Category": value_counts.index,
            "Count": value_counts.values,
            "Percentage": (
                value_counts.values
                / len(data)
                * 100
            ).round(2)
        })

        st.dataframe(
            category_table,
            hide_index=True,
            use_container_width=True
        )

    else:

        st.info(
            "No categorical values available."
        )


# ==========================================================
# FEATURE ANALYSIS DISPATCHER
# ==========================================================

def render_feature_analysis(df, feature):

    if pd.api.types.is_numeric_dtype(
        df[feature]
    ):

        render_numerical_analysis(
            df,
            feature
        )

    else:

        render_categorical_analysis(
            df,
            feature
        )


# ==========================================================
# TARGET ANALYSIS
# ==========================================================

def render_target_analysis(df, target):

    st.subheader(
        "🎯 Target Analysis"
    )

    target_data = df[target]

    target_is_numeric = (
        pd.api.types.is_numeric_dtype(
            target_data
        )
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Target Type",
            "Numerical"
            if target_is_numeric
            else "Categorical"
        )

    with col2:

        st.metric(
            "Missing",
            int(
                target_data.isnull().sum()
            )
        )

    with col3:

        st.metric(
            "Unique Values",
            int(
                target_data.nunique()
            )
        )

    with col4:

        st.metric(
            "Rows",
            len(target_data)
        )

    # ======================================================
    # NUMERICAL TARGET
    # ======================================================

    if target_is_numeric:

        col1, col2 = st.columns(2)

        with col1:

            st.write(
                "**Target Statistics**"
            )

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
                    target_data.mean(),
                    target_data.median(),
                    target_data.std(),
                    target_data.min(),
                    target_data.max(),
                    target_data.skew()
                ]
            })

            st.dataframe(
                stats,
                hide_index=True,
                use_container_width=True
            )

        with col2:

            st.write(
                "**Target Distribution**"
            )

            clean_target = (
                target_data.dropna()
            )

            if not clean_target.empty:

                st.bar_chart(
                    clean_target
                    .value_counts()
                    .sort_index()
                )

            else:

                st.info(
                    "No target values available."
                )

        # --------------------------------------------------
        # Target Box Plot
        # --------------------------------------------------

        st.write(
            "**Target Box Plot**"
        )

        clean_target = (
            target_data.dropna()
        )

        if not clean_target.empty:

            fig = px.box(
                clean_target,
                y=target,
                points="outliers",
                title=f"Box Plot - {target}"
            )

            fig.update_layout(
                height=400,
                showlegend=False
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # ======================================================
    # CATEGORICAL TARGET
    # ======================================================

    else:

        st.write(
            "**Class Distribution**"
        )

        class_counts = (
            target_data
            .fillna("Missing")
            .astype(str)
            .value_counts()
        )

        class_table = pd.DataFrame({
            "Class": class_counts.index,
            "Count": class_counts.values,
            "Percentage": (
                class_counts.values
                / len(target_data)
                * 100
            ).round(2)
        })

        st.dataframe(
            class_table,
            hide_index=True,
            use_container_width=True
        )

        st.bar_chart(
            class_counts
        )


# ==========================================================
# FULL SWEETVIZ REPORT
# ==========================================================

def generate_sweetviz_report(df, target):

    temp_path = None

    try:

        MAX_EDA_ROWS = 5000

        if len(df) > MAX_EDA_ROWS:

            eda_df = df.sample(
                n=MAX_EDA_ROWS,
                random_state=42
            )

        else:

            eda_df = df.copy()

        temp_file = tempfile.NamedTemporaryFile(
            suffix=".html",
            delete=False
        )

        temp_path = temp_file.name

        temp_file.close()

        report = sv.analyze(
            eda_df,
            target_feat=target,
            pairwise_analysis="off"
        )

        report.show_html(
            filepath=temp_path,
            open_browser=False,
            layout="widescreen"
        )

        with open(
            temp_path,
            "rb"
        ) as html_file:

            report_bytes = (
                html_file.read()
            )

        return report_bytes

    finally:

        if (
            temp_path is not None
            and
            os.path.exists(temp_path)
        ):

            try:

                os.remove(
                    temp_path
                )

            except Exception:

                pass


# ==========================================================
# RUN SWEETVIZ
# ==========================================================

def run_sweetviz(df, target):

    st.session_state.eda_running = True

    try:

        with st.spinner(
            "Generating full Sweetviz report..."
        ):

            report_bytes = (
                generate_sweetviz_report(
                    df,
                    target
                )
            )

        st.session_state.eda_report_bytes = (
            report_bytes
        )

        st.session_state.eda_generated = True

        st.success(
            "✅ Full EDA report generated."
        )

    except Exception as e:

        st.error(
            f"Sweetviz report generation failed: "
            f"{str(e)}"
        )

    finally:

        st.session_state.eda_running = False


# ==========================================================
# ON-SCREEN EDA
# ==========================================================

def render_full_eda(df, target_column):

    st.divider()

    st.subheader(
        "📊 Exploratory Data Analysis"
    )

    # ------------------------------------------------------
    # Target
    # ------------------------------------------------------

    render_target_analysis(
        df,
        target_column
    )

    # ------------------------------------------------------
    # Feature Types
    # ------------------------------------------------------

    numerical_features = [
        column
        for column in df.columns
        if pd.api.types.is_numeric_dtype(
            df[column]
        )
        and column != target_column
    ]

    categorical_features = [
        column
        for column in df.columns
        if not pd.api.types.is_numeric_dtype(
            df[column]
        )
        and column != target_column
    ]

    # ======================================================
    # NUMERICAL
    # ======================================================

    with st.expander(
        f"➕ Numerical Features "
        f"({len(numerical_features)})",
        expanded=False
    ):

        if numerical_features:

            for feature in numerical_features:

                with st.expander(
                    f"🔎 {feature}",
                    expanded=False
                ):

                    render_feature_analysis(
                        df,
                        feature
                    )

        else:

            st.info(
                "No numerical features available."
            )

    # ======================================================
    # CATEGORICAL
    # ======================================================

    with st.expander(
        f"➕ Categorical Features "
        f"({len(categorical_features)})",
        expanded=False
    ):

        if categorical_features:

            for feature in categorical_features:

                with st.expander(
                    f"🔎 {feature}",
                    expanded=False
                ):

                    render_feature_analysis(
                        df,
                        feature
                    )

        else:

            st.info(
                "No categorical features available."
            )


# ==========================================================
# SWEETVIZ SECTION
# ==========================================================

def render_sweetviz_section(
    df,
    target_column,
    key_suffix
):

    st.divider()

    st.subheader(
        "📋 Full EDA Report"
    )

    st.write(
        "The on-screen EDA above analyzes every feature. "
        "You can optionally generate a comprehensive "
        "interactive Sweetviz report."
    )

    st.warning(
        "⚠️ Full report generation can take around "
        "5 minutes depending on the size and complexity "
        "of your dataset. Preprocessing will be disabled "
        "while the report is being generated."
    )

    generate_disabled = (
        st.session_state.eda_running
        or
        st.session_state.processing_running
    )

    if st.button(
        "📊 Generate Full Sweetviz Report",
        use_container_width=True,
        key=f"generate_sweetviz_{key_suffix}",
        disabled=generate_disabled
    ):

        run_sweetviz(
            df,
            target_column
        )

    if (
        st.session_state.eda_generated
        and
        st.session_state.eda_report_bytes is not None
    ):

        st.download_button(
            label=(
                "📄 Download Full EDA Report (HTML)"
            ),
            data=st.session_state.eda_report_bytes,
            file_name="eda_report.html",
            mime="text/html",
            use_container_width=True,
            key=f"download_sweetviz_{key_suffix}"
        )


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
    st.session_state.previous_dataset_type != dataset_type
):

    clear_results()

st.session_state.previous_dataset_type = (
    dataset_type
)


# ==========================================================
# DATASET TYPE INFORMATION
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
        "Your uploaded file will be treated entirely "
        "as training data. No train/test split will "
        "be performed."
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
    # Overview
    # ------------------------------------------------------

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
            df.shape[0]
        )

    with col2:

        st.metric(
            "Columns",
            df.shape[1]
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
            int(
                df.isnull()
                .sum()
                .sum()
            )
        )

    st.write(
        f"**Duplicate Rows:** "
        f"{int(df.duplicated().sum())}"
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
            "The last column is selected as the "
            "default target."
        ),
        key="single_dataset_target"
    )

    # ------------------------------------------------------
    # EDA
    # ------------------------------------------------------

    render_full_eda(
        df,
        target_column
    )

    # ------------------------------------------------------
    # Sweetviz
    # ------------------------------------------------------

    render_sweetviz_section(
        df,
        target_column,
        "single"
    )

    # ======================================================
    # PROCESS DATASET
    # ======================================================

    st.divider()

    process_disabled = (
        st.session_state.eda_running
        or
        st.session_state.processing_running
    )

    if st.button(
        "🚀 Process Dataset",
        use_container_width=True,
        key="process_single_dataset",
        disabled=process_disabled
    ):

        st.session_state.processing_running = True

        try:

            with st.spinner(
                "Running preprocessing and feature selection..."
            ):

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

                st.session_state.zip_bytes = (
                    response.content
                )

                # --------------------------------------------------
                # Extract files
                # --------------------------------------------------

                with zipfile.ZipFile(
                    io.BytesIO(
                        response.content
                    ),
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
                "The request timed out."
            )

        except Exception as e:

            st.error(
                f"An unexpected error occurred: "
                f"{str(e)}"
            )

        finally:

            st.session_state.processing_running = False


# ==========================================================
# TEST DATASET WORKFLOW
# ==========================================================

if (
    dataset_type == "Test Dataset"
    and
    train_file is not None
    and
    test_file is not None
):

    # ------------------------------------------------------
    # Read training
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

    train_tab, test_tab = st.tabs(
        [
            "👀 Training Dataset Preview",
            "👀 Test Dataset Preview"
        ]
    )

    with train_tab:

        st.dataframe(
            train_df.head(20),
            use_container_width=True
        )

    with test_tab:

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
            "The target is selected from the "
            "training dataset."
        ),
        key="test_dataset_target"
    )

    # ------------------------------------------------------
    # EDA
    # ------------------------------------------------------

    render_full_eda(
        train_df,
        target_column
    )

    # ------------------------------------------------------
    # Sweetviz
    # ------------------------------------------------------

    render_sweetviz_section(
        train_df,
        target_column,
        "test"
    )

    # ======================================================
    # PROCESS TEST DATASET
    # ======================================================

    st.divider()

    process_disabled = (
        st.session_state.eda_running
        or
        st.session_state.processing_running
    )

    if st.button(
        "🚀 Process Test Dataset",
        use_container_width=True,
        key="process_test_dataset",
        disabled=process_disabled
    ):

        st.session_state.processing_running = True

        try:

            with st.spinner(
                "Fitting preprocessing on training data "
                "and transforming test data..."
            ):

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

                st.session_state.zip_bytes = (
                    response.content
                )

                with zipfile.ZipFile(
                    io.BytesIO(
                        response.content
                    ),
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
                f"An unexpected error occurred: "
                f"{str(e)}"
            )

        finally:

            st.session_state.processing_running = False


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

    st.download_button(
        label="📦 Download All Files (ZIP)",
        data=st.session_state.zip_bytes,
        file_name="processed_dataset.zip",
        mime="application/zip",
        use_container_width=True,
        key="download_all_files"
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

    # ------------------------------------------------------
    # TRAINING PREVIEW
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