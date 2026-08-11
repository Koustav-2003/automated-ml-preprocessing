import streamlit as st
import pandas as pd
import requests
import io
import zipfile
import tempfile
import os
import textwrap
import sweetviz as sv
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

    # Sweetviz
    "eda_report_bytes": None,
    "eda_generated": False,

    # Locks
    "eda_running": False,
    "processing_running": False,

    # Metadata
    "processed_target": None,
    "processed_dataset_type": None,

    "previous_ml_task": None,
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

    st.session_state.processed_bytes = None

    st.session_state.pipeline_info_bytes = None

    st.session_state.eda_report_bytes = None
    st.session_state.eda_generated = False

    st.session_state.processed_target = None
    st.session_state.processed_dataset_type = None


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
    "feature engineering, scaling and feature selection — "
    "all in one workflow."
)

st.markdown(
    "**📁 Upload** → **📊 EDA** → **⚙️ Preprocess** → "
    "**🎯 Feature Selection** → **📥 Download**"
)

st.divider()



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

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Type",
            "Numerical"
        )

    with col2:

        st.metric(
            "Missing",
            f"{missing_count} ({missing_percentage:.2f}%)"
        )

    with col3:

        st.metric(
            "Unique Values",
            unique_count
        )

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

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Type",
            "Categorical"
        )

    with col2:

        st.metric(
            "Missing",
            f"{missing_count} ({missing_percentage:.2f}%)"
        )

    with col3:

        st.metric(
            "Unique Values",
            unique_count
        )

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

            "Category":
                value_counts.index,

            "Count":
                value_counts.values,

            "Percentage":
                (
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
# FEATURE ANALYSIS
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

    if target_is_numeric:

        col1, col2 = st.columns(2)

        with col1:

            st.write(
                "**Target Statistics**"
            )

            target_stats = pd.DataFrame({

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
                target_stats,
                hide_index=True,
                use_container_width=True
            )

        with col2:

            clean_target = (
                target_data.dropna()
            )

            if not clean_target.empty:

                fig = px.histogram(
                    clean_target,
                    x=target,
                    title=f"Target Distribution - {target}"
                )

                fig.update_layout(
                    height=400
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

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

            "Class":
                class_counts.index,

            "Count":
                class_counts.values,

            "Percentage":
                (
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
# FULL ON-SCREEN EDA
# ==========================================================

def render_full_eda(
    df,
    target_column=None,
    unsupervised=False
):

    st.divider()

    st.subheader(
        "📊 Exploratory Data Analysis"
    )

    if unsupervised:

        st.caption(
            "The on-screen EDA analyzes every feature "
            "without using a target variable."
        )

    else:

        st.caption(
            "The on-screen EDA analyzes every feature "
            "and provides a separate analysis of the target."
        )

        render_target_analysis(
            df,
            target_column
        )

    numerical_features = [

        column

        for column in df.columns

        if (
            pd.api.types.is_numeric_dtype(
                df[column]
            )

            and

            (
                unsupervised
                or
                column != target_column
            )
        )
    ]

    categorical_features = [

        column

        for column in df.columns

        if (
            not pd.api.types.is_numeric_dtype(
                df[column]
            )

            and

            (
                unsupervised
                or
                column != target_column
            )
        )
    ]

    # ======================================================
    # NUMERICAL FEATURES
    # ======================================================

    with st.expander(
        f"➕ Numerical Features ({len(numerical_features)})",
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
    # CATEGORICAL FEATURES
    # ======================================================

    with st.expander(
        f"➕ Categorical Features ({len(categorical_features)})",
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
# SWEETVIZ REPORT GENERATION
# ==========================================================

def generate_sweetviz_report(
    df,
    target=None
):

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

        if target is not None:

            report = sv.analyze(
                eda_df,
                target_feat=target,
                pairwise_analysis="off"
            )

        else:

            report = sv.analyze(
                eda_df,
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

            report_bytes = html_file.read()

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
# SWEETVIZ SECTION
# ==========================================================

def render_sweetviz_section(
    df,
    target_column=None,
    key_suffix="default",
    description=None
):

    st.divider()

    st.subheader(
        "📋 Full EDA Report"
    )

    if description:

        st.write(
            description
        )

    else:

        st.write(
            "The on-screen EDA above analyzes every feature. "
            "You can optionally generate a comprehensive "
            "interactive Sweetviz report."
        )

    st.warning(
        "⚠️ Full report generation can take around "
        "5 minutes depending on dataset size and complexity. "
        "The preprocessing button will be disabled while "
        "the report is being generated."
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

        st.session_state.eda_running = True

        try:

            with st.spinner(
                "Generating full Sweetviz report..."
            ):

                report_bytes = (
                    generate_sweetviz_report(
                        df,
                        target_column
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
                f"Sweetviz report generation failed: {str(e)}"
            )

        finally:

            st.session_state.eda_running = False

    if (
        st.session_state.eda_generated
        and
        st.session_state.eda_report_bytes is not None
    ):

        st.download_button(
            label="📄 Download Full EDA Report (HTML)",
            data=st.session_state.eda_report_bytes,
            file_name="eda_report.html",
            mime="text/html",
            use_container_width=True,
            key=f"download_sweetviz_{key_suffix}"
        )


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

    clear_results()

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

        clear_results()

    st.session_state.previous_dataset_type = dataset_type

    # ======================================================
    # ENTIRE DATASET
    # ======================================================

    if dataset_type == "Entire Dataset":

        st.info(
            "Your complete dataset will be automatically split "
            "into training and testing sets. The preprocessing "
            "pipeline will be fitted only on the training data."
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

        # ==================================================
        # EDA
        # ==================================================

        render_full_eda(
            df,
            target_column=target_column
        )

        # ==================================================
        # SWEETVIZ
        # ==================================================

        render_sweetviz_section(
            df,
            target_column,
            "supervised_single"
        )

        # ==================================================
        # PROCESSING
        # ==================================================

        st.divider()

        st.subheader(
            "⚙️ Automated Processing"
        )

        st.caption(
            "The pipeline will automatically preprocess the data, "
            "engineer features and perform feature selection."
        )

        processing_disabled = (
            st.session_state.eda_running
            or
            st.session_state.processing_running
        )

        if st.button(
            "🚀 Process Dataset",
            use_container_width=True,
            key="process_single_dataset",
            disabled=processing_disabled
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
                            "ml_task":
                                "Supervised Learning",

                            "dataset_type":
                                dataset_type,

                            "target":
                                target_column
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
                            f"Processing failed: {error_detail}"
                        )

                    else:

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
                    "The request timed out. The backend may "
                    "be waking up or the dataset may be too large."
                )

            except zipfile.BadZipFile:

                st.error(
                    "The API returned an invalid ZIP file."
                )

            except Exception as e:

                st.error(
                    f"An unexpected error occurred: {str(e)}"
                )

            finally:

                st.session_state.processing_running = False

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

        st.info(
            "ℹ️ EDA is performed on the **training dataset only**. "
            "The test dataset is kept unseen because it should "
            "not influence preprocessing or feature-selection decisions."
        )

        render_full_eda(
            train_df,
            target_column
        )

        render_sweetviz_section(
            train_df,
            target_column,
            "test",
            description=(
                "The on-screen EDA and Sweetviz report are generated "
                "using the **training dataset only**. The test dataset "
                "is kept separate and is only transformed after the "
                "preprocessing pipeline has been fitted on the "
                "training data."
            )
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
                            "ml_task":
                                "Supervised Learning",

                            "dataset_type":
                                "Test Dataset",

                            "target":
                                target_column
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
                            f"Processing failed: {error_detail}"
                        )

                    else:

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

            except zipfile.BadZipFile:

                st.error(
                    "The API returned an invalid ZIP file."
                )

            except Exception as e:

                st.error(
                    f"An unexpected error occurred: {str(e)}"
                )

            finally:

                st.session_state.processing_running = False


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

        clear_results()


    st.session_state.previous_unsupervised_dataset_type = (
        unsupervised_dataset_type
    )


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
            "into training and testing sets using an 80/20 split. "
            "The unsupervised preprocessing pipeline will be "
            "fitted only on the training data."
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


        # ==================================================
        # EDA
        # ==================================================

        render_full_eda(
            unsupervised_df,
            unsupervised=True
        )

        # ==================================================
        # SWEETVIZ
        # ==================================================

        render_sweetviz_section(
            unsupervised_df,
            target_column=None,
            key_suffix="unsupervised",
            description=(
                "The on-screen EDA above analyzes every feature. "
                "You can optionally generate a comprehensive "
                "target-free Sweetviz report."
            )
        )

        # ==================================================
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
            st.session_state.eda_running
            or
            st.session_state.processing_running
        )

        if st.button(
            "🚀 Process Unsupervised Dataset",
            use_container_width=True,
            key="process_unsupervised_dataset",
            disabled=processing_disabled
        ):

            st.session_state.processing_running = True

            try:

                with st.spinner(
                    "Running unsupervised preprocessing..."
                ):

                    # Build the multipart request according to the
                    # selected unsupervised workflow.
                    if (
                        unsupervised_dataset_type
                        == "Test Dataset"
                    ):

                        if (
                            unsupervised_train_file is None
                            or
                            unsupervised_test_file is None
                        ):

                            st.error(
                                "Please upload both the training "
                                "and test datasets."
                            )

                            st.stop()

                        unsupervised_train_file.seek(0)
                        unsupervised_test_file.seek(0)

                        request_files = {
                            "train_file": (
                                unsupervised_train_file.name,
                                unsupervised_train_file,
                                "text/csv"
                            ),

                            "test_file": (
                                unsupervised_test_file.name,
                                unsupervised_test_file,
                                "text/csv"
                            )
                        }

                    else:

                        if unsupervised_file is None:

                            st.error(
                                "Please upload a dataset first."
                            )

                            st.stop()

                        unsupervised_file.seek(0)

                        request_files = {
                            "file": (
                                unsupervised_file.name,
                                unsupervised_file,
                                "text/csv"
                            )
                        }

                    response = requests.post(
                        API_URL,

                        files=request_files,

                        data={
                            "ml_task":
                                "Unsupervised Learning",

                            "dataset_type":
                                unsupervised_dataset_type
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
                            f"Processing failed: {error_detail}"
                        )

                    else:

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

                            # The unsupervised pipeline now returns
                            # X_train.csv and X_test.csv, just like the
                            # supervised pipeline. Keep the same output
                            # handling for both learning types.

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

                            # Keep processed_bytes for backward compatibility
                            # with any existing session state, but the actual
                            # unsupervised outputs are X_train/X_test.
                            st.session_state.processed_bytes = None

                            if (
                                "pipeline_info.txt"
                                in files_in_zip
                            ):

                                st.session_state.pipeline_info_bytes = (
                                    zip_file.read(
                                        "pipeline_info.txt"
                                    )
                                )

                            else:

                                st.session_state.pipeline_info_bytes = None

                        st.session_state.processed = True

                        st.session_state.processed_target = None

                        st.session_state.processed_dataset_type = (
                            "Unsupervised Dataset"
                        )

                        st.success(
                            "✅ Unsupervised dataset processed successfully!"
                        )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not connect to the preprocessing API."
                )

            except requests.exceptions.Timeout:

                st.error(
                    "The request timed out."
                )

            except zipfile.BadZipFile:

                st.error(
                    "The API returned an invalid ZIP file."
                )

            except Exception as e:

                st.error(
                    f"An unexpected error occurred: {str(e)}"
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

                st.caption(
                    "Preview of the processed unsupervised "
                    "training and testing outputs."
                )

                train_rows = (
                    len(unsup_train_preview)
                    if unsup_train_preview is not None
                    else 0
                )

                test_rows = (
                    len(unsup_test_preview)
                    if unsup_test_preview is not None
                    else 0
                )

                reference_df = (
                    unsup_train_preview
                    if unsup_train_preview is not None
                    else unsup_test_preview
                )

                output_features = (
                    reference_df.shape[1]
                    if reference_df is not None
                    else 0
                )

                missing_values = (
                    int(
                        reference_df.isnull()
                        .sum()
                        .sum()
                    )
                    if reference_df is not None
                    else 0
                )

                col1, col2, col3, col4 = st.columns(4)

                with col1:

                    st.metric(
                        "Training Rows",
                        f"{train_rows:,}"
                    )

                with col2:

                    st.metric(
                        "Test Rows",
                        f"{test_rows:,}"
                    )

                with col3:

                    st.metric(
                        "Output Features",
                        f"{output_features:,}"
                    )

                with col4:

                    st.metric(
                        "Missing Values",
                        f"{missing_values:,}"
                    )

                train_tab, test_tab = st.tabs(
                    [
                        "X_train.csv",
                        "X_test.csv"
                    ]
                )

                with train_tab:

                    if unsup_train_preview is not None:

                        st.dataframe(
                            unsup_train_preview.head(20),
                            use_container_width=True
                        )

                    else:

                        st.info(
                            "No processed training dataset "
                            "was returned."
                        )

                with test_tab:

                    if unsup_test_preview is not None:

                        st.dataframe(
                            unsup_test_preview.head(20),
                            use_container_width=True
                        )

                    else:

                        st.info(
                            "No processed test dataset "
                            "was returned."
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
        ):

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

                st.caption(
                    "Preview of the processed training output."
                )

                col1, col2, col3, col4 = st.columns(4)

                with col1:

                    st.metric(
                        "Training Rows",
                        f"{x_train_preview.shape[0]:,}"
                    )

                with col2:

                    if (
                        st.session_state.x_test_bytes
                        is not None
                    ):

                        x_test_preview = pd.read_csv(
                            io.BytesIO(
                                st.session_state.x_test_bytes
                            )
                        )

                        st.metric(
                            "Test Rows",
                            f"{x_test_preview.shape[0]:,}"
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

                    if (
                        st.session_state.x_test_bytes
                        is not None
                    ):

                        x_test_preview = pd.read_csv(
                            io.BytesIO(
                                st.session_state.x_test_bytes
                            )
                        )

                        st.dataframe(
                            x_test_preview.head(20),
                            use_container_width=True
                        )

                    else:

                        st.info(
                            "No processed test dataset "
                            "was returned for this workflow."
                        )

            except Exception as e:

                st.warning(
                    f"Could not display processed output: {str(e)}"
                )


# ==========================================================
# FOOTER
# ==========================================================

st.caption(
    "Auto ML Preprocessor · Automated EDA · "
    "Feature Engineering · Feature Selection"
)
