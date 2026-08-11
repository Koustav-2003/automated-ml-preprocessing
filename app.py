import streamlit as st
import pandas as pd
import requests
import io
import zipfile
import sweetviz as sv
import tempfile
import os
import plotly.express as px
import textwrap


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
# HTML RENDER HELPER
# ==========================================================

def render_html(html):

    if hasattr(st, "html"):

        st.html(html)

    else:

        st.markdown(
            html,
            unsafe_allow_html=True
        )


# ==========================================================
# CUSTOM UI / CSS
# ==========================================================

render_html(
    """
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

h1, h2, h3 {
    letter-spacing: -0.5px;
}


/* =====================================================
   HERO
   ===================================================== */

.hero {
    padding: 2.2rem 2.5rem;
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
    margin-bottom: 0.35rem;
    line-height: 1.15;
}

.hero-subtitle {
    font-size: 1.18rem;
    font-weight: 600;
    margin-bottom: 0.65rem;
}

.hero-description {
    font-size: 1rem;
    opacity: 0.78;
    max-width: 850px;
    line-height: 1.6;
    margin-bottom: 1.2rem;
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


/* =====================================================
   SECTION CARDS
   ===================================================== */

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
}


/* =====================================================
   DOWNLOAD CARD
   ===================================================== */

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


/* =====================================================
   FOOTER
   ===================================================== */

.footer {
    text-align: center;
    opacity: 0.5;
    font-size: 0.85rem;
    padding-top: 2rem;
}


/* =====================================================
   BUTTONS
   ===================================================== */

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


/* =====================================================
   FILE UPLOADER
   ===================================================== */

[data-testid="stFileUploader"] {
    border-radius: 14px;
}


/* =====================================================
   METRICS
   ===================================================== */

[data-testid="stMetric"] {
    padding: 0.8rem;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,0.15);
}


/* =====================================================
   EXPANDERS
   ===================================================== */

[data-testid="stExpander"] {
    border-radius: 10px;
}

</style>
"""
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
    "previous_ml_task": None
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
# HERO
# ==========================================================

render_html(
    """
<div class="hero">

    <div class="hero-title">
        ⚙️ Auto ML Preprocessor
    </div>

    <div class="hero-subtitle">
        No more manual EDA. No more repetitive preprocessing.
    </div>

    <div class="hero-description">
        Upload your dataset and automatically perform
        exploratory data analysis, preprocessing,
        feature engineering, scaling and feature selection —
        all in one workflow.
    </div>

    <div class="workflow">

        <div class="workflow-step">
            📁 Upload
        </div>

        <div class="workflow-arrow">
            →
        </div>

        <div class="workflow-step">
            📊 Analyze
        </div>

        <div class="workflow-arrow">
            →
        </div>

        <div class="workflow-step">
            ⚙️ Process
        </div>

        <div class="workflow-arrow">
            →
        </div>

        <div class="workflow-step">
            ✨ Engineer
        </div>

        <div class="workflow-arrow">
            →
        </div>

        <div class="workflow-step">
            📥 Download
        </div>

    </div>

</div>
"""
)


# ==========================================================
# ML TASK SELECTION
# ==========================================================

render_html(
    """
<div class="section-card">

    <div class="section-card-title">
        🧠 What type of ML problem are you preparing for?
    </div>

    <div class="section-card-description">
        Choose whether your dataset has a target variable
        or will be used for an unsupervised learning task.
    </div>

</div>
"""
)


ml_task = st.radio(
    "Select ML task:",
    [
        "Supervised Learning",
        "Unsupervised Learning"
    ],
    horizontal=True,
    key="ml_task_selection"
)


# ==========================================================
# CLEAR RESULTS WHEN ML TASK CHANGES
# ==========================================================

if (
    st.session_state.previous_ml_task is not None
    and
    st.session_state.previous_ml_task != ml_task
):

    clear_results()

    st.session_state.previous_dataset_type = None


st.session_state.previous_ml_task = ml_task


# ==========================================================
# ==========================================================
# SUPERVISED LEARNING
# ==========================================================
# ==========================================================

if ml_task == "Supervised Learning":

    # ======================================================
    # DATASET WORKFLOW CARD
    # ======================================================

    render_html(
        """
<div class="section-card">

    <div class="section-card-title">
        📂 Choose your dataset workflow
    </div>

    <div class="section-card-description">
        Tell the application how your data is currently
        structured so it can apply the correct workflow.
    </div>

</div>
"""
    )


    # ======================================================
    # DATASET TYPE
    # ======================================================

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


    # ======================================================
    # DATASET TYPE CHANGE
    # ======================================================

    if (
        st.session_state.previous_dataset_type is not None
        and
        st.session_state.previous_dataset_type != dataset_type
    ):

        clear_results()


    st.session_state.previous_dataset_type = dataset_type


    # ======================================================
    # DATASET TYPE INFORMATION
    # ======================================================

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


    # ======================================================
    # FILE UPLOAD
    # ======================================================

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


        # ==================================================
        # DATASET PREVIEW
        # ==================================================

        st.subheader(
            "👀 Dataset Preview"
        )

        st.caption(
            "Showing the first 20 rows."
        )

        st.dataframe(
            df.head(20),
            use_container_width=True
        )


        # ==================================================
        # DATASET INFORMATION
        # ==================================================

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
        # TARGET SELECTION
        # ==================================================

        st.subheader(
            "🎯 Target Selection"
        )

        st.caption(
            "Select the variable you want the supervised "
            "learning pipeline to predict."
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


        # ==================================================
        # TARGET ANALYSIS
        # ==================================================

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


        # ==================================================
        # FEATURE ANALYSIS
        # ==================================================

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

            col1, col2, col3, col4 = (
                st.columns(4)
            )

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


            col1, col2 = st.columns(2)


            with col1:

                st.write(
                    "**Summary Statistics**"
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

                st.write(
                    "**Distribution**"
                )

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


            st.write(
                "**Box Plot**"
            )

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
                    f"{missing_count} "
                    f"({missing_percentage:.2f}%)"
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


        # ==================================================
        # EDA
        # ==================================================

        render_target_analysis(
            df,
            target_column
        )


        st.divider()

        st.subheader(
            "📊 Exploratory Data Analysis"
        )

        st.caption(
            "Explore your dataset feature-by-feature before "
            "running the automated preprocessing pipeline."
        )


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


        with st.expander(
            f"➕ Numerical Features "
            f"({len(numerical_features)})",
            expanded=False
        ):

            for feature in numerical_features:

                with st.expander(
                    f"🔎 {feature}",
                    expanded=False
                ):

                    render_feature_analysis(
                        df,
                        feature
                    )


        with st.expander(
            f"➕ Categorical Features "
            f"({len(categorical_features)})",
            expanded=False
        ):

            for feature in categorical_features:

                with st.expander(
                    f"🔎 {feature}",
                    expanded=False
                ):

                    render_feature_analysis(
                        df,
                        feature
                    )


        # ==================================================
        # SWEETVIZ
        # ==================================================

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


        if st.button(
            "📊 Generate Full Sweetviz Report",
            use_container_width=True,
            key="generate_sweetviz_single",
            disabled=(
                st.session_state.eda_running
                or
                st.session_state.processing_running
            )
        ):

            st.session_state.eda_running = True

            try:

                with st.spinner(
                    "Generating full Sweetviz report..."
                ):

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
                        target_feat=target_column,
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

                        st.session_state.eda_report_bytes = (
                            html_file.read()
                        )


                    st.session_state.eda_generated = True

                    st.success(
                        "✅ Full EDA report generated."
                    )


                if os.path.exists(temp_path):

                    os.remove(
                        temp_path
                    )

            except Exception as e:

                st.error(
                    f"Sweetviz report generation failed: "
                    f"{str(e)}"
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
                key="download_sweetviz_single"
            )


        # ==================================================
        # PROCESS DATASET
        # ==================================================

        st.divider()

        st.subheader(
            "⚙️ Automated Processing"
        )

        st.caption(
            "The pipeline will automatically detect the task, "
            "split the data when required, fit preprocessing "
            "on training data, engineer features and perform "
            "feature selection."
        )


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
                            "dataset_type":
                                dataset_type,

                            "target":
                                target_column,

                            "ml_task":
                                "Supervised Learning"
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


    # ======================================================
    # SEPARATE TRAIN + TEST WORKFLOW
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

            test_file.seek(0)

            test_df = pd.read_csv(
                test_file
            )

        except Exception as e:

            st.error(
                f"Could not read dataset files: {str(e)}"
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
            "The test dataset is kept unseen because it should not "
            "influence preprocessing or feature-selection decisions."
        )


        # --------------------------------------------------
        # Simple EDA on training data
        # --------------------------------------------------

        st.divider()

        st.subheader(
            "📊 Exploratory Data Analysis"
        )

        st.caption(
            "EDA is performed on the training dataset only."
        )


        numerical_features = [
            column
            for column in train_df.columns
            if pd.api.types.is_numeric_dtype(
                train_df[column]
            )
            and column != target_column
        ]


        categorical_features = [
            column
            for column in train_df.columns
            if not pd.api.types.is_numeric_dtype(
                train_df[column]
            )
            and column != target_column
        ]


        with st.expander(
            f"➕ Numerical Features "
            f"({len(numerical_features)})"
        ):

            for feature in numerical_features:

                with st.expander(
                    f"🔎 {feature}"
                ):

                    data = train_df[feature]

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(
                            "Missing",
                            int(
                                data.isnull().sum()
                            )
                        )

                    with col2:

                        st.metric(
                            "Unique",
                            int(
                                data.nunique()
                            )
                        )

                    with col3:

                        mean_value = data.mean()

                        st.metric(
                            "Mean",
                            "N/A"
                            if pd.isna(mean_value)
                            else f"{mean_value:.3f}"
                        )


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


        with st.expander(
            f"➕ Categorical Features "
            f"({len(categorical_features)})"
        ):

            for feature in categorical_features:

                with st.expander(
                    f"🔎 {feature}"
                ):

                    data = train_df[feature]

                    st.metric(
                        "Unique Values",
                        int(
                            data.nunique()
                        )
                    )

                    st.bar_chart(
                        data
                        .fillna("Missing")
                        .astype(str)
                        .value_counts()
                        .head(15)
                    )


        # --------------------------------------------------
        # Sweetviz
        # --------------------------------------------------

        st.divider()

        st.subheader(
            "📋 Full EDA Report"
        )

        st.write(
            "The on-screen EDA and Sweetviz report are "
            "**generated using the training dataset only**."
        )

        st.warning(
            "⚠️ Full report generation can take around "
            "5 minutes. Preprocessing will be disabled "
            "while the report is being generated."
        )


        if st.button(
            "📊 Generate Full Sweetviz Report",
            use_container_width=True,
            key="generate_sweetviz_train_test",
            disabled=(
                st.session_state.eda_running
                or
                st.session_state.processing_running
            )
        ):

            st.session_state.eda_running = True

            try:

                with st.spinner(
                    "Generating full Sweetviz report..."
                ):

                    eda_df = train_df.copy()

                    if len(eda_df) > 5000:

                        eda_df = eda_df.sample(
                            n=5000,
                            random_state=42
                        )


                    temp_file = tempfile.NamedTemporaryFile(
                        suffix=".html",
                        delete=False
                    )

                    temp_path = temp_file.name

                    temp_file.close()


                    report = sv.analyze(
                        eda_df,
                        target_feat=target_column,
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

                        st.session_state.eda_report_bytes = (
                            html_file.read()
                        )


                    st.session_state.eda_generated = True

                    st.success(
                        "✅ Full EDA report generated."
                    )


                if os.path.exists(temp_path):

                    os.remove(
                        temp_path
                    )


            except Exception as e:

                st.error(
                    f"Sweetviz report generation failed: "
                    f"{str(e)}"
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
                key="download_sweetviz_train_test"
            )


        # --------------------------------------------------
        # Process
        # --------------------------------------------------

        st.divider()

        st.subheader(
            "⚙️ Automated Processing"
        )

        st.caption(
            "The preprocessing pipeline will be fitted on the "
            "training dataset and the learned parameters will "
            "then be applied to the test dataset."
        )


        if st.button(
            "🚀 Process Test Dataset",
            use_container_width=True,
            key="process_test_dataset",
            disabled=(
                st.session_state.eda_running
                or
                st.session_state.processing_running
            )
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

                            "dataset_type":
                                "Test Dataset",

                            "target":
                                target_column,

                            "ml_task":
                                "Supervised Learning"
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
# ==========================================================
# UNSUPERVISED LEARNING
# ==========================================================
# ==========================================================

else:

    st.info(
        "🔬 **Unsupervised Learning:** "
        "No target variable is required. The dataset will be "
        "prepared using only the feature columns."
    )

    render_html(
        """
<div class="section-card">

    <div class="section-card-title">
        🔬 Unsupervised Dataset
    </div>

    <div class="section-card-description">
        Upload a dataset without selecting a target variable.
        The pipeline will automatically handle missing values,
        categorical features, skewness, scaling and other
        preprocessing steps required for unsupervised learning.
    </div>

</div>
"""
    )

    unsupervised_file = st.file_uploader(
        "📁 Upload your dataset",
        type=["csv"],
        key="unsupervised_dataset_upload"
    )

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

        # ==================================================
        # DATASET PREVIEW
        # ==================================================

        st.subheader(
            "👀 Dataset Preview"
        )

        st.caption(
            "Showing the first 20 rows."
        )

        st.dataframe(
            unsupervised_df.head(20),
            use_container_width=True
        )

        # ==================================================
        # DATASET INFORMATION
        # ==================================================

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

        st.divider()

        st.subheader(
            "📊 Exploratory Data Analysis"
        )

        st.caption(
            "EDA is performed across all features without "
            "using a target variable."
        )

        # ==================================================
        # NUMERICAL FEATURES
        # ==================================================

        with st.expander(
            f"➕ Numerical Features "
            f"({len(numerical_features)})",
            expanded=False
        ):

            for feature in numerical_features:

                with st.expander(
                    f"🔎 {feature}",
                    expanded=False
                ):

                    data = unsupervised_df[feature]

                    missing_count = int(
                        data.isnull().sum()
                    )

                    missing_percentage = (
                        data.isnull().mean() * 100
                    )

                    unique_count = int(
                        data.nunique()
                    )

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:

                        st.metric(
                            "Type",
                            "Numerical"
                        )

                    with col2:

                        st.metric(
                            "Missing",
                            f"{missing_count}"
                        )

                    with col3:

                        st.metric(
                            "Missing %",
                            f"{missing_percentage:.2f}%"
                        )

                    with col4:

                        st.metric(
                            "Unique Values",
                            unique_count
                        )

                    col1, col2 = st.columns(2)

                    with col1:

                        st.write(
                            "**Summary Statistics**"
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

                        st.write(
                            "**Distribution**"
                        )

                        clean_data = data.dropna()

                        if not clean_data.empty:

                            st.bar_chart(
                                clean_data
                                .value_counts()
                                .sort_index()
                            )

                        else:

                            st.info(
                                "No values available."
                            )

                    st.write(
                        "**Box Plot**"
                    )

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

        # ==================================================
        # CATEGORICAL FEATURES
        # ==================================================

        with st.expander(
            f"➕ Categorical Features "
            f"({len(categorical_features)})",
            expanded=False
        ):

            for feature in categorical_features:

                with st.expander(
                    f"🔎 {feature}",
                    expanded=False
                ):

                    data = unsupervised_df[feature]

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
                            f"{missing_count}"
                        )

                    with col3:

                        st.metric(
                            "Missing %",
                            f"{missing_percentage:.2f}%"
                        )

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

        # ==================================================
        # SWEETVIZ
        # ==================================================

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
            "of your dataset. Processing will be disabled "
            "while the report is being generated."
        )

        if st.button(
            "📊 Generate Full Sweetviz Report",
            use_container_width=True,
            key="generate_sweetviz_unsupervised",
            disabled=(
                st.session_state.eda_running
                or
                st.session_state.processing_running
            )
        ):

            st.session_state.eda_running = True

            try:

                with st.spinner(
                    "Generating full Sweetviz report..."
                ):

                    MAX_EDA_ROWS = 5000

                    if len(unsupervised_df) > MAX_EDA_ROWS:

                        eda_df = unsupervised_df.sample(
                            n=MAX_EDA_ROWS,
                            random_state=42
                        )

                    else:

                        eda_df = unsupervised_df.copy()

                    temp_file = tempfile.NamedTemporaryFile(
                        suffix=".html",
                        delete=False
                    )

                    temp_path = temp_file.name

                    temp_file.close()

                    # IMPORTANT:
                    # No target_feat for unsupervised learning.

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

                        st.session_state.eda_report_bytes = (
                            html_file.read()
                        )

                    st.session_state.eda_generated = True

                    st.success(
                        "✅ Full unsupervised EDA report generated."
                    )

                if os.path.exists(temp_path):

                    os.remove(
                        temp_path
                    )

            except Exception as e:

                st.error(
                    f"Sweetviz report generation failed: "
                    f"{str(e)}"
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
                file_name="unsupervised_eda_report.html",
                mime="text/html",
                use_container_width=True,
                key="download_sweetviz_unsupervised"
            )

        # ==================================================
        # UNSUPERVISED PROCESSING
        # ==================================================

        st.divider()

        st.subheader(
            "⚙️ Unsupervised Processing"
        )

        st.caption(
            "No target variable is used. The pipeline will "
            "detect ID-like columns, handle missing values, "
            "process categorical variables, transform skewed "
            "features and scale the resulting feature matrix."
        )

        if st.button(
            "🚀 Process Unsupervised Dataset",
            use_container_width=True,
            key="process_unsupervised_dataset",
            disabled=(
                st.session_state.eda_running
                or
                st.session_state.processing_running
            )
        ):

            st.session_state.processing_running = True

            try:

                with st.spinner(
                    "Running unsupervised preprocessing..."
                ):

                    unsupervised_file.seek(0)

                    response = requests.post(

                        API_URL,

                        files={
                            "file": (
                                unsupervised_file.name,
                                unsupervised_file,
                                "text/csv"
                            )
                        },

                        data={
                            "ml_task":
                                "Unsupervised Learning",

                            "dataset_type":
                                "Unsupervised Dataset"
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

                    # --------------------------------------
                    # Store ZIP
                    # --------------------------------------

                    st.session_state.zip_bytes = (
                        response.content
                    )

                    # --------------------------------------
                    # Read processed output
                    # --------------------------------------

                    with zipfile.ZipFile(
                        io.BytesIO(
                            response.content
                        ),
                        "r"
                    ) as zip_file:

                        files_in_zip = (
                            zip_file.namelist()
                        )

                        if "X_processed.csv" in files_in_zip:

                            st.session_state.x_train_bytes = (
                                zip_file.read(
                                    "X_processed.csv"
                                )
                            )

                        else:

                            st.session_state.x_train_bytes = None

                        if "pipeline_info.txt" in files_in_zip:

                            st.session_state.pipeline_info_bytes = (
                                zip_file.read(
                                    "pipeline_info.txt"
                                )
                            )

                        else:

                            st.session_state.pipeline_info_bytes = None

                        st.session_state.x_test_bytes = None

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

            except Exception as e:

                st.error(
                    f"An unexpected error occurred: "
                    f"{str(e)}"
                )

            finally:

                st.session_state.processing_running = False