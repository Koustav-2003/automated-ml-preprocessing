from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException
)

from fastapi.responses import (
    StreamingResponse
)

import pandas as pd
import numpy as np
import io
import zipfile
import base64
import html
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split
)

from pipeline import (
    SupervisedPreprocessor,
    UnsupervisedPreprocessor,
    process_supervised_dataset,
    process_unsupervised_dataset
)


# ==========================================================
# APP
# ==========================================================

app = FastAPI(
    title="Auto Data Preprocessing API",
    description=(
        "Automated EDA-ready preprocessing, "
        "feature engineering "
        "for supervised and unsupervised learning."
    ),
    version="2.0"
)


# ==========================================================
# ROOT
# ==========================================================

@app.get("/")
def root():

    return {
        "message":
            "Auto Data Preprocessing API is running",

        "status":
            "OK",

        "version":
            "2.0"
    }


# ==========================================================
# PIPELINE REPORT
# ==========================================================

def create_pipeline_report(info):

    report = []

    report.append(
        "=" * 70
    )

    report.append(
        "        AUTO DATA PREPROCESSING - PIPELINE REPORT"
    )

    report.append(
        "=" * 70
    )

    # ======================================================
    # GENERAL INFORMATION
    # ======================================================

    report.append("")

    report.append(
        "GENERAL INFORMATION"
    )

    report.append(
        "-" * 70
    )

    report.append(
        f"Dataset Type            : "
        f"{info.get('dataset_type', 'Unknown')}"
    )

    report.append(
        f"Task                    : "
        f"{str(info.get('task', 'Unknown')).title()}"
    )

    report.append(
        f"Target Column           : "
        f"{info.get('target', 'None')}"
    )

    report.append(
        f"Rows Processed          : "
        f"{info.get('rows_processed', 'Unknown')}"
    )

    # ======================================================
    # FEATURE SUMMARY
    # ======================================================

    report.append("")

    report.append(
        "FEATURE SUMMARY"
    )

    report.append(
        "-" * 70
    )

    original_features = info.get(
        "original_feature_count",
        0
    )

    selected_features_count = info.get(
        "selected_feature_count",
        0
    )

    removed_features = (
        original_features
        -
        selected_features_count
    )

    report.append(
        f"Features Before Preprocessing : "
        f"{original_features}"
    )

    report.append(
        f"Features After Preprocessing  : "
        f"{selected_features_count}"
    )

    report.append(
        f"Features Removed by Selection          : "
        f"{removed_features}"
    )

    # ======================================================
    # ID COLUMNS
    # ======================================================

    report.append("")

    report.append(
        "IDENTIFIER COLUMNS"
    )

    report.append(
        "-" * 70
    )

    id_columns = info.get(
        "id_columns",
        []
    )

    if id_columns:

        for column in id_columns:

            report.append(
                f"  • {column}"
            )

    else:

        report.append(
            "  None detected."
        )

    # ======================================================
    # MISSING VALUES
    # ======================================================

    report.append("")

    report.append(
        "MISSING VALUE HANDLING"
    )

    report.append(
        "-" * 70
    )

    missing_features = info.get(
        "missing_value_features",
        []
    )

    numeric_missing = info.get(
        "numeric_missing_features",
        []
    )

    categorical_missing = info.get(
        "categorical_missing_features",
        []
    )

    if not missing_features:

        report.append(
            "  No missing values detected."
        )

    else:

        report.append(
            f"  Total features with missing values : "
            f"{len(missing_features)}"
        )

        report.append("")

        if numeric_missing:

            report.append(
                "  Numerical features:"
            )

            for column in numeric_missing:

                report.append(
                    f"    • {column}"
                )

        if categorical_missing:

            report.append("")

            report.append(
                "  Categorical features:"
            )

            for column in categorical_missing:

                report.append(
                    f"    • {column}"
                )

    # ======================================================
    # SKEWNESS
    # ======================================================

    report.append("")

    report.append(
        "SKEWNESS HANDLING"
    )

    report.append(
        "-" * 70
    )

    skewed_features = info.get(
        "skewed_features",
        []
    )

    if skewed_features:

        report.append(
            f"  Features transformed : "
            f"{len(skewed_features)}"
        )

        report.append("")

        for column in skewed_features:

            report.append(
                f"    • {column}"
            )

    else:

        report.append(
            "  No significantly skewed features detected."
        )

    # ======================================================
    # SCALING
    # ======================================================

    report.append("")

    report.append(
        "FEATURE SCALING"
    )

    report.append(
        "-" * 70
    )

    scaled_features = info.get(
        "scaled_features",
        []
    )

    report.append(
        f"  Features scaled : "
        f"{len(scaled_features)}"
    )

    # ======================================================
    # FEATURE SELECTION (DISABLED)
    # ======================================================

    report.append("")

    report.append(
        "FEATURE SELECTION (DISABLED)"
    )

    report.append(
        "-" * 70
    )

    method = info.get(
        "feature_selection_method",
        "Unknown"
    )

    report.append(
        f"  Method : {method}"
    )

    report.append(
        f"  Features retained : "
        f"{selected_features_count}"
    )

    selected_features = info.get(
        "selected_features",
        []
    )

    report.append("")

    if selected_features:

        report.append(
            "  Selected Features:"
        )

        for number, feature in enumerate(
            selected_features,
            start=1
        ):

            report.append(
                f"    {number}. {feature}"
            )

    else:

        report.append(
            "  No features selected."
        )

    # ======================================================
    # END
    # ======================================================

    report.append("")

    report.append(
        "=" * 70
    )

    report.append(
        "              END OF PIPELINE REPORT"
    )

    report.append(
        "=" * 70
    )

    return "\n".join(report)


# ==========================================================
# CSV READER
# ==========================================================

async def read_csv_file(
    file,
    description
):

    if file is None:

        raise HTTPException(
            status_code=400,
            detail=(
                f"{description} was not uploaded."
            )
        )

    if not file.filename.lower().endswith(
        ".csv"
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                f"{description} must be a CSV file."
            )
        )

    try:

        contents = await file.read()

        if not contents:

            raise HTTPException(
                status_code=400,
                detail=(
                    f"{description} is empty."
                )
            )

        df = pd.read_csv(
            io.BytesIO(contents)
        )

        if df.empty:

            raise HTTPException(
                status_code=400,
                detail=(
                    f"{description} contains no data."
                )
            )

        return df

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Could not read "
                f"{description}: {str(e)}"
            )
        )


# ==========================================================
# ADD IDS
# ==========================================================

def add_ids(
    processed_df,
    original_df,
    id_columns
):

    if not id_columns:

        return processed_df

    ids = pd.DataFrame(
        index=original_df.index
    )

    for column in id_columns:

        if column in original_df.columns:

            ids[column] = (
                original_df[column]
            )

    if ids.empty:

        return processed_df

    return pd.concat(
        [
            ids.reset_index(
                drop=True
            ),

            processed_df.reset_index(
                drop=True
            )
        ],
        axis=1
    )


# ==========================================================
# PROCESS DATASET
# ==========================================================


# ==========================================================
# NOTEBOOK-BASED EDA REPORT
# ==========================================================

def _fig_to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _table_html(df):
    if df is None or df.empty:
        return "<p>No data available.</p>"
    return df.to_html(
        index=False,
        classes="data-table",
        border=0,
        escape=True
    )


def _image_html(image_b64, alt="EDA chart"):
    return (
        f'<img class="chart" src="data:image/png;base64,'
        f'{image_b64}" alt="{html.escape(alt)}">'
    )


def _safe_numeric(value):
    try:
        if pd.isna(value):
            return "NaN"
        return f"{value:.6g}"
    except Exception:
        return str(value)


def create_eda_report(
    dataset,
    supervised=False,
    target_column=None,
    dataset_name="dataset"
):
    """Create the downloadable EDA report using the uploaded notebooks."""

    df = dataset.copy()

    if supervised and (
        target_column is None or target_column not in df.columns
    ):
        raise ValueError("A valid target column is required for supervised EDA.")

    # These definitions mirror the notebooks.
    # Match the notebook's intent (numerical vs categorical), but use
    # pandas' dtype API so pandas StringDtype/category columns cannot be
    # accidentally sent through numpy.log().
    numerical_values = [
        feature for feature in df.columns
        if pd.api.types.is_numeric_dtype(df[feature])
    ]

    year_feature = [
        feature for feature in numerical_values
        if "Yr" in feature or "Year" in feature
    ]

    discrete_feature = [
        feature for feature in numerical_values
        if len(df[feature].unique()) < 25
        and feature not in year_feature + ["Id"]
    ]

    continuous_feature = [
        feature for feature in numerical_values
        if feature not in discrete_feature + year_feature + ["Id"]
    ]

    categorical_features = [
        feature for feature in df.columns
        if not pd.api.types.is_numeric_dtype(df[feature])
    ]

    features_with_nan = [
        feature for feature in df.columns
        if df[feature].isnull().sum() > 0
    ]

    sections = []

    sections.append(
        "<section><h2>1. Dataset Overview</h2>"
        "<div class='cards'>"
        f"<div class='card'><b>Rows</b><span>{len(df):,}</span></div>"
        f"<div class='card'><b>Columns</b><span>{len(df.columns):,}</span></div>"
        f"<div class='card'><b>Numerical</b><span>{len(numerical_values):,}</span></div>"
        f"<div class='card'><b>Categorical</b><span>{len(categorical_features):,}</span></div>"
        f"<div class='card'><b>Missing Values</b><span>{int(df.isnull().sum().sum()):,}</span></div>"
        f"<div class='card'><b>Duplicate Rows</b><span>{int(df.duplicated().sum()):,}</span></div>"
        "</div><h3>First 5 Records</h3>"
        + _table_html(df.head())
        + "</section>"
    )

    # Missing values
    missing_table = pd.DataFrame({
        "Feature": features_with_nan,
        "Missing Count": [
            int(df[c].isnull().sum()) for c in features_with_nan
        ],
        "Missing Percentage": [
            round(float(df[c].isnull().mean()), 4)
            for c in features_with_nan
        ]
    })

    missing_section = "<section><h2>2. Missing Values</h2>"
    missing_section += _table_html(missing_table)

    if features_with_nan:
        sorted_missing = missing_table.sort_values(
            "Missing Percentage", ascending=True
        )
        fig, ax = plt.subplots(
            figsize=(10, max(4, 0.28 * len(sorted_missing)))
        )
        ax.barh(
            sorted_missing["Feature"].astype(str),
            sorted_missing["Missing Percentage"] * 100
        )
        ax.set_xlabel("Missing values (%)")
        ax.set_title("Missing Value Percentage by Feature")
        missing_section += _image_html(
            _fig_to_base64(fig),
            "Missing value percentages"
        )
    else:
        missing_section += "<p>No missing values detected.</p>"

    missing_section += "</section>"
    sections.append(missing_section)

    # Target
    if supervised:
        target_data = df[target_column]
        target_is_numeric = pd.api.types.is_numeric_dtype(target_data)

        target_section = "<section><h2>3. Target Analysis</h2>"

        target_info = pd.DataFrame({
            "Statistic": ["Target Type", "Missing", "Unique Values", "Rows"],
            "Value": [
                "Numerical" if target_is_numeric else "Categorical",
                int(target_data.isnull().sum()),
                int(target_data.nunique()),
                len(target_data)
            ]
        })
        target_section += _table_html(target_info)

        if target_is_numeric:
            target_stats = pd.DataFrame({
                "Statistic": [
                    "Mean", "Median", "Std Dev",
                    "Minimum", "Maximum", "Skewness"
                ],
                "Value": [
                    _safe_numeric(target_data.mean()),
                    _safe_numeric(target_data.median()),
                    _safe_numeric(target_data.std()),
                    _safe_numeric(target_data.min()),
                    _safe_numeric(target_data.max()),
                    _safe_numeric(target_data.skew())
                ]
            })
            target_section += "<h3>Target Statistics</h3>" + _table_html(target_stats)

            clean_target = target_data.dropna()
            if not clean_target.empty:
                fig, ax = plt.subplots(figsize=(9, 4.5))
                ax.hist(clean_target, bins=30)
                ax.set_xlabel(target_column)
                ax.set_ylabel("Count")
                ax.set_title(f"Target Distribution - {target_column}")
                target_section += _image_html(
                    _fig_to_base64(fig), "Target distribution"
                )
        else:
            class_counts = (
                target_data.fillna("Missing")
                .astype(str)
                .value_counts()
            )
            class_table = pd.DataFrame({
                "Class": class_counts.index.astype(str),
                "Count": class_counts.values,
                "Percentage": (
                    class_counts.values / len(target_data) * 100
                ).round(2)
            })
            target_section += "<h3>Class Distribution</h3>" + _table_html(class_table)

            fig, ax = plt.subplots(figsize=(10, 5))
            class_counts.head(30).plot.bar(ax=ax)
            ax.set_xlabel(target_column)
            ax.set_ylabel("Count")
            ax.set_title(f"Class Distribution - {target_column}")
            plt.xticks(rotation=45, ha="right")
            target_section += _image_html(
                _fig_to_base64(fig), "Target class distribution"
            )

        target_section += "</section>"
        sections.append(target_section)

    # Missing-value relationship / distribution.
    section_no = 4 if supervised else 3
    if features_with_nan:
        title = (
            "Missing Values vs Target"
            if supervised else
            "Missing Value Distributions"
        )
        section = f"<section><h2>{section_no}. {title}</h2>"

        for feature in features_with_nan:
            if supervised:
                data = df[[feature, target_column]].copy()
                data["_missing_flag"] = np.where(
                    data[feature].isnull(), 1, 0
                )

                target_series = data[target_column]

                if pd.api.types.is_numeric_dtype(target_series):
                    grouped = (
                        data.groupby("_missing_flag")[target_column]
                        .median()
                    )

                    fig, ax = plt.subplots(figsize=(6, 3.5))
                    grouped.plot.bar(ax=ax)
                    ax.set_xlabel(
                        "Missing flag (0 = present, 1 = missing)"
                    )
                    ax.set_ylabel(f"Median {target_column}")
                    ax.set_title(feature)

                else:
                    grouped = pd.crosstab(
                        data["_missing_flag"],
                        data[target_column]
                        .fillna("Missing")
                        .astype(str),
                        normalize="index"
                    ) * 100

                    fig, ax = plt.subplots(figsize=(8, 4.5))
                    grouped.plot(
                        kind="bar",
                        stacked=True,
                        ax=ax
                    )
                    ax.set_xlabel(
                        "Missing flag (0 = present, 1 = missing)"
                    )
                    ax.set_ylabel("Target class percentage")
                    ax.set_title(
                        f"{feature} missingness vs {target_column}"
                    )
                    ax.legend(
                        title=target_column,
                        bbox_to_anchor=(1.02, 1),
                        loc="upper left"
                    )
            else:
                flags = np.where(df[feature].isnull(), 1, 0)
                counts = pd.Series(flags).value_counts().sort_index()

                fig, ax = plt.subplots(figsize=(6, 3.5))
                counts.plot.bar(ax=ax)
                ax.set_xlabel("Missing flag (0 = present, 1 = missing)")
                ax.set_ylabel("Number of rows")
                ax.set_title(feature)

            section += (
                f"<h3>{html.escape(feature)}</h3>"
                + _image_html(
                    _fig_to_base64(fig),
                    f"Missing analysis for {feature}"
                )
            )

        section += "</section>"
        sections.append(section)
    section_no += 1

    # Numerical features
    sections.append(
        f"<section><h2>{section_no}. Numerical Features</h2>"
        f"<p>Number of numerical features: {len(numerical_values)}</p>"
        + _table_html(pd.DataFrame({"Feature": numerical_values}))
        + "</section>"
    )
    section_no += 1

    # Temporal
    temporal = f"<section><h2>{section_no}. Temporal Variables</h2>"
    temporal += _table_html(pd.DataFrame({
        "Feature": year_feature,
        "Unique Values": [int(df[c].nunique()) for c in year_feature]
    }))

    for feature in year_feature:
        vals = df[feature].dropna()
        if vals.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 4))
        vals.value_counts().sort_index().plot.bar(ax=ax)
        ax.set_xlabel(feature)
        ax.set_ylabel("Number of rows")
        ax.set_title(f"{feature} - distribution")
        plt.xticks(rotation=45, ha="right")
        temporal += _image_html(
            _fig_to_base64(fig), f"{feature} distribution"
        )

        if (
            supervised
            and feature != "YrSold"
            and "YrSold" in df.columns
        ):
            pair = df[[feature, "YrSold", target_column]].dropna().copy()

            if not pair.empty:
                pair[feature] = (
                    pair["YrSold"] - pair[feature]
                )

                target_series = pair[target_column]

                if pd.api.types.is_numeric_dtype(target_series):
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.scatter(
                        pair[feature],
                        pair[target_column],
                        s=12,
                        alpha=0.55
                    )
                    ax.set_xlabel(feature)
                    ax.set_ylabel(target_column)
                    ax.set_title(f"{feature} vs {target_column}")
                    temporal += _image_html(
                        _fig_to_base64(fig),
                        f"{feature} versus target"
                    )
                else:
                    grouped = pd.crosstab(
                        pair[feature],
                        pair[target_column]
                        .fillna("Missing")
                        .astype(str),
                        normalize="index"
                    ) * 100

                    fig, ax = plt.subplots(figsize=(9, 4.5))
                    grouped.plot(
                        kind="bar",
                        stacked=True,
                        ax=ax
                    )
                    ax.set_xlabel(feature)
                    ax.set_ylabel("Target class percentage")
                    ax.set_title(
                        f"{feature} vs {target_column}"
                    )
                    ax.legend(
                        title=target_column,
                        bbox_to_anchor=(1.02, 1),
                        loc="upper left"
                    )
                    plt.xticks(rotation=45, ha="right")

                    temporal += _image_html(
                        _fig_to_base64(fig),
                        f"{feature} versus categorical target"
                    )

    sections.append(temporal + "</section>")
    section_no += 1

    # Discrete
    discrete = (
        f"<section><h2>{section_no}. Discrete Numerical Features</h2>"
        f"<p>Total: {len(discrete_feature)}</p>"
    )
    for feature in discrete_feature:
        if supervised:
            target_series = df[target_column]

            if pd.api.types.is_numeric_dtype(target_series):
                series = (
                    df.groupby(feature)[target_column]
                    .median()
                    .sort_index()
                )
                ylabel = f"Median {target_column}"
                title = feature

            else:
                temp = df[[feature, target_column]].copy()
                temp[target_column] = (
                    temp[target_column]
                    .fillna("Missing")
                    .astype(str)
                )

                series = (
                    pd.crosstab(
                        temp[feature],
                        temp[target_column],
                        normalize="index"
                    ) * 100
                )
                ylabel = "Target class percentage"
                title = f"{feature} vs {target_column}"
        else:
            series = df[feature].value_counts().sort_index()
            ylabel = "Number of rows"
            title = feature

        fig, ax = plt.subplots(figsize=(8, 4))

        if isinstance(series, pd.DataFrame):
            series.plot(
                kind="bar",
                stacked=True,
                ax=ax
            )
            ax.legend(
                title=target_column,
                bbox_to_anchor=(1.02, 1),
                loc="upper left"
            )
        else:
            series.plot.bar(ax=ax)

        ax.set_xlabel(feature)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        plt.xticks(rotation=45, ha="right")

        discrete += (
            f"<h3>{html.escape(feature)}</h3>"
            + _image_html(_fig_to_base64(fig), f"Discrete feature {feature}")
        )

    sections.append(discrete + "</section>")
    section_no += 1

    # Continuous
    continuous = (
        f"<section><h2>{section_no}. Continuous Numerical Features</h2>"
        f"<p>Total: {len(continuous_feature)}</p>"
    )
    for feature in continuous_feature:
        data = df[feature].dropna()
        if data.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(data, bins=25)
        ax.set_xlabel(feature)
        ax.set_ylabel("Number of rows")
        ax.set_title(feature)
        continuous += (
            f"<h3>{html.escape(feature)}</h3>"
            + _image_html(_fig_to_base64(fig), f"Distribution of {feature}")
        )

        # Notebook logic inspects log-transformed continuous features.
        # Only apply log when all observed values are strictly positive;
        # this preserves the notebook's intent while preventing invalid
        # log operations on zero/negative/string-like data.
        if (not data.empty) and np.isfinite(data).all() and (data > 0).all():
            log_data = np.log(data.to_numpy(dtype=float))
            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.boxplot(log_data, vert=False)
            ax.set_xlabel(f"log({feature})")
            ax.set_title(f"Log-transformed Box Plot - {feature}")
            continuous += _image_html(
                _fig_to_base64(fig),
                f"Log transformed boxplot of {feature}"
            )

    sections.append(continuous + "</section>")
    section_no += 1

    note = (
        "The supervised notebook checks log transformation of continuous "
        "features when zero is absent and also logs SalePrice in its "
        "transformation cell. The unsupervised notebook performs the same "
        "continuous-feature inspection without a target."
    )
    sections.append(
        f"<section><h2>{section_no}. Transformation Note</h2>"
        f"<p>{html.escape(note)}</p></section>"
    )
    section_no += 1

    # Correlation
    corr = df.corr(numeric_only=True)
    corr_section = (
        f"<section><h2>{section_no}. Multicollinearity / Correlation</h2>"
        + _table_html(
            corr.reset_index()
            .rename(columns={"index": "Feature"})
            .round(4)
        )
    )
    if not corr.empty:
        n = len(corr.columns)
        fig, ax = plt.subplots(
            figsize=(max(10, min(20, 0.45*n)),
                     max(8, min(16, 0.45*n)))
        )
        sns.heatmap(corr, cmap="coolwarm", ax=ax)
        ax.set_title("Numerical Correlation Matrix")
        corr_section += _image_html(
            _fig_to_base64(fig), "Numerical correlation heatmap"
        )
    sections.append(corr_section + "</section>")
    section_no += 1

    # Categorical
    cat_section = (
        f"<section><h2>{section_no}. Categorical Features</h2>"
        f"<p>Total categorical features: {len(categorical_features)}</p>"
        "<h3>Cardinality</h3>"
        + _table_html(pd.DataFrame({
            "Feature": categorical_features,
            "Cardinality": [
                int(df[c].nunique(dropna=False))
                for c in categorical_features
            ]
        }))
    )

    for feature in categorical_features:
        counts = (
            df[feature].fillna("Missing")
            .astype(str)
            .value_counts()
        )

        if supervised:
            temp = df.assign(
                _cat=df[feature]
                .fillna("Missing")
                .astype(str)
            )

            target_series = temp[target_column]

            if pd.api.types.is_numeric_dtype(target_series):
                series = (
                    temp.groupby("_cat")[target_column]
                    .median()
                    .sort_values(ascending=False)
                    .head(30)
                )
                ylabel = f"Median {target_column}"
                title = f"{feature} vs {target_column}"
            else:
                series = (
                    pd.crosstab(
                        temp["_cat"],
                        target_series
                        .fillna("Missing")
                        .astype(str),
                        normalize="index"
                    ).head(30) * 100
                )
                ylabel = "Target class percentage"
                title = f"{feature} vs {target_column}"
        else:
            series = counts.head(30)
            ylabel = "Number of rows"
            title = f"{feature} - distribution"

        fig, ax = plt.subplots(figsize=(10, 5))

        if isinstance(series, pd.DataFrame):
            series.plot(
                kind="bar",
                stacked=True,
                ax=ax
            )
            ax.legend(
                title=target_column,
                bbox_to_anchor=(1.02, 1),
                loc="upper left"
            )
        else:
            series.plot.bar(ax=ax)

        ax.set_xlabel(feature)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        plt.xticks(rotation=45, ha="right")

        category_table = pd.DataFrame({
            "Category": counts.head(30).index.astype(str),
            "Count": counts.head(30).values,
            "Percentage": (
                counts.head(30).values / len(df) * 100
            ).round(2)
        })

        cat_section += (
            f"<h3>{html.escape(feature)}</h3>"
            + _image_html(_fig_to_base64(fig), f"Categorical analysis for {feature}")
            + _table_html(category_table)
        )

    sections.append(cat_section + "</section>")

    mode = "Supervised" if supervised else "Unsupervised"
    target_text = (
        f"Target: {html.escape(str(target_column))}"
        if supervised else "No target variable"
    )

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{mode} EDA Report</title>
<style>
body {{font-family:Arial,Helvetica,sans-serif;margin:0;background:#0f1117;color:#e8e8e8;line-height:1.55}}
.container {{max-width:1400px;margin:auto;padding:36px}}
h1 {{font-size:32px}}
h2 {{margin-top:42px;border-bottom:1px solid #444;padding-bottom:8px}}
h3 {{margin-top:26px}}
.cards {{display:flex;flex-wrap:wrap;gap:12px;margin:20px 0}}
.card {{border:1px solid #333;border-radius:10px;padding:16px;min-width:150px;background:#151821}}
.card span {{display:block;font-size:25px;margin-top:5px}}
.data-table {{border-collapse:collapse;width:100%;margin:12px 0 24px;background:#151821}}
.data-table th,.data-table td {{border:1px solid #333;padding:7px 9px;text-align:left}}
.data-table th {{background:#20232d}}
.chart {{max-width:100%;height:auto;background:white;padding:4px;border-radius:6px;margin:8px 0 24px}}
.note {{padding:14px;border-left:4px solid #888;background:#171a22}}
</style>
</head>
<body>
<div class="container">
<h1>📊 {mode} Exploratory Data Analysis Report</h1>
<p><b>Dataset:</b> {html.escape(str(dataset_name))}<br><b>{target_text}</b></p>
<p class="note">
Generated from the uploaded EDA_supervised and EDA_unsupervised notebooks.
No external EDA library is used. The uploaded dataset is not modified by this report.
</p>
{"".join(sections)}
</div>
</body>
</html>"""


@app.post("/eda")
async def generate_eda_endpoint(
    ml_task: str = Form("Supervised Learning"),
    dataset_type: str = Form("Entire Dataset"),
    target: str = Form(None),
    file: UploadFile = File(None),
    train_file: UploadFile = File(None),
    test_file: UploadFile = File(None)
):
    supervised = ml_task == "Supervised Learning"

    if ml_task not in ["Supervised Learning", "Unsupervised Learning"]:
        raise HTTPException(status_code=400, detail="Invalid ML task.")

    if dataset_type == "Test Dataset":
        if train_file is None:
            raise HTTPException(
                status_code=400,
                detail="Training dataset is required for EDA."
            )
        df = await read_csv_file(train_file, "Training Dataset")
        report_name = train_file.filename or "training_dataset.csv"
    else:
        if file is None:
            raise HTTPException(
                status_code=400,
                detail="A dataset is required for EDA."
            )
        df = await read_csv_file(file, "Dataset")
        report_name = file.filename or "dataset.csv"

    if df.empty:
        raise HTTPException(status_code=400, detail="Dataset contains no rows.")

    if supervised and (target is None or target not in df.columns):
        raise HTTPException(
            status_code=400,
            detail="A valid target column is required for supervised EDA."
        )

    try:
        report_html = create_eda_report(
            df,
            supervised=supervised,
            target_column=target if supervised else None,
            dataset_name=report_name
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"EDA report generation failed: {str(e)}"
        )

    return StreamingResponse(
        io.BytesIO(report_html.encode("utf-8")),
        media_type="text/html",
        headers={
            "Content-Disposition":
                'attachment; filename="EDA_report.html"'
        }
    )


@app.post("/process")
async def process_dataset(

    ml_task: str = Form(
        "Supervised Learning"
    ),

    dataset_type: str = Form(
        "Entire Dataset"
    ),

    target: str = Form(
        None
    ),

    test_size: float = Form(
        0.20
    ),

    # Entire / Training / Unsupervised
    file: UploadFile = File(
        None
    ),

    # Test workflow
    train_file: UploadFile = File(
        None
    ),

    test_file: UploadFile = File(
        None
    )
):

    # ======================================================
    # VALIDATE ML TASK
    # ======================================================

    valid_ml_tasks = [
        "Supervised Learning",
        "Unsupervised Learning"
    ]

    if ml_task not in valid_ml_tasks:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid ML task. Choose "
                "Supervised Learning or "
                "Unsupervised Learning."
            )
        )

    # ======================================================
    # VALIDATE TEST SIZE
    # ======================================================

    if not 0 < test_size < 1:

        raise HTTPException(
            status_code=400,
            detail=(
                "Test size must be between 0 and 1 "
                "(for example, 0.20 for 20%)."
            )
        )

    # ======================================================
    # UNSUPERVISED
    # ======================================================

    if ml_task == "Unsupervised Learning":

        valid_unsupervised_dataset_types = [
            "Entire Dataset",
            "Training Dataset",
            "Test Dataset"
        ]

        if dataset_type not in valid_unsupervised_dataset_types:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid unsupervised dataset type. Choose "
                    "Entire Dataset, Training Dataset, "
                    "or Test Dataset."
                )
            )

        # ==================================================
        # ENTIRE DATASET
        # ==================================================

        if dataset_type == "Entire Dataset":

            if file is None:

                raise HTTPException(
                    status_code=400,
                    detail=(
                        "A dataset is required "
                        "for unsupervised learning."
                    )
                )

            df = await read_csv_file(
                file,
                "Dataset"
            )

            if df.empty:

                raise HTTPException(
                    status_code=400,
                    detail="Dataset contains no rows."
                )

            # --------------------------------------------------
            # The standalone unsupervised pipeline performs the
            # 80/20 split and fits preprocessing ONLY on X_train.
            # --------------------------------------------------

            try:

                result = process_unsupervised_dataset(
                    df=df,
                    test_size=test_size,
                    random_state=42
                )

            except Exception as e:

                raise HTTPException(
                    status_code=500,
                    detail=(
                        "Unsupervised preprocessing "
                        f"failed: {str(e)}"
                    )
                )

            zip_buffer = io.BytesIO()

            with zipfile.ZipFile(
                zip_buffer,
                "w",
                zipfile.ZIP_DEFLATED
            ) as zip_file:

                zip_file.writestr(
                    "X_train.csv",
                    result["X_train"].to_csv(
                        index=False
                    )
                )

                zip_file.writestr(
                    "X_test.csv",
                    result["X_test"].to_csv(
                        index=False
                    )
                )

                zip_file.writestr(
                    "pipeline_info.txt",
                    create_pipeline_report(
                        {
                            **result["info"],
                            "dataset_type":
                                dataset_type,
                            "rows_processed":
                                len(df),
                            "task":
                                "Unsupervised",
                            "target":
                                "None",
                            "original_feature_count":
                                result["info"].get(
                                    "feature_count_before_processing",
                                    0
                                ),
                            "selected_feature_count":
                                result["info"].get(
                                    "final_feature_count",
                                    0
                                ),
                            "feature_selection_method":
                                "None"
                        }
                    )
                )

            zip_buffer.seek(0)

            return StreamingResponse(

                zip_buffer,

                media_type="application/zip",

                headers={
                    "Content-Disposition":
                        "attachment; "
                        "filename="
                        "processed_unsupervised_dataset.zip"
                }
            )

        # ==================================================
        # TRAINING DATASET
        # ==================================================

        elif dataset_type == "Training Dataset":

            if file is None:

                raise HTTPException(
                    status_code=400,
                    detail=(
                        "A training dataset is required "
                        "for unsupervised learning."
                    )
                )

            train_df = await read_csv_file(
                file,
                "Training Dataset"
            )

            if train_df.empty:

                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Training dataset contains no rows."
                    )
                )

            X_train = train_df.copy()

            # --------------------------------------------------
            # Fit ONLY on the supplied training dataset.
            # No split is performed here because the user has
            # already supplied the training portion.
            # --------------------------------------------------

            processor = UnsupervisedPreprocessor()

            try:

                X_train_processed, train_ids = (
                    processor.fit_transform(
                        X_train
                    )
                )

            except Exception as e:

                raise HTTPException(
                    status_code=500,
                    detail=(
                        "Unsupervised preprocessing on the "
                        f"training dataset failed: {str(e)}"
                    )
                )

            train_output = X_train_processed.copy()

            if not train_ids.empty:

                train_output = pd.concat(
                    [
                        train_ids.reset_index(
                            drop=True
                        ),

                        train_output.reset_index(
                            drop=True
                        )
                    ],
                    axis=1
                )

            info = processor.get_info()

            info["dataset_type"] = (
                dataset_type
            )

            info["rows_processed"] = (
                len(train_df)
            )

            info_text = create_pipeline_report(
                {
                    **info,
                    "task":
                        "Unsupervised",
                    "target":
                        "None",
                    "original_feature_count":
                        info.get(
                            "feature_count_before_processing",
                            0
                        ),
                    "selected_feature_count":
                        info.get(
                            "final_feature_count",
                            0
                        ),
                    "feature_selection_method":
                        "None"
                }
            )

            zip_buffer = io.BytesIO()

            with zipfile.ZipFile(
                zip_buffer,
                "w",
                zipfile.ZIP_DEFLATED
            ) as zip_file:

                zip_file.writestr(
                    "X_train.csv",
                    train_output.to_csv(
                        index=False
                    )
                )

                zip_file.writestr(
                    "pipeline_info.txt",
                    info_text
                )

            zip_buffer.seek(0)

            return StreamingResponse(

                zip_buffer,

                media_type="application/zip",

                headers={
                    "Content-Disposition":
                        "attachment; "
                        "filename="
                        "processed_unsupervised_training_dataset.zip"
                }
            )

        # ==================================================
        # TEST DATASET
        # ==================================================

        else:

            if train_file is None:

                raise HTTPException(
                    status_code=400,
                    detail=(
                        "A training dataset is required "
                        "when processing an unsupervised "
                        "test dataset."
                    )
                )

            if test_file is None:

                raise HTTPException(
                    status_code=400,
                    detail=(
                        "A test dataset is required "
                        "for unsupervised learning."
                    )
                )

            train_df = await read_csv_file(
                train_file,
                "Training Dataset"
            )

            test_df = await read_csv_file(
                test_file,
                "Test Dataset"
            )

            if train_df.empty:

                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Training dataset contains no rows."
                    )
                )

            if test_df.empty:

                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Test dataset contains no rows."
                    )
                )

            X_train = train_df.copy()
            X_test = test_df.copy()

            # --------------------------------------------------
            # Fit ONLY on the supplied training dataset.
            # The test dataset is transformed using the fitted
            # training pipeline.
            # --------------------------------------------------

            processor = UnsupervisedPreprocessor()

            try:

                X_train_processed, train_ids = (
                    processor.fit_transform(
                        X_train
                    )
                )

            except Exception as e:

                raise HTTPException(
                    status_code=500,
                    detail=(
                        "Unsupervised preprocessing could "
                        "not be fitted on the training "
                        f"dataset: {str(e)}"
                    )
                )

            try:

                X_test_processed, test_ids = (
                    processor.transform(
                        X_test
                    )
                )

            except Exception as e:

                raise HTTPException(
                    status_code=500,
                    detail=(
                        "Unsupervised test transformation "
                        f"failed: {str(e)}"
                    )
                )

            # --------------------------------------------------
            # Add detected ID columns back to both outputs.
            # --------------------------------------------------

            train_output = (
                X_train_processed.copy()
            )

            if not train_ids.empty:

                train_output = pd.concat(
                    [
                        train_ids.reset_index(
                            drop=True
                        ),

                        train_output.reset_index(
                            drop=True
                        )
                    ],
                    axis=1
                )

            test_output = (
                X_test_processed.copy()
            )

            if not test_ids.empty:

                test_output = pd.concat(
                    [
                        test_ids.reset_index(
                            drop=True
                        ),

                        test_output.reset_index(
                            drop=True
                        )
                    ],
                    axis=1
                )

            info = processor.get_info()

            info["dataset_type"] = (
                dataset_type
            )

            info["rows_processed"] = (
                len(test_df)
            )

            info_text = create_pipeline_report(
                {
                    **info,
                    "task":
                        "Unsupervised",
                    "target":
                        "None",
                    "original_feature_count":
                        info.get(
                            "feature_count_before_processing",
                            0
                        ),
                    "selected_feature_count":
                        info.get(
                            "final_feature_count",
                            0
                        ),
                    "feature_selection_method":
                        "None"
                }
            )

            zip_buffer = io.BytesIO()

            with zipfile.ZipFile(
                zip_buffer,
                "w",
                zipfile.ZIP_DEFLATED
            ) as zip_file:

                zip_file.writestr(
                    "X_train.csv",
                    train_output.to_csv(
                        index=False
                    )
                )

                zip_file.writestr(
                    "X_test.csv",
                    test_output.to_csv(
                        index=False
                    )
                )

                zip_file.writestr(
                    "pipeline_info.txt",
                    info_text
                )

            zip_buffer.seek(0)

            return StreamingResponse(

                zip_buffer,

                media_type="application/zip",

                headers={
                    "Content-Disposition":
                        "attachment; "
                        "filename="
                        "processed_unsupervised_test_dataset.zip"
                }
            )

    # ======================================================
    # SUPERVISED VALIDATION
    # ======================================================

    valid_dataset_types = [
        "Entire Dataset",
        "Training Dataset",
        "Test Dataset"
    ]

    if dataset_type not in valid_dataset_types:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid dataset type. Choose "
                "Entire Dataset, Training Dataset, "
                "or Test Dataset."
            )
        )

    if target is None or target == "":

        raise HTTPException(
            status_code=400,
            detail=(
                "A target column is required "
                "for supervised learning."
            )
        )

    # ======================================================
    # ENTIRE DATASET
    # ======================================================

    if dataset_type == "Entire Dataset":

        df = await read_csv_file(
            file,
            "Dataset"
        )

        if target not in df.columns:

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Target column '{target}' "
                    f"not found."
                )
            )

        # --------------------------------------------------
        # Use standalone supervised pipeline
        # --------------------------------------------------

        try:

            result = process_supervised_dataset(
                df=df,
                target_col=target,
                test_size=test_size,
                random_state=42
            )

        except Exception as e:

            raise HTTPException(
                status_code=500,
                detail=(
                    "Supervised preprocessing "
                    f"failed: {str(e)}"
                )
            )

        # --------------------------------------------------
        # Return generated train/test outputs
        # --------------------------------------------------

        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(
            zip_buffer,
            "w",
            zipfile.ZIP_DEFLATED
        ) as zip_file:

            zip_file.writestr(
                "X_train.csv",
                result["X_train"].to_csv(
                    index=False
                )
            )

            zip_file.writestr(
                "X_test.csv",
                result["X_test"].to_csv(
                    index=False
                )
            )

            zip_file.writestr(
                "pipeline_info.txt",
                create_pipeline_report(
                    {
                        **result["info"],
                        "dataset_type":
                            dataset_type,
                        "rows_processed":
                            len(df),
                        "feature_selection_method": "None"
                    }
                )
            )

        zip_buffer.seek(0)

        return StreamingResponse(

            zip_buffer,

            media_type="application/zip",

            headers={
                "Content-Disposition":
                    "attachment; "
                    "filename="
                    "processed_dataset.zip"
            }
        )

    # ======================================================
    # TRAINING DATASET
    # ======================================================

    elif dataset_type == "Training Dataset":

        df = await read_csv_file(
            file,
            "Training Dataset"
        )

        if target not in df.columns:

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Target column '{target}' "
                    f"not found."
                )
            )

        X_train = df.drop(
            columns=[target]
        )

        y_train = df[target]

        processor = SupervisedPreprocessor(
            target_col=target
        )

        try:

            X_train_processed = (
                processor.fit_transform(
                    X_train,
                    y_train
                )
            )

        except Exception as e:

            raise HTTPException(
                status_code=500,
                detail=(
                    "Pipeline fitting failed: "
                    f"{str(e)}"
                )
            )

        train_output = (
            X_train_processed.copy()
        )

        train_output[target] = (
            y_train.values
        )

        train_output = add_ids(
            train_output,
            X_train,
            processor.id_cols
        )

        info = processor.get_info()

        info["dataset_type"] = (
            dataset_type
        )

        info["rows_processed"] = len(df)

        info_text = create_pipeline_report(
            info
        )

        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(
            zip_buffer,
            "w",
            zipfile.ZIP_DEFLATED
        ) as zip_file:

            zip_file.writestr(
                "X_train.csv",
                train_output.to_csv(
                    index=False
                )
            )

            zip_file.writestr(
                "pipeline_info.txt",
                info_text
            )

        zip_buffer.seek(0)

        return StreamingResponse(

            zip_buffer,

            media_type="application/zip",

            headers={
                "Content-Disposition":
                    "attachment; "
                    "filename="
                    "processed_training_dataset.zip"
            }
        )

    # ======================================================
    # TEST DATASET
    # ======================================================

    else:

        if train_file is None:

            raise HTTPException(
                status_code=400,
                detail=(
                    "A training dataset is required "
                    "when processing a test dataset."
                )
            )

        if test_file is None:

            raise HTTPException(
                status_code=400,
                detail=(
                    "A test dataset is required."
                )
            )

        train_df = await read_csv_file(
            train_file,
            "Training Dataset"
        )

        test_df = await read_csv_file(
            test_file,
            "Test Dataset"
        )

        if target not in train_df.columns:

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Target column '{target}' "
                    "not found in training dataset."
                )
            )

        X_train = train_df.drop(
            columns=[target]
        )

        y_train = train_df[target]

        # Preserve the test target when it is present in test.csv.
        # It is NOT used for fitting the preprocessing pipeline.
        # It is only restored to the processed test output so the
        # user can later compare predictions against actual labels.
        has_test_target = target in test_df.columns

        if has_test_target:
            y_test = test_df[target].copy()
            X_test = test_df.drop(
                columns=[target]
            ).copy()
        else:
            y_test = None
            X_test = test_df.copy()

        processor = SupervisedPreprocessor(
            target_col=target
        )

        task = processor.detect_task(
            y_train
        )

        # --------------------------------------------------
        # Fit ONLY on training data
        # --------------------------------------------------

        try:

            X_train_processed = processor.fit_transform(
                X_train,
                y_train
            )

        except Exception as e:

            raise HTTPException(
                status_code=500,
                detail=(
                    "Pipeline fitting on training "
                    f"data failed: {str(e)}"
                )
            )

        # --------------------------------------------------
        # Build processed training output
        # --------------------------------------------------

        train_output = X_train_processed.copy()

        # The target is retained in X_train because this is the
        # labelled training dataset.
        train_output[target] = y_train.to_numpy()

        train_ids = pd.DataFrame(
            index=X_train.index
        )

        for col in processor.id_cols:

            if col in X_train.columns:

                train_ids[col] = X_train[col]

        if not train_ids.empty:

            train_output = pd.concat(
                [
                    train_ids.reset_index(
                        drop=True
                    ),

                    train_output.reset_index(
                        drop=True
                    )
                ],
                axis=1
            )

        # --------------------------------------------------
        # Transform test
        # --------------------------------------------------

        try:

            X_test_processed, test_ids = (
                processor.transform(
                    X_test
                )
            )

        except Exception as e:

            raise HTTPException(
                status_code=500,
                detail=(
                    "Test transformation failed: "
                    f"{str(e)}"
                )
            )

        # --------------------------------------------------
        # Add IDs
        # --------------------------------------------------

        if not test_ids.empty:

            X_test_processed = pd.concat(
                [
                    test_ids.reset_index(
                        drop=True
                    ),

                    X_test_processed.reset_index(
                        drop=True
                    )
                ],
                axis=1
            )

        # --------------------------------------------------
        # Restore the original test target, if test.csv
        # contained it. Assign positionally to avoid pandas
        # index-alignment turning valid labels into NaN/None.
        # --------------------------------------------------

        if y_test is not None:
            X_test_processed[target] = y_test.to_numpy()

        # --------------------------------------------------
        # Report
        # --------------------------------------------------

        info = processor.get_info()

        info["dataset_type"] = (
            dataset_type
        )

        info["rows_processed"] = len(test_df)

        info_text = create_pipeline_report(
            info
        )

        # --------------------------------------------------
        # ZIP
        # --------------------------------------------------

        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(
            zip_buffer,
            "w",
            zipfile.ZIP_DEFLATED
        ) as zip_file:

            zip_file.writestr(
                "X_train.csv",
                train_output.to_csv(
                    index=False
                )
            )

            zip_file.writestr(
                "X_test.csv",
                X_test_processed.to_csv(
                    index=False
                )
            )

            zip_file.writestr(
                "pipeline_info.txt",
                info_text
            )

        zip_buffer.seek(0)

        return StreamingResponse(

            zip_buffer,

            media_type="application/zip",

            headers={
                "Content-Disposition":
                    "attachment; "
                    "filename="
                    "processed_test_dataset.zip"
            }
        )