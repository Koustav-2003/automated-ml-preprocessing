from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse

import pandas as pd
import io
import zipfile

from sklearn.model_selection import train_test_split

from pipeline import DataPreprocessor


app = FastAPI(
    title="Auto Data Preprocessing API",
    description="Automated feature engineering and feature selection",
    version="1.0"
)


# ==========================================================
# ROOT
# ==========================================================

@app.get("/")
def root():

    return {
        "message": "Auto Data Preprocessing API is running",
        "status": "OK"
    }


# ==========================================================
# PIPELINE INFO REPORT
# ==========================================================

def create_pipeline_report(info):
    """
    Convert pipeline information into a user-friendly
    human-readable report.
    """

    report = []

    # ------------------------------------------------------
    # HEADER
    # ------------------------------------------------------

    report.append("=" * 70)
    report.append("        AUTO DATA PREPROCESSING - PIPELINE REPORT")
    report.append("=" * 70)

    # ------------------------------------------------------
    # GENERAL INFORMATION
    # ------------------------------------------------------

    report.append("")
    report.append("GENERAL INFORMATION")
    report.append("-" * 70)

    report.append(
        f"Task                    : "
        f"{str(info.get('task', 'Unknown')).title()}"
    )

    report.append(
        f"Target Column           : "
        f"{info.get('target', 'Unknown')}"
    )

    # ------------------------------------------------------
    # FEATURE SUMMARY
    # ------------------------------------------------------

    report.append("")
    report.append("FEATURE SUMMARY")
    report.append("-" * 70)

    original_features = info.get(
        "original_feature_count",
        0
    )

    selected_features_count = info.get(
        "selected_feature_count",
        0
    )

    removed_features = (
        original_features -
        selected_features_count
    )

    report.append(
        f"Features Before Selection : "
        f"{original_features}"
    )

    report.append(
        f"Features After Selection  : "
        f"{selected_features_count}"
    )

    report.append(
        f"Features Removed          : "
        f"{removed_features}"
    )

    # ------------------------------------------------------
    # ID COLUMNS
    # ------------------------------------------------------

    report.append("")
    report.append("IDENTIFIER COLUMNS")
    report.append("-" * 70)

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

    # ------------------------------------------------------
    # MISSING VALUES
    # ------------------------------------------------------

    report.append("")
    report.append("MISSING VALUE HANDLING")
    report.append("-" * 70)

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

    # ------------------------------------------------------
    # SKEWNESS
    # ------------------------------------------------------

    report.append("")
    report.append("SKEWNESS HANDLING")
    report.append("-" * 70)

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

    # ------------------------------------------------------
    # SCALING
    # ------------------------------------------------------

    report.append("")
    report.append("FEATURE SCALING")
    report.append("-" * 70)

    scaled_features = info.get(
        "scaled_features",
        []
    )

    report.append(
        f"  Features scaled : "
        f"{len(scaled_features)}"
    )

    # ------------------------------------------------------
    # FEATURE SELECTION
    # ------------------------------------------------------

    report.append("")
    report.append("FEATURE SELECTION")
    report.append("-" * 70)

    report.append(
        "  Method : L1-based feature selection"
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

    # ------------------------------------------------------
    # END
    # ------------------------------------------------------

    report.append("")
    report.append("=" * 70)
    report.append("              END OF PIPELINE REPORT")
    report.append("=" * 70)

    return "\n".join(report)


# ==========================================================
# PROCESS DATASET
# ==========================================================

@app.post("/process")
async def process_dataset(
    file: UploadFile = File(...),
    target: str = Form(...)
):

    # ------------------------------------------------------
    # Validate file
    # ------------------------------------------------------

    if not file.filename.lower().endswith(".csv"):

        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported."
        )

    # ------------------------------------------------------
    # Read CSV
    # ------------------------------------------------------

    try:

        contents = await file.read()

        df = pd.read_csv(
            io.BytesIO(contents)
        )

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=f"Could not read CSV: {str(e)}"
        )

    # ------------------------------------------------------
    # Validate target
    # ------------------------------------------------------

    if target not in df.columns:

        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target}' not found."
        )

    # ------------------------------------------------------
    # Separate X and y
    # ------------------------------------------------------

    X = df.drop(
        columns=[target]
    )

    y = df[target]

    # ------------------------------------------------------
    # Create processor
    # ------------------------------------------------------

    processor = DataPreprocessor(
        target_col=target
    )

    # ------------------------------------------------------
    # Detect task
    # ------------------------------------------------------

    task = processor._detect_task(y)

    # ------------------------------------------------------
    # Train/test split
    # ------------------------------------------------------

    if task == "classification":

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )
        )

    else:

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42
            )
        )

    # ------------------------------------------------------
    # FIT pipeline ONLY on training data
    # ------------------------------------------------------

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
            detail=f"Pipeline fitting failed: {str(e)}"
        )

    # ------------------------------------------------------
    # Transform test
    # ------------------------------------------------------

    try:

        X_test_processed, test_ids = (
            processor.transform(
                X_test
            )
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Test transformation failed: {str(e)}"
        )

    # ------------------------------------------------------
    # Create training output
    # ------------------------------------------------------

    train_output = (
        X_train_processed.copy()
    )

    train_output[target] = (
        y_train.values
    )

    # ------------------------------------------------------
    # Add IDs back to training data
    # ------------------------------------------------------

    train_ids = pd.DataFrame(
        index=X_train.index
    )

    for col in processor.id_cols:

        if col in X_train.columns:

            train_ids[col] = X_train[col]

    if not train_ids.empty:

        train_output = pd.concat(
            [
                train_ids.reset_index(drop=True),
                train_output.reset_index(drop=True)
            ],
            axis=1
        )

    # ------------------------------------------------------
    # Add IDs back to test data
    # ------------------------------------------------------

    if not test_ids.empty:

        X_test_processed = pd.concat(
            [
                test_ids.reset_index(drop=True),
                X_test_processed.reset_index(drop=True)
            ],
            axis=1
        )

    # ======================================================
    # CREATE PIPELINE INFORMATION
    # ======================================================

    info = processor.get_info()

    info_text = create_pipeline_report(
        info
    )

    # ======================================================
    # CREATE ZIP
    # ======================================================

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(
        zip_buffer,
        "w",
        zipfile.ZIP_DEFLATED
    ) as zip_file:

        # ----------------------------------------------
        # X_train
        # ----------------------------------------------

        zip_file.writestr(
            "X_train.csv",
            train_output.to_csv(
                index=False
            )
        )

        # ----------------------------------------------
        # X_test
        # ----------------------------------------------

        zip_file.writestr(
            "X_test.csv",
            X_test_processed.to_csv(
                index=False
            )
        )

        # ----------------------------------------------
        # Pipeline information
        # ----------------------------------------------

        zip_file.writestr(
            "pipeline_info.txt",
            info_text
        )

    # Reset buffer
    zip_buffer.seek(0)

    # ======================================================
    # RETURN ZIP
    # ======================================================

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition":
                "attachment; "
                "filename=processed_dataset.zip"
        }
    )