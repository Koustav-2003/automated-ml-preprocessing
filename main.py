from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse

import pandas as pd
import io
import zipfile
import tempfile
import os

import sweetviz as sv

from sklearn.model_selection import train_test_split

from pipeline import DataPreprocessor


app = FastAPI(
    title="Auto Data Preprocessing API",
    description="Automated EDA, feature engineering and feature selection",
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
# SWEETVIZ EDA REPORT
# ==========================================================

def create_eda_report(df, target):

    try:

        # --------------------------------------------------
        # Create temporary HTML file
        # --------------------------------------------------

        temp_file = tempfile.NamedTemporaryFile(
            suffix=".html",
            delete=False
        )

        temp_path = temp_file.name

        temp_file.close()

        # --------------------------------------------------
        # Generate Sweetviz report
        # --------------------------------------------------

        if target in df.columns:

            report = sv.analyze(
                df,
                target_feat=target
            )

        else:

            report = sv.analyze(
                df
            )

        report.show_html(
            filepath=temp_path,
            open_browser=False,
            layout="widescreen"
        )

        # --------------------------------------------------
        # Read generated HTML
        # --------------------------------------------------

        with open(
            temp_path,
            "rb"
        ) as html_file:

            report_bytes = html_file.read()

        # --------------------------------------------------
        # Delete temporary file
        # --------------------------------------------------

        os.remove(
            temp_path
        )

        return report_bytes

    except Exception as e:

        # --------------------------------------------------
        # Clean temporary file if something failed
        # --------------------------------------------------

        try:

            if os.path.exists(temp_path):

                os.remove(
                    temp_path
                )

        except Exception:

            pass

        raise HTTPException(
            status_code=500,
            detail=(
                f"EDA report generation failed: "
                f"{str(e)}"
            )
        )


# ==========================================================
# PIPELINE INFO REPORT
# ==========================================================

def create_pipeline_report(info):

    report = []

    report.append("=" * 70)
    report.append(
        "        AUTO DATA PREPROCESSING - PIPELINE REPORT"
    )
    report.append("=" * 70)

    # ------------------------------------------------------
    # GENERAL INFORMATION
    # ------------------------------------------------------

    report.append("")
    report.append("GENERAL INFORMATION")
    report.append("-" * 70)

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
        f"{info.get('target', 'Unknown')}"
    )

    report.append(
        f"Rows Processed          : "
        f"{info.get('rows_processed', 'Unknown')}"
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
    report.append(
        "              END OF PIPELINE REPORT"
    )
    report.append("=" * 70)

    return "\n".join(report)


# ==========================================================
# CSV READER
# ==========================================================

async def read_csv_file(file, description):

    if file is None:

        raise HTTPException(
            status_code=400,
            detail=f"{description} was not uploaded."
        )

    if not file.filename.lower().endswith(".csv"):

        raise HTTPException(
            status_code=400,
            detail=f"{description} must be a CSV file."
        )

    try:

        contents = await file.read()

        if not contents:

            raise HTTPException(
                status_code=400,
                detail=f"{description} is empty."
            )

        df = pd.read_csv(
            io.BytesIO(contents)
        )

        if df.empty:

            raise HTTPException(
                status_code=400,
                detail=f"{description} contains no data."
            )

        return df

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Could not read {description}: "
                f"{str(e)}"
            )
        )


# ==========================================================
# PROCESS DATASET
# ==========================================================

@app.post("/process")
async def process_dataset(

    dataset_type: str = Form(...),

    target: str = Form(...),

    # Used for Entire Dataset / Training Dataset
    file: UploadFile = File(None),

    # Used for Test Dataset
    train_file: UploadFile = File(None),

    test_file: UploadFile = File(None)
):

    valid_dataset_types = [
        "Entire Dataset",
        "Training Dataset",
        "Test Dataset"
    ]

    # ======================================================
    # VALIDATE DATASET TYPE
    # ======================================================

    if dataset_type not in valid_dataset_types:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid dataset type. "
                "Choose Entire Dataset, "
                "Training Dataset, or Test Dataset."
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

        # --------------------------------------------------
        # Validate target
        # --------------------------------------------------

        if target not in df.columns:

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Target column '{target}' "
                    f"not found."
                )
            )

        # --------------------------------------------------
        # Generate EDA from RAW dataset
        # --------------------------------------------------

        eda_report_bytes = create_eda_report(
            df,
            target
        )

        # --------------------------------------------------
        # Separate X and y
        # --------------------------------------------------

        X = df.drop(
            columns=[target]
        )

        y = df[target]

        # --------------------------------------------------
        # Create processor
        # --------------------------------------------------

        processor = DataPreprocessor(
            target_col=target
        )

        # --------------------------------------------------
        # Detect task
        # --------------------------------------------------

        task = processor._detect_task(
            y
        )

        # --------------------------------------------------
        # Train/test split
        # --------------------------------------------------

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

        # --------------------------------------------------
        # FIT ONLY ON TRAIN
        # --------------------------------------------------

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
                    f"Pipeline fitting failed: "
                    f"{str(e)}"
                )
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
                    f"Test transformation failed: "
                    f"{str(e)}"
                )
            )

        # --------------------------------------------------
        # Create train output
        # --------------------------------------------------

        train_output = (
            X_train_processed.copy()
        )

        train_output[target] = (
            y_train.values
        )

        # --------------------------------------------------
        # Add IDs to train
        # --------------------------------------------------

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
        # Add IDs to test
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
        # Pipeline info
        # --------------------------------------------------

        info = processor.get_info()

        info["dataset_type"] = dataset_type
        info["rows_processed"] = len(df)

        info_text = create_pipeline_report(
            info
        )

        # --------------------------------------------------
        # Create ZIP
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

            zip_file.writestr(
                "eda_report.html",
                eda_report_bytes
            )

        zip_buffer.seek(0)

        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition":
                    "attachment; "
                    "filename=processed_dataset.zip"
            }
        )

    # ======================================================
    # TRAINING DATASET ONLY
    # ======================================================

    elif dataset_type == "Training Dataset":

        df = await read_csv_file(
            file,
            "Training Dataset"
        )

        # --------------------------------------------------
        # Validate target
        # --------------------------------------------------

        if target not in df.columns:

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Target column '{target}' "
                    f"not found in training dataset."
                )
            )

        # --------------------------------------------------
        # Generate EDA from RAW training dataset
        # --------------------------------------------------

        eda_report_bytes = create_eda_report(
            df,
            target
        )

        # --------------------------------------------------
        # Separate X and y
        # --------------------------------------------------

        X_train = df.drop(
            columns=[target]
        )

        y_train = df[target]

        # --------------------------------------------------
        # Create processor
        # --------------------------------------------------

        processor = DataPreprocessor(
            target_col=target
        )

        task = processor._detect_task(
            y_train
        )

        # --------------------------------------------------
        # Fit on entire uploaded training dataset
        # --------------------------------------------------

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
                    f"Pipeline fitting failed: "
                    f"{str(e)}"
                )
            )

        # --------------------------------------------------
        # Create output
        # --------------------------------------------------

        train_output = (
            X_train_processed.copy()
        )

        train_output[target] = (
            y_train.values
        )

        # --------------------------------------------------
        # Add IDs
        # --------------------------------------------------

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
        # Pipeline info
        # --------------------------------------------------

        info = processor.get_info()

        info["dataset_type"] = dataset_type
        info["rows_processed"] = len(df)

        info_text = create_pipeline_report(
            info
        )

        # --------------------------------------------------
        # Create ZIP
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
                "pipeline_info.txt",
                info_text
            )

            zip_file.writestr(
                "eda_report.html",
                eda_report_bytes
            )

        zip_buffer.seek(0)

        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition":
                    "attachment; "
                    "filename=processed_training_dataset.zip"
            }
        )

    # ======================================================
    # TEST DATASET
    # ======================================================

    else:

        # --------------------------------------------------
        # Both files are REQUIRED
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Read both datasets
        # --------------------------------------------------

        train_df = await read_csv_file(
            train_file,
            "Training Dataset"
        )

        test_df = await read_csv_file(
            test_file,
            "Test Dataset"
        )

        # --------------------------------------------------
        # Target must exist in TRAIN
        # --------------------------------------------------

        if target not in train_df.columns:

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Target column '{target}' "
                    f"not found in training dataset."
                )
            )

        # --------------------------------------------------
        # Generate EDA from TRAINING dataset
        #
        # The test dataset normally does not contain the
        # target, so the training dataset is used as the
        # primary EDA source.
        # --------------------------------------------------

        eda_report_bytes = create_eda_report(
            train_df,
            target
        )

        # --------------------------------------------------
        # Target should NOT be required in TEST
        # --------------------------------------------------

        X_train = train_df.drop(
            columns=[target]
        )

        y_train = train_df[target]

        # --------------------------------------------------
        # Test data
        # --------------------------------------------------

        X_test = test_df.copy()

        # If test happens to contain target,
        # remove it before transformation.

        if target in X_test.columns:

            X_test = X_test.drop(
                columns=[target]
            )

        # --------------------------------------------------
        # Create processor
        # --------------------------------------------------

        processor = DataPreprocessor(
            target_col=target
        )

        task = processor._detect_task(
            y_train
        )

        # --------------------------------------------------
        # FIT ONLY ON TRAINING DATA
        # --------------------------------------------------

        try:

            processor.fit_transform(
                X_train,
                y_train
            )

        except Exception as e:

            raise HTTPException(
                status_code=500,
                detail=(
                    f"Pipeline fitting on training "
                    f"data failed: {str(e)}"
                )
            )

        # --------------------------------------------------
        # Transform TEST using fitted processor
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
                    f"Test transformation failed: "
                    f"{str(e)}"
                )
            )

        # --------------------------------------------------
        # Add IDs back
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
        # Pipeline info
        # --------------------------------------------------

        info = processor.get_info()

        info["dataset_type"] = dataset_type
        info["rows_processed"] = len(test_df)

        info_text = create_pipeline_report(
            info
        )

        # --------------------------------------------------
        # Create ZIP
        # --------------------------------------------------

        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(
            zip_buffer,
            "w",
            zipfile.ZIP_DEFLATED
        ) as zip_file:

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

            zip_file.writestr(
                "eda_report.html",
                eda_report_bytes
            )

        zip_buffer.seek(0)

        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition":
                    "attachment; "
                    "filename=processed_test_dataset.zip"
            }
        )