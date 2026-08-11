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
import io
import zipfile

from sklearn.model_selection import (
    train_test_split
)

from pipeline import (
    DataPreprocessor
)


# ==========================================================
# APP
# ==========================================================

app = FastAPI(
    title="Auto Data Preprocessing API",
    description=(
        "Automated EDA-ready preprocessing, "
        "feature engineering and feature selection "
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
    # FEATURE SELECTION
    # ======================================================

    report.append("")

    report.append(
        "FEATURE SELECTION"
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
    # UNSUPERVISED
    # ======================================================

    if ml_task == "Unsupervised Learning":

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
        # Create target-free processor
        # --------------------------------------------------

        processor = DataPreprocessor(
            target_col=None,
            task="unsupervised"
        )

        # --------------------------------------------------
        # Fit + transform
        # --------------------------------------------------

        try:

            X_processed = (
                processor
                .fit_transform_unsupervised(
                    df
                )
            )

        except Exception as e:

            raise HTTPException(
                status_code=500,
                detail=(
                    "Unsupervised preprocessing "
                    f"failed: {str(e)}"
                )
            )

        # --------------------------------------------------
        # Add IDs back
        # --------------------------------------------------

        output = add_ids(
            X_processed,
            df,
            processor.id_cols
        )

        # --------------------------------------------------
        # Pipeline information
        # --------------------------------------------------

        info = processor.get_info()

        info["dataset_type"] = (
            "Unsupervised Dataset"
        )

        info["rows_processed"] = len(df)

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
                "X_processed.csv",
                output.to_csv(
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
                    "processed_unsupervised_dataset.zip"
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

        X = df.drop(
            columns=[target]
        )

        y = df[target]

        processor = DataPreprocessor(
            target_col=target
        )

        task = processor._detect_task(
            y
        )

        # --------------------------------------------------
        # Split
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
        # Fit
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
                    "Pipeline fitting failed: "
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
                    "Test transformation failed: "
                    f"{str(e)}"
                )
            )

        # --------------------------------------------------
        # Training output
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Test output
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Report
        # --------------------------------------------------

        info = processor.get_info()

        info["dataset_type"] = (
            dataset_type
        )

        info["rows_processed"] = len(df)

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

        processor = DataPreprocessor(
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

        X_test = test_df.copy()

        if target in X_test.columns:

            X_test = X_test.drop(
                columns=[target]
            )

        processor = DataPreprocessor(
            target_col=target
        )

        task = processor._detect_task(
            y_train
        )

        # --------------------------------------------------
        # Fit ONLY on training data
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
                    "Pipeline fitting on training "
                    f"data failed: {str(e)}"
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