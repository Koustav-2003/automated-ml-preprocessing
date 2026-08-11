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

from Supervised_pipeline import (
    SupervisedPreprocessor,
    process_supervised_dataset
)

from unsupervised_pipeline import (
    UnsupervisedPreprocessor,
    process_unsupervised_dataset
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

            processor = UnsupervisedPreprocessor(
                test_size=0.20,
                random_state=42
            )

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

            processor = UnsupervisedPreprocessor(
                test_size=0.20,
                random_state=42
            )

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
                        "feature_selection_method":
                            (
                                "Lasso"
                                if result["task"] ==
                                "regression"
                                else
                                "L1 Logistic Regression"
                            )
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

        X_test = test_df.copy()

        if target in X_test.columns:

            X_test = X_test.drop(
                columns=[target]
            )

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
        train_output[target] = y_train.values

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