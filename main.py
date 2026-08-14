from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
import pandas as pd
import io
import zipfile
from pathlib import Path

from pipeline_updated import (
    SupervisedPreprocessor,
    UnsupervisedPreprocessor,
    process_supervised_dataset,
)


app = FastAPI(
    title="Automated ML Data Pipeline API",
    version="2.1.0"
)


# ==========================================================
# HELPERS
# ==========================================================

async def read_csv_file(file, label="Dataset"):
    if file is None:
        raise HTTPException(
            status_code=400,
            detail=f"{label} was not uploaded."
        )

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail=f"{label} has no filename."
        )

    if Path(file.filename).suffix.lower() != ".csv":
        raise HTTPException(
            status_code=400,
            detail=f"{label} must be a CSV file."
        )

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read {label}: {e}"
        )

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail=f"{label} contains no rows."
        )

    return df


def create_pipeline_report(info):
    lines = [
        "AUTOMATED ML PREPROCESSING REPORT",
        "=" * 50,
        "",
    ]

    for key, value in info.items():
        lines.append(f"{key}: {value}")

    return "\n".join(lines)


def make_zip(files):
    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        "w",
        zipfile.ZIP_DEFLATED
    ) as z:
        for filename, data in files.items():
            z.writestr(filename, data)

    buffer.seek(0)
    return buffer


def zip_response(buffer, filename):
    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition":
                f'attachment; filename="{filename}"'
        }
    )


# ==========================================================
# HEALTH
# ==========================================================

@app.get("/")
def root():
    return {
        "status": "running",
        "service": "Automated ML Data Pipeline API"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


# ==========================================================
# MAIN PROCESSING ENDPOINT
# ==========================================================

@app.post("/process")
async def process_dataset(
    ml_task: str = Form(...),
    dataset_type: str = Form(...),

    file: UploadFile = File(None),
    train_file: UploadFile = File(None),
    test_file: UploadFile = File(None),

    target: str = Form(None),
    test_size: float = Form(0.20),
    random_state: int = Form(42)
):

    valid_tasks = [
        "Supervised Learning",
        "Unsupervised Learning"
    ]

    valid_dataset_types = [
        "Entire Dataset",
        "Training Dataset",
        "Test Dataset"
    ]

    if ml_task not in valid_tasks:
        raise HTTPException(
            status_code=400,
            detail="Invalid learning type."
        )

    if dataset_type not in valid_dataset_types:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid dataset type. Choose "
                "Entire Dataset, Training Dataset, "
                "or Test Dataset."
            )
        )

    if not 0 < test_size < 1:
        raise HTTPException(
            status_code=400,
            detail="test_size must be between 0 and 1."
        )

    supervised = ml_task == "Supervised Learning"

    if supervised and not target:
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
            "Complete Dataset"
        )

        if supervised:

            if target not in df.columns:
                raise HTTPException(
                    status_code=400,
                    detail=f"Target '{target}' not found."
                )

            try:
                result = process_supervised_dataset(
                    df=df,
                    target_col=target,
                    test_size=test_size,
                    random_state=random_state
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Supervised preprocessing failed: {e}"
                )

            train_output = result["X_train"]
            test_output = result["X_test"]
            info = result["info"]

        else:

            train_df = df.sample(
                frac=1 - test_size,
                random_state=random_state
            )
            test_df = df.drop(
                train_df.index
            )

            processor = UnsupervisedPreprocessor(
                test_size=test_size,
                random_state=random_state
            )

            try:
                train_output, train_ids = (
                    processor.fit_transform(
                        train_df
                    )
                )
                test_output, test_ids = (
                    processor.transform(
                        test_df
                    )
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Unsupervised preprocessing failed: {e}"
                )

            if not train_ids.empty:
                train_output = pd.concat(
                    [
                        train_ids.reset_index(drop=True),
                        train_output.reset_index(drop=True)
                    ],
                    axis=1
                )

            if not test_ids.empty:
                test_output = pd.concat(
                    [
                        test_ids.reset_index(drop=True),
                        test_output.reset_index(drop=True)
                    ],
                    axis=1
                )

            info = processor.get_info()
            info["task"] = "Unsupervised"

        info["dataset_type"] = dataset_type
        info["rows_processed"] = len(df)

        package = make_zip({
            "X_train.csv":
                train_output.to_csv(index=False),
            "X_test.csv":
                test_output.to_csv(index=False),
            "pipeline_info.txt":
                create_pipeline_report(info)
        })

        return zip_response(
            package,
            "processed_dataset.zip"
        )

    # ======================================================
    # TRAINING DATASET ONLY
    # ======================================================

    if dataset_type == "Training Dataset":

        train_df = await read_csv_file(
            file,
            "Training Dataset"
        )

        if supervised:

            if target not in train_df.columns:
                raise HTTPException(
                    status_code=400,
                    detail=f"Target '{target}' not found."
                )

            X_train = train_df.drop(
                columns=[target]
            )
            y_train = train_df[target]

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
                    detail=f"Supervised preprocessing failed: {e}"
                )

            train_output = (
                X_train_processed.reset_index(
                    drop=True
                )
            )

            ids = pd.DataFrame(
                index=X_train.index
            )

            for col in processor.id_cols:
                if col in X_train.columns:
                    ids[col] = X_train[col]

            if not ids.empty:
                train_output = pd.concat(
                    [
                        ids.reset_index(drop=True),
                        train_output
                    ],
                    axis=1
                )

            train_output[target] = (
                y_train.reset_index(drop=True)
            )

            info = processor.get_info()

        else:

            processor = UnsupervisedPreprocessor(
                test_size=test_size,
                random_state=random_state
            )

            try:
                train_output, train_ids = (
                    processor.fit_transform(
                        train_df
                    )
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Unsupervised preprocessing failed: {e}"
                )

            if not train_ids.empty:
                train_output = pd.concat(
                    [
                        train_ids.reset_index(drop=True),
                        train_output.reset_index(drop=True)
                    ],
                    axis=1
                )

            info = processor.get_info()
            info["task"] = "Unsupervised"

        info["dataset_type"] = dataset_type
        info["rows_processed"] = len(train_df)

        package = make_zip({
            "X_train.csv":
                train_output.to_csv(index=False),
            "pipeline_info.txt":
                create_pipeline_report(info)
        })

        return zip_response(
            package,
            "processed_training_dataset.zip"
        )

    # ======================================================
    # TRAINING + TEST DATASETS
    # ======================================================

    if train_file is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "A training dataset is required "
                "for the Training + Test workflow."
            )
        )

    if test_file is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "A test dataset is required "
                "for the Training + Test workflow."
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

    if supervised:

        if target not in train_df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Target '{target}' not found in training dataset."
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

        try:
            X_train_processed = (
                processor.fit_transform(
                    X_train,
                    y_train
                )
            )

            X_test_processed, test_ids = (
                processor.transform(
                    X_test
                )
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Supervised preprocessing failed: {e}"
            )

        train_output = (
            X_train_processed.reset_index(
                drop=True
            )
        )

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
                    train_output
                ],
                axis=1
            )

        train_output[target] = (
            y_train.reset_index(drop=True)
        )

        test_output = (
            X_test_processed.reset_index(
                drop=True
            )
        )

        if not test_ids.empty:
            test_output = pd.concat(
                [
                    test_ids.reset_index(drop=True),
                    test_output
                ],
                axis=1
            )

        info = processor.get_info()

    else:

        processor = UnsupervisedPreprocessor(
            test_size=test_size,
            random_state=random_state
        )

        try:
            train_output, train_ids = (
                processor.fit_transform(
                    train_df
                )
            )

            test_output, test_ids = (
                processor.transform(
                    test_df
                )
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Unsupervised preprocessing failed: {e}"
            )

        if not train_ids.empty:
            train_output = pd.concat(
                [
                    train_ids.reset_index(drop=True),
                    train_output.reset_index(drop=True)
                ],
                axis=1
            )

        if not test_ids.empty:
            test_output = pd.concat(
                [
                    test_ids.reset_index(drop=True),
                    test_output.reset_index(drop=True)
                ],
                axis=1
            )

        info = processor.get_info()
        info["task"] = "Unsupervised"

    info["dataset_type"] = dataset_type
    info["rows_processed"] = (
        len(train_df) + len(test_df)
    )

    package = make_zip({
        "X_train.csv":
            train_output.to_csv(index=False),
        "X_test.csv":
            test_output.to_csv(index=False),
        "pipeline_info.txt":
            create_pipeline_report(info)
    })

    return zip_response(
        package,
        "processed_train_test_dataset.zip"
    )


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
