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


@app.get("/")
def root():

    return {
        "message": "Auto Data Preprocessing API is running",
        "status": "OK"
    }


@app.post("/process")
async def process_dataset(
    file: UploadFile = File(...),
    target: str = Form(...)
):

    # -----------------------------------------
    # Validate file
    # -----------------------------------------

    if not file.filename.lower().endswith(".csv"):

        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported."
        )


    # -----------------------------------------
    # Read CSV
    # -----------------------------------------

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


    # -----------------------------------------
    # Validate target
    # -----------------------------------------

    if target not in df.columns:

        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target}' not found."
        )


    # -----------------------------------------
    # Separate X and y
    # -----------------------------------------

    X = df.drop(
        columns=[target]
    )

    y = df[target]


    # -----------------------------------------
    # Create processor
    # -----------------------------------------

    processor = DataPreprocessor(
        target_col=target
    )


    # -----------------------------------------
    # Detect task
    # -----------------------------------------

    task = processor._detect_task(y)


    # -----------------------------------------
    # Train/test split
    # -----------------------------------------

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


    # -----------------------------------------
    # FIT pipeline ONLY on training data
    # -----------------------------------------

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


    # -----------------------------------------
    # Transform test
    # -----------------------------------------

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


    # -----------------------------------------
    # Create training output
    # -----------------------------------------

    train_output = (
        X_train_processed.copy()
    )

    train_output[target] = (
        y_train.values
    )


    # -----------------------------------------
    # Add IDs back
    # -----------------------------------------

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


    if not test_ids.empty:

        X_test_processed = pd.concat(
            [
                test_ids.reset_index(drop=True),
                X_test_processed.reset_index(drop=True)
            ],
            axis=1
        )


    # -----------------------------------------
    # Create ZIP
    # -----------------------------------------

    zip_buffer = io.BytesIO()


    with zipfile.ZipFile(
        zip_buffer,
        "w",
        zipfile.ZIP_DEFLATED
    ) as zip_file:

        zip_file.writestr(
            "X_train.csv",
            train_output.to_csv(index=False)
        )

        zip_file.writestr(
            "X_test.csv",
            X_test_processed.to_csv(index=False)
        )


        # Pipeline information
        info = processor.get_info()

        info_text = "\n".join(
            f"{key}: {value}"
            for key, value in info.items()
        )

        zip_file.writestr(
            "pipeline_info.txt",
            info_text
        )


    zip_buffer.seek(0)


    # -----------------------------------------
    # Return ZIP
    # -----------------------------------------

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition":
                "attachment; "
                "filename=processed_dataset.zip"
        }
    )