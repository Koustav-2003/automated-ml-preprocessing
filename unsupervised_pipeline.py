import os
from datetime import datetime

import numpy as np
import pandas as pd

from scipy import stats

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class UnsupervisedPreprocessor:
    """
    Unsupervised preprocessing pipeline.

    Workflow
    --------
    1. Detect ID-like columns
    2. Split the complete dataset into train/test
    3. Fit all preprocessing parameters ONLY on X_train
    4. Create missing-value indicators
    5. Handle categorical missing values
    6. Handle numerical missing values using train medians
    7. Handle optional temporal features
    8. Group rare categorical labels
    9. Frequency-encode categorical features
    10. MinMax scaling
    11. Transform X_test using the fitted training parameters

    There is no target variable and therefore:
    - no target-guided encoding
    - no Lasso
    - no target-based feature selection

    Output
    ------
    outputs/
        X_train_<timestamp>.csv
        X_test_<timestamp>.csv
        pipeline_info_<timestamp>.txt
    """

    def __init__(
        self,
        test_size=0.20,
        random_state=42,
        rare_label_threshold=0.01,
        skew_threshold=0.75,
        scale_features=True,
    ):
        self.test_size = test_size
        self.random_state = random_state

        self.rare_label_threshold = rare_label_threshold
        self.skew_threshold = skew_threshold

        self.scale_features = scale_features

        self.id_cols = []

        self.features_with_nan = []
        self.features_nan_cat = []
        self.features_nan_num = []

        self.train_medians = {}
        self.frequent_labels = {}

        self.categorical_features = []

        self.frequency_mappings = {}
        self.frequency_fallbacks = {}

        self.skewed_features = []
        self.skew_shifts = {}

        self.feature_columns = []
        self.scalable_features = []

        self.scaler = None

        self.fitted = False

    # ======================================================
    # ID DETECTION
    # ======================================================

    def _detect_ids(self, X):

        self.id_cols = []

        for col in X.columns:

            if X[col].nunique(
                dropna=False
            ) == len(X):

                self.id_cols.append(col)

    # ======================================================
    # CATEGORICAL DETECTION
    # ======================================================

    def _get_categorical_features(self, X):

        return [
            col
            for col in X.columns
            if (
                pd.api.types.is_object_dtype(X[col])
                or pd.api.types.is_string_dtype(X[col])
                or pd.api.types.is_categorical_dtype(X[col])
            )
        ]

    # ======================================================
    # MISSING VALUE DETECTION
    # ======================================================

    def _detect_missing_features(self, X):

        self.features_with_nan = [
            col
            for col in X.columns
            if X[col].isnull().sum() > 0
        ]

        self.features_nan_cat = [
            col
            for col in self.features_with_nan
            if col in self._get_categorical_features(X)
        ]

        self.features_nan_num = [
            col
            for col in self.features_with_nan
            if col not in self.features_nan_cat
        ]

    def _add_missing_indicators(self, X):

        X = X.copy()

        for feature in self.features_with_nan:

            if feature in X.columns:

                X[feature + "_nan"] = np.where(
                    X[feature].isnull(),
                    1,
                    0
                )

        return X

    # ======================================================
    # IMPUTATION
    # ======================================================

    def _fit_imputation(self, X):

        self.train_medians = {}

        for feature in self.features_nan_num:

            if feature in X.columns:

                self.train_medians[feature] = (
                    X[feature].median()
                )

    def _apply_imputation(self, X):

        X = X.copy()

        for feature in self.features_nan_cat:

            if feature in X.columns:

                X[feature] = (
                    X[feature]
                    .fillna("Missing")
                )

        for feature, median_value in self.train_medians.items():

            if feature in X.columns:

                X[feature] = (
                    X[feature]
                    .fillna(median_value)
                )

        return X

    # ======================================================
    # RARE LABELS
    # ======================================================

    def _fit_rare_labels(self, X):

        self.categorical_features = (
            self._get_categorical_features(X)
        )

        self.frequent_labels = {}

        for feature in self.categorical_features:

            frequencies = (
                X[feature]
                .value_counts(normalize=True)
            )

            self.frequent_labels[feature] = (
                frequencies[
                    frequencies > self.rare_label_threshold
                ].index.tolist()
            )

    def _apply_rare_labels(self, X):

        X = X.copy()

        for feature in self.categorical_features:

            if feature not in X.columns:
                continue

            kept = self.frequent_labels.get(
                feature,
                []
            )

            X[feature] = np.where(
                X[feature].isin(kept),
                X[feature],
                "Rare_var"
            )

        return X

    # ======================================================
    # SKEWNESS
    # ======================================================

    def _fit_skewness(self, X):

        self.skewed_features = []
        self.skew_shifts = {}

        numeric_features = [
            col
            for col in X.columns
            if pd.api.types.is_numeric_dtype(X[col])
        ]

        numeric_features = [
            col
            for col in numeric_features
            if not col.endswith("_nan")
        ]

        for feature in numeric_features:

            values = X[feature].dropna()

            if len(values) < 3:
                continue

            skew_value = stats.skew(values)

            if not np.isfinite(skew_value):
                continue

            if abs(skew_value) <= self.skew_threshold:
                continue

            min_value = values.min()

            if min_value <= 0:

                self.skew_shifts[feature] = (
                    abs(min_value) + 1
                )

            self.skewed_features.append(feature)

    def _apply_skewness(self, X):

        X = X.copy()

        for feature in self.skewed_features:

            if feature not in X.columns:
                continue

            values = X[feature]

            shift = self.skew_shifts.get(
                feature,
                0
            )

            if shift:

                values = values + shift

            X[feature] = np.log1p(
                values
            )

        return X

    # ======================================================
    # FREQUENCY ENCODING
    # ======================================================

    def _fit_frequency_encoding(self, X):

        self.frequency_mappings = {}
        self.frequency_fallbacks = {}

        for feature in self.categorical_features:

            if feature not in X.columns:
                continue

            frequencies = (
                X[feature]
                .value_counts(normalize=True)
            )

            self.frequency_mappings[feature] = (
                frequencies.to_dict()
            )

            # Unseen categories in future data receive
            # the smallest observed training frequency.
            self.frequency_fallbacks[feature] = (
                float(frequencies.min())
                if not frequencies.empty
                else 0.0
            )

    def _apply_frequency_encoding(self, X):

        X = X.copy()

        for feature in self.categorical_features:

            if feature not in X.columns:
                continue

            mapping = (
                self.frequency_mappings
                .get(feature, {})
            )

            fallback = (
                self.frequency_fallbacks
                .get(feature, 0.0)
            )

            X[feature] = (
                X[feature]
                .map(mapping)
                .fillna(fallback)
            )

        return X

    # ======================================================
    # NUMERIC VALIDATION
    # ======================================================

    def _ensure_numeric(self, X):

        X = X.copy()

        remaining_categorical = (
            self._get_categorical_features(X)
        )

        if remaining_categorical:

            raise ValueError(
                "Categorical columns remain after "
                "frequency encoding: "
                + ", ".join(remaining_categorical)
            )

        return X

    # ======================================================
    # FEATURE ALIGNMENT
    # ======================================================

    def _align_features(self, X):

        X = X.copy()

        for feature in self.feature_columns:

            if feature not in X.columns:

                X[feature] = 0

        extra_features = [
            col
            for col in X.columns
            if col not in self.feature_columns
        ]

        if extra_features:

            X = X.drop(
                columns=extra_features
            )

        return X[
            self.feature_columns
        ]

    # ======================================================
    # SCALING
    # ======================================================

    def _fit_scaler(self, X):

        self.scalable_features = [
            col
            for col in X.columns
            if pd.api.types.is_numeric_dtype(
                X[col]
            )
        ]

        if not self.scale_features:

            self.scaler = None
            return

        self.scaler = MinMaxScaler()

        if self.scalable_features:

            self.scaler.fit(
                X[self.scalable_features]
            )

    def _apply_scaling(self, X):

        X = X.copy()

        if (
            not self.scale_features
            or self.scaler is None
            or not self.scalable_features
        ):

            return X

        X[self.scalable_features] = (
            self.scaler.transform(
                X[self.scalable_features]
            )
        )

        return X

    # ======================================================
    # FIT
    # ======================================================

    def fit(self, X_train):

        X_train = X_train.copy()

        # --------------------------------------------------
        # IDs
        # --------------------------------------------------

        self._detect_ids(
            X_train
        )

        X = X_train.drop(
            columns=self.id_cols,
            errors="ignore"
        )

        # --------------------------------------------------
        # Missing values
        # --------------------------------------------------

        self._detect_missing_features(
            X
        )

        X = self._add_missing_indicators(
            X
        )

        self._fit_imputation(
            X
        )

        X = self._apply_imputation(
            X
        )

        # --------------------------------------------------
        # Rare categories
        # --------------------------------------------------

        self._fit_rare_labels(
            X
        )

        X = self._apply_rare_labels(
            X
        )

        # --------------------------------------------------
        # Skewness
        # --------------------------------------------------

        self._fit_skewness(
            X
        )

        X = self._apply_skewness(
            X
        )

        # --------------------------------------------------
        # Frequency encoding
        # --------------------------------------------------

        self._fit_frequency_encoding(
            X
        )

        X = self._apply_frequency_encoding(
            X
        )

        # --------------------------------------------------
        # Numeric validation
        # --------------------------------------------------

        X = self._ensure_numeric(
            X
        )

        self.feature_columns = (
            X.columns.tolist()
        )

        # --------------------------------------------------
        # Scaling
        # --------------------------------------------------

        self._fit_scaler(
            X
        )

        self.fitted = True

        return self

    # ======================================================
    # TRANSFORM
    # ======================================================

    def transform(self, X):

        if not self.fitted:

            raise RuntimeError(
                "Pipeline has not been fitted yet."
            )

        X = X.copy()

        ids = pd.DataFrame(
            index=X.index
        )

        # --------------------------------------------------
        # Preserve IDs
        # --------------------------------------------------

        for col in self.id_cols:

            if col in X.columns:

                ids[col] = X[col]

        # --------------------------------------------------
        # Remove IDs
        # --------------------------------------------------

        X = X.drop(
            columns=self.id_cols,
            errors="ignore"
        )

        # --------------------------------------------------
        # Missing indicators
        # --------------------------------------------------

        X = self._add_missing_indicators(
            X
        )

        # --------------------------------------------------
        # Imputation
        # --------------------------------------------------

        X = self._apply_imputation(
            X
        )

        # --------------------------------------------------
        # Rare labels
        # --------------------------------------------------

        X = self._apply_rare_labels(
            X
        )

        # --------------------------------------------------
        # Skewness
        # --------------------------------------------------

        X = self._apply_skewness(
            X
        )

        # --------------------------------------------------
        # Frequency encoding
        # --------------------------------------------------

        X = self._apply_frequency_encoding(
            X
        )

        # --------------------------------------------------
        # Feature alignment
        # --------------------------------------------------

        X = self._align_features(
            X
        )

        # --------------------------------------------------
        # Numeric validation
        # --------------------------------------------------

        X = self._ensure_numeric(
            X
        )

        # --------------------------------------------------
        # Scaling
        # --------------------------------------------------

        X = self._apply_scaling(
            X
        )

        return X, ids

    def fit_transform(self, X_train):

        self.fit(
            X_train
        )

        X_processed, ids = (
            self.transform(
                X_train
            )
        )

        return X_processed, ids

    # ======================================================
    # INFORMATION
    # ======================================================

    def get_info(self):

        if not self.fitted:

            raise RuntimeError(
                "Pipeline has not been fitted yet."
            )

        return {
            "pipeline": "unsupervised",
            "test_size": self.test_size,
            "random_state": self.random_state,
            "id_columns": self.id_cols,
            "missing_value_features": self.features_with_nan,
            "numeric_missing_features": self.features_nan_num,
            "categorical_missing_features": self.features_nan_cat,
            "skewed_features": self.skewed_features,
            "categorical_features": self.categorical_features,
            "scaled_features": self.scalable_features,
            "feature_count_before_processing": (
                len(self.feature_columns)
            ),
            "final_feature_count": (
                len(self.feature_columns)
            ),
        }


# ==========================================================
# PROCESS COMPLETE DATASET
# ==========================================================

def process_unsupervised_dataset(
    df,
    test_size=0.20,
    random_state=42,
    output_dir=None,
):
    """
    Split a complete dataset into train/test and run the
    unsupervised preprocessing pipeline.

    No target variable is used.

    Returns
    -------
    result : dict
        Contains processed train/test DataFrames,
        output paths and pipeline information.
    """

    if not isinstance(df, pd.DataFrame):

        raise TypeError(
            "df must be a pandas DataFrame."
        )

    if not 0 < test_size < 1:

        raise ValueError(
            "test_size must be between 0 and 1."
        )

    if len(df) < 2:

        raise ValueError(
            "Dataset must contain at least 2 rows."
        )

    X = df.copy()

    processor = UnsupervisedPreprocessor(
        test_size=test_size,
        random_state=random_state,
    )

    # ------------------------------------------------------
    # Split BEFORE fitting preprocessing
    # ------------------------------------------------------

    X_train, X_test = train_test_split(
        X,
        test_size=test_size,
        random_state=random_state,
    )

    # ------------------------------------------------------
    # Fit ONLY on X_train
    # ------------------------------------------------------

    X_train_processed, train_ids = (
        processor.fit_transform(
            X_train
        )
    )

    # ------------------------------------------------------
    # Transform X_test using fitted parameters
    # ------------------------------------------------------

    X_test_processed, test_ids = (
        processor.transform(
            X_test
        )
    )

    # ------------------------------------------------------
    # Build training output
    # ------------------------------------------------------

    train_output = X_train_processed.copy()

    if not train_ids.empty:

        train_output = pd.concat(
            [
                train_ids.reset_index(
                    drop=True
                ),
                train_output.reset_index(
                    drop=True
                ),
            ],
            axis=1,
        )

    # ------------------------------------------------------
    # Build test output
    # ------------------------------------------------------

    test_output = X_test_processed.copy()

    if not test_ids.empty:

        test_output = pd.concat(
            [
                test_ids.reset_index(
                    drop=True
                ),
                test_output.reset_index(
                    drop=True
                ),
            ],
            axis=1,
        )

    # ------------------------------------------------------
    # Output folder
    # ------------------------------------------------------

    if output_dir is None:

        output_dir = os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)
            ),
            "outputs"
        )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    train_path = os.path.join(
        output_dir,
        f"X_train_{timestamp}.csv"
    )

    test_path = os.path.join(
        output_dir,
        f"X_test_{timestamp}.csv"
    )

    info_path = os.path.join(
        output_dir,
        f"pipeline_info_{timestamp}.txt"
    )

    train_output.to_csv(
        train_path,
        index=False
    )

    test_output.to_csv(
        test_path,
        index=False
    )

    # ------------------------------------------------------
    # Pipeline information
    # ------------------------------------------------------

    info = processor.get_info()

    with open(
        info_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "UNSUPERVISED PREPROCESSING PIPELINE\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        file.write(
            f"Generated: "
            f"{datetime.now().isoformat()}\n"
        )

        file.write(
            f"Test size: {test_size}\n"
        )

        file.write(
            f"Random state: {random_state}\n\n"
        )

        file.write(
            f"Original dataset shape: "
            f"{df.shape}\n"
        )

        file.write(
            f"X_train shape before processing: "
            f"{X_train.shape}\n"
        )

        file.write(
            f"X_test shape before processing: "
            f"{X_test.shape}\n"
        )

        file.write(
            f"X_train shape after processing: "
            f"{train_output.shape}\n"
        )

        file.write(
            f"X_test shape after processing: "
            f"{test_output.shape}\n\n"
        )

        file.write(
            f"ID columns: "
            f"{processor.id_cols}\n\n"
        )

        file.write(
            f"Missing-value features: "
            f"{processor.features_with_nan}\n\n"
        )

        file.write(
            f"Categorical features frequency-encoded: "
            f"{processor.categorical_features}\n\n"
        )

        file.write(
            f"Skewed features: "
            f"{processor.skewed_features}\n\n"
        )

        file.write(
            f"Final feature count: "
            f"{len(processor.feature_columns)}\n"
        )

    return {
        "processor": processor,
        "X_train": train_output,
        "X_test": test_output,
        "train_path": train_path,
        "test_path": test_path,
        "pipeline_info_path": info_path,
        "timestamp": timestamp,
        "info": info,
    }


# ==========================================================
# LOCAL TEST
# ==========================================================

if __name__ == "__main__":

    INPUT_FILE = "train.csv"

    print("=" * 70)
    print("UNSUPERVISED PREPROCESSING PIPELINE")
    print("=" * 70)

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Dataset shape: {df.shape}"
    )

    result = process_unsupervised_dataset(
        df=df,
        test_size=0.20,
        random_state=42,
    )

    print(
        f"\nX_train output: "
        f"{result['X_train'].shape}"
    )

    print(
        f"X_test output: "
        f"{result['X_test'].shape}"
    )

    print(
        "\nSaved files:"
    )

    print(
        result["train_path"]
    )

    print(
        result["test_path"]
    )

    print(
        result["pipeline_info_path"]
    )
