import os
from datetime import datetime

import numpy as np
import pandas as pd

from scipy import stats

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split


class SupervisedPreprocessor:
    """
    Supervised preprocessing pipeline.

    Workflow
    --------
    1. Detect ID-like columns
    2. Split the complete dataset into train/test
    3. Fit all preprocessing parameters ONLY on training data
    4. Create missing-value indicators
    5. Handle categorical missing values
    6. Handle numerical missing values using train medians
    7. Group rare categorical labels
    8. Handle skewed numerical features
    9. Target-guided categorical encoding
    10. MinMax scaling
    11. L1-based feature selection
    12. Transform X_test using the fitted training parameters

    Supported
    ---------
    - Regression
    - Classification

    Output
    ------
    outputs/
        X_train_<timestamp>.csv
        X_test_<timestamp>.csv
        pipeline_info_<timestamp>.txt
    """

    def __init__(
        self,
        target_col,
        test_size=0.20,
        random_state=42,
        rare_label_threshold=0.01,
        skew_threshold=0.75,
        lasso_alpha=0.0005,
        logistic_c=1.0,
        scale_features=True,
    ):
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state

        self.rare_label_threshold = rare_label_threshold
        self.skew_threshold = skew_threshold

        self.lasso_alpha = lasso_alpha
        self.logistic_c = logistic_c
        self.scale_features = scale_features

        self.task = None

        self.id_cols = []

        self.features_with_nan = []
        self.features_nan_cat = []
        self.features_nan_num = []

        self.train_medians = {}
        self.frequent_labels = {}

        self.categorical_features = []
        self.label_mappings = {}
        self.global_target_mean = None

        self.skewed_features = []
        self.skew_shifts = {}

        self.feature_columns = []
        self.scalable_features = []

        self.scaler = None

        self.feature_selector = None
        self.selected_features = []

        self.fitted = False

    # ======================================================
    # TASK DETECTION
    # ======================================================

    def detect_task(self, y):
        """
        Automatically detect regression vs classification.

        Heuristic:
            object/string -> classification
            bool -> classification
            numeric with <= 20 unique values -> classification
            otherwise -> regression
        """

        if (
            pd.api.types.is_object_dtype(y)
            or pd.api.types.is_string_dtype(y)
        ):
            return "classification"

        if pd.api.types.is_bool_dtype(y):
            return "classification"

        if pd.api.types.is_numeric_dtype(y):

            if y.nunique(dropna=True) <= 20:
                return "classification"

            return "regression"

        return "classification"

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
    # ID DETECTION
    # ======================================================

    def _detect_ids(self, X):
        self.id_cols = []

        for col in X.columns:

            if X[col].nunique(dropna=False) == len(X):
                self.id_cols.append(col)

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
            if (
                pd.api.types.is_object_dtype(X[col])
                or pd.api.types.is_string_dtype(X[col])
                or pd.api.types.is_categorical_dtype(X[col])
            )
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

            if abs(skew_value) <= self.skew_threshold:
                continue

            min_value = values.min()

            if min_value <= 0:

                shift = abs(min_value) + 1

                self.skew_shifts[feature] = shift

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

            X[feature] = np.log1p(values)

        return X

    # ======================================================
    # TARGET GUIDED ENCODING
    # ======================================================

    def _fit_target_encoding(self, X, y):

        self.label_mappings = {}

        # Global fallback for unseen categories
        if pd.api.types.is_numeric_dtype(y):

            self.global_target_mean = float(
                y.mean()
            )

        else:

            self.global_target_mean = float(
                y.map(
                    y.value_counts(normalize=True)
                ).mean()
            )

        for feature in self.categorical_features:

            if feature not in X.columns:
                continue

            temp = pd.DataFrame({
                "feature": X[feature],
                "target": y.values
            })

            means = (
                temp
                .groupby("feature")["target"]
                .mean()
            )

            ordered_labels = (
                means
                .sort_values()
                .index
                .tolist()
            )

            mapping = {
                label: index
                for index, label
                in enumerate(
                    ordered_labels,
                    start=0
                )
            }

            self.label_mappings[feature] = mapping

    def _apply_target_encoding_train(self, X, y):

        X = X.copy()

        for feature in self.categorical_features:

            if feature not in X.columns:
                continue

            mapping = self.label_mappings.get(
                feature,
                {}
            )

            X[feature] = (
                X[feature]
                .map(mapping)
            )

            # This should only happen in unusual cases.
            X[feature] = (
                X[feature]
                .fillna(
                    X[feature].median()
                    if X[feature].notna().any()
                    else 0
                )
            )

        return X

    def _apply_target_encoding_test(self, X):

        X = X.copy()

        for feature in self.categorical_features:

            if feature not in X.columns:
                continue

            mapping = self.label_mappings.get(
                feature,
                {}
            )

            X[feature] = (
                X[feature]
                .map(mapping)
            )

            X[feature] = (
                X[feature]
                .fillna(
                    self._encoded_train_median(
                        feature
                    )
                )
            )

        return X

    def _encoded_train_median(self, feature):

        mapping = self.label_mappings.get(
            feature,
            {}
        )

        if not mapping:
            return 0

        return float(
            np.median(
                list(mapping.values())
            )
        )

    # ======================================================
    # NUMERIC VALIDATION
    # ======================================================

    def _ensure_numeric(self, X):

        X = X.copy()

        categorical_remaining = (
            self._get_categorical_features(X)
        )

        if categorical_remaining:

            raise ValueError(
                "Categorical columns remain after "
                "target encoding: "
                + ", ".join(categorical_remaining)
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

        return X[self.feature_columns]

    # ======================================================
    # SCALING
    # ======================================================

    def _fit_scaler(self, X):

        self.scalable_features = [
            col
            for col in X.columns
            if pd.api.types.is_numeric_dtype(X[col])
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
    # FEATURE SELECTION
    # ======================================================

    def _fit_feature_selector(self, X, y):

        if self.task == "regression":

            estimator = Lasso(
                alpha=self.lasso_alpha,
                max_iter=10000
            )

        elif self.task == "classification":

            estimator = LogisticRegression(
                penalty="l1",
                solver="liblinear",
                C=self.logistic_c,
                max_iter=5000
            )

        else:

            raise ValueError(
                f"Unsupported task: {self.task}"
            )

        self.feature_selector = (
            SelectFromModel(
                estimator=estimator,
                threshold="mean"
            )
        )

        self.feature_selector.fit(
            X,
            y
        )

        support = (
            self.feature_selector
            .get_support()
        )

        self.selected_features = [
            feature
            for feature, selected
            in zip(
                X.columns,
                support
            )
            if selected
        ]

        if not self.selected_features:

            # Safety fallback: retain all features rather
            # than returning an unusable zero-column dataset.
            self.selected_features = (
                X.columns.tolist()
            )

    def _apply_feature_selection(self, X):

        X = X.copy()

        for feature in self.selected_features:

            if feature not in X.columns:
                X[feature] = 0

        return X[self.selected_features]

    # ======================================================
    # FIT
    # ======================================================

    def fit(self, X_train, y_train):

        if y_train is None:
            raise ValueError(
                "Target values are required for "
                "supervised preprocessing."
            )

        X_train = X_train.copy()
        y_train = pd.Series(
            y_train,
            index=X_train.index
        )

        self.task = self.detect_task(
            y_train
        )

        # --------------------------------------------------
        # IDs
        # --------------------------------------------------

        self._detect_ids(X_train)

        X = X_train.drop(
            columns=self.id_cols,
            errors="ignore"
        )

        # --------------------------------------------------
        # Missing values
        # --------------------------------------------------

        self._detect_missing_features(X)

        X = self._add_missing_indicators(X)

        self._fit_imputation(X)

        X = self._apply_imputation(X)

        # --------------------------------------------------
        # Rare categories
        # --------------------------------------------------

        self._fit_rare_labels(X)

        X = self._apply_rare_labels(X)

        # --------------------------------------------------
        # Skewness
        # --------------------------------------------------

        self._fit_skewness(X)

        X = self._apply_skewness(X)

        # --------------------------------------------------
        # Target encoding
        # --------------------------------------------------

        self._fit_target_encoding(
            X,
            y_train
        )

        X = self._apply_target_encoding_train(
            X,
            y_train
        )

        # --------------------------------------------------
        # Numeric validation
        # --------------------------------------------------

        X = self._ensure_numeric(X)

        self.feature_columns = (
            X.columns.tolist()
        )

        # --------------------------------------------------
        # Scaling
        # --------------------------------------------------

        self._fit_scaler(X)

        X = self._apply_scaling(X)

        # --------------------------------------------------
        # Feature selection
        # --------------------------------------------------

        self._fit_feature_selector(
            X,
            y_train
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

        X = self._add_missing_indicators(X)

        # --------------------------------------------------
        # Imputation
        # --------------------------------------------------

        X = self._apply_imputation(X)

        # --------------------------------------------------
        # Rare labels
        # --------------------------------------------------

        X = self._apply_rare_labels(X)

        # --------------------------------------------------
        # Skewness
        # --------------------------------------------------

        X = self._apply_skewness(X)

        # --------------------------------------------------
        # Target encoding
        # --------------------------------------------------

        X = self._apply_target_encoding_test(X)

        # --------------------------------------------------
        # Align feature structure
        # --------------------------------------------------

        X = self._align_features(X)

        # --------------------------------------------------
        # Numeric validation
        # --------------------------------------------------

        X = self._ensure_numeric(X)

        # --------------------------------------------------
        # Scaling
        # --------------------------------------------------

        X = self._apply_scaling(X)

        # --------------------------------------------------
        # Feature selection
        # --------------------------------------------------

        X = self._apply_feature_selection(X)

        return X, ids

    def fit_transform(self, X_train, y_train):

        self.fit(
            X_train,
            y_train
        )

        X_processed = self._transform_train(
            X_train
        )

        return X_processed

    def _transform_train(self, X):

        X = X.copy()

        X = X.drop(
            columns=self.id_cols,
            errors="ignore"
        )

        X = self._add_missing_indicators(X)

        X = self._apply_imputation(X)

        X = self._apply_rare_labels(X)

        X = self._apply_skewness(X)

        X = self._apply_target_encoding_train(
            X,
            pd.Series(
                index=X.index,
                dtype=float
            )
        )

        # The call above does not need y because the mapping was
        # already learned in fit(). Re-apply using the mapping.
        # Replace any NaNs generated by mapping with safe medians.
        X = self._apply_target_encoding_test(X)

        X = self._align_features(X)

        X = self._ensure_numeric(X)

        X = self._apply_scaling(X)

        X = self._apply_feature_selection(X)

        return X

    # ======================================================
    # INFORMATION
    # ======================================================

    def get_info(self):

        if not self.fitted:
            raise RuntimeError(
                "Pipeline has not been fitted yet."
            )

        return {
            "pipeline": "supervised",
            "task": self.task,
            "target": self.target_col,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "id_columns": self.id_cols,
            "missing_value_features": self.features_with_nan,
            "numeric_missing_features": self.features_nan_num,
            "categorical_missing_features": self.features_nan_cat,
            "skewed_features": self.skewed_features,
            "scaled_features": self.scalable_features,
            "original_feature_count": len(
                self.feature_columns
            ),
            "selected_feature_count": len(
                self.selected_features
            ),
            "selected_features": self.selected_features,
        }


# ==========================================================
# PROCESS COMPLETE DATASET
# ==========================================================

def process_supervised_dataset(
    df,
    target_col,
    test_size=0.20,
    random_state=42,
    output_dir=None,
):
    """
    Split a complete labelled dataset into train/test and run
    the supervised preprocessing pipeline.

    Returns
    -------
    result : dict
        Contains processed train/test DataFrames, task information,
        output paths and pipeline information.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found."
        )

    if not 0 < test_size < 1:
        raise ValueError(
            "test_size must be between 0 and 1."
        )

    if df[target_col].isnull().any():
        raise ValueError(
            "Target column contains missing values. "
            "Please handle missing target values before processing."
        )

    X = df.drop(
        columns=[target_col]
    )

    y = df[target_col]

    processor = SupervisedPreprocessor(
        target_col=target_col,
        test_size=test_size,
        random_state=random_state,
    )

    task = processor.detect_task(y)

    # ------------------------------------------------------
    # Split
    # ------------------------------------------------------

    if task == "classification":

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=y,
            )
        )

    else:

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
            )
        )

    # ------------------------------------------------------
    # Fit ONLY on training data
    # ------------------------------------------------------

    X_train_processed = processor.fit_transform(
        X_train,
        y_train
    )

    # ------------------------------------------------------
    # Transform test using train-fitted parameters
    # ------------------------------------------------------

    X_test_processed, test_ids = processor.transform(
        X_test
    )

    # ------------------------------------------------------
    # Build training output
    # ------------------------------------------------------

    train_output = X_train_processed.copy()

    train_output[target_col] = (
        y_train.values
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
                train_output.reset_index(drop=True),
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
                test_ids.reset_index(drop=True),
                test_output.reset_index(drop=True),
            ],
            axis=1,
        )

    # ------------------------------------------------------
    # Output folder
    # ------------------------------------------------------

    if output_dir is None:

        output_dir = (
            os.path.join(
                os.path.dirname(
                    os.path.abspath(__file__)
                ),
                "outputs"
            )
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
            "SUPERVISED PREPROCESSING PIPELINE\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        file.write(
            f"Generated: "
            f"{datetime.now().isoformat()}\n"
        )

        file.write(
            f"Task: {task}\n"
        )

        file.write(
            f"Target: {target_col}\n"
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
            f"Skewed features: "
            f"{processor.skewed_features}\n\n"
        )

        file.write(
            f"Original processed feature count: "
            f"{len(processor.feature_columns)}\n"
        )

        file.write(
            f"Selected feature count: "
            f"{len(processor.selected_features)}\n\n"
        )

        file.write(
            "Selected features:\n"
        )

        for feature in processor.selected_features:

            file.write(
                f" - {feature}\n"
            )

    return {
        "processor": processor,
        "task": task,
        "X_train": train_output,
        "X_test": test_output,
        "y_train": y_train,
        "y_test": y_test,
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
    TARGET_COL = "SalePrice"

    print("=" * 70)
    print("SUPERVISED PREPROCESSING PIPELINE")
    print("=" * 70)

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Dataset shape: {df.shape}"
    )

    result = process_supervised_dataset(
        df=df,
        target_col=TARGET_COL,
        test_size=0.20,
        random_state=42,
    )

    print(
        f"\nDetected task: {result['task']}"
    )

    print(
        f"X_train output: "
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
