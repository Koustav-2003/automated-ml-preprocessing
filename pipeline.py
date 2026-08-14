"""
Automated ML preprocessing pipeline.

IMPORTANT:
    This module performs preprocessing and feature engineering only.
    It intentionally performs NO feature selection.

Supported:
    - Supervised classification
    - Supervised regression
    - Unsupervised preprocessing

The pipeline is leakage-safe:
    - supervised train data is fitted first
    - test data is transformed using the fitted train preprocessing
    - target encoding, when used, is fitted without using the test set
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")


# ==========================================================
# HELPERS
# ==========================================================

def _safe_numeric_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=np.number).columns.tolist()


def _safe_categorical_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()


def _make_ohe() -> OneHotEncoder:
    """
    sklearn compatibility helper.
    """
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )


# ==========================================================
# ID DETECTION
# ==========================================================

def detect_id_columns(
    df: pd.DataFrame,
    uniqueness_threshold: float = 0.95
) -> List[str]:
    """
    Detect likely identifier columns.

    These columns are excluded from model features but are returned
    separately so they can be retained alongside processed output.
    """

    id_cols = []

    for col in df.columns:

        nunique = df[col].nunique(dropna=False)
        ratio = nunique / max(len(df), 1)

        name = str(col).lower()

        looks_like_id = (
            name == "id"
            or name.endswith("_id")
            or name.endswith("id")
            or "identifier" in name
        )

        highly_unique = (
            ratio >= uniqueness_threshold
            and nunique > 10
        )

        if looks_like_id or highly_unique:
            id_cols.append(col)

    return id_cols


# ==========================================================
# RARE CATEGORY HANDLING
# ==========================================================

class RareCategoryGrouper(
    BaseEstimator,
    TransformerMixin
):

    def __init__(
        self,
        min_frequency: float = 0.01,
        replacement: str = "__RARE__"
    ):
        self.min_frequency = min_frequency
        self.replacement = replacement

    def fit(self, X, y=None):

        X = pd.DataFrame(X).copy()

        self.columns_ = X.columns.tolist()
        self.frequent_categories_ = {}

        for col in self.columns_:

            counts = (
                X[col]
                .value_counts(
                    normalize=True,
                    dropna=True
                )
            )

            self.frequent_categories_[col] = set(
                counts[
                    counts >= self.min_frequency
                ].index
            )

        return self

    def transform(self, X):

        X = pd.DataFrame(
            X,
            columns=self.columns_
        ).copy()

        for col in self.columns_:

            allowed = self.frequent_categories_.get(
                col,
                set()
            )

            X[col] = X[col].where(
                X[col].isin(allowed),
                self.replacement
            )

        return X


# ==========================================================
# SKEW / LOG TRANSFORMATION
# ==========================================================

class SkewedLogTransformer(
    BaseEstimator,
    TransformerMixin
):

    def __init__(
        self,
        skew_threshold: float = 1.0
    ):
        self.skew_threshold = skew_threshold

    def fit(self, X, y=None):

        X = pd.DataFrame(X).copy()

        self.columns_ = X.columns.tolist()
        self.log_columns_ = []

        for col in self.columns_:

            series = pd.to_numeric(
                X[col],
                errors="coerce"
            )

            if series.dropna().empty:
                continue

            skew = series.skew()

            if pd.notna(skew) and abs(skew) > self.skew_threshold:

                # Log1p is valid for non-negative values.
                # For negative values, shift by the train minimum.
                minimum = series.min()

                if pd.notna(minimum):

                    self.log_columns_.append(col)

        self.shifts_ = {}

        for col in self.log_columns_:

            series = pd.to_numeric(
                X[col],
                errors="coerce"
            )

            minimum = series.min()

            self.shifts_[col] = (
                -minimum
                if pd.notna(minimum) and minimum < 0
                else 0.0
            )

        return self

    def transform(self, X):

        X = pd.DataFrame(
            X,
            columns=self.columns_
        ).copy()

        for col in self.log_columns_:

            values = pd.to_numeric(
                X[col],
                errors="coerce"
            )

            shift = self.shifts_.get(
                col,
                0.0
            )

            values = values + shift

            # Numerical safety.
            values = values.clip(
                lower=0
            )

            X[col] = np.log1p(values)

        return X


# ==========================================================
# TARGET ENCODING
# ==========================================================

class LeaveOneOutTargetEncoder(
    BaseEstimator,
    TransformerMixin
):
    """
    Leave-one-out target encoding for supervised categorical features.

    During fit_transform on training data, each training row excludes its
    own target from the category statistic.

    During transform (e.g. test data), learned category statistics are used.
    """

    def __init__(
        self,
        smoothing: float = 10.0
    ):
        self.smoothing = smoothing

    def fit(self, X, y):

        X = pd.DataFrame(X).copy()
        y = pd.Series(y).reset_index(drop=True)

        self.columns_ = X.columns.tolist()

        # Classification labels are encoded to stable numeric values.
        self.global_mean_ = float(
            pd.to_numeric(
                y,
                errors="coerce"
            ).mean()
        )

        if pd.isna(self.global_mean_):
            self.global_mean_ = 0.0

        self.mapping_ = {}

        for col in self.columns_:

            temp = pd.DataFrame({
                "category": X[col].reset_index(drop=True),
                "target": y
            })

            stats = (
                temp.groupby(
                    "category",
                    dropna=False
                )["target"]
                .agg(["mean", "count"])
            )

            smooth = (
                stats["count"] * stats["mean"]
                + self.smoothing * self.global_mean_
            ) / (
                stats["count"]
                + self.smoothing
            )

            self.mapping_[col] = smooth.to_dict()

        return self

    def fit_transform(self, X, y=None, **fit_params):

        if y is None:
            return self.fit(X, y).transform(X)

        X = pd.DataFrame(X).copy()
        y_series = pd.Series(y).reset_index(drop=True)

        self.fit(X, y_series)

        output = pd.DataFrame(
            index=X.index
        )

        for col in self.columns_:

            values = X[col].reset_index(drop=True)

            target_values = y_series

            global_mean = self.global_mean_

            sums = (
                pd.DataFrame({
                    "category": values,
                    "target": target_values
                })
                .groupby(
                    "category",
                    dropna=False
                )["target"]
                .transform("sum")
            )

            counts = (
                pd.DataFrame({
                    "category": values
                })
                .groupby(
                    "category",
                    dropna=False
                )["category"]
                .transform("count")
            )

            loo_sum = sums - target_values
            loo_count = counts - 1

            encoded = (
                loo_sum
                + self.smoothing * global_mean
            ) / (
                loo_count
                + self.smoothing
            )

            output[col] = encoded.fillna(
                global_mean
            ).to_numpy()

        return output

    def transform(self, X):

        X = pd.DataFrame(
            X,
            columns=self.columns_
        ).copy()

        output = pd.DataFrame(
            index=X.index
        )

        for col in self.columns_:

            mapping = self.mapping_.get(
                col,
                {}
            )

            output[col] = (
                X[col]
                .map(mapping)
                .fillna(self.global_mean_)
                .astype(float)
            )

        return output


# ==========================================================
# BASE PREPROCESSOR
# ==========================================================

class BasePreprocessor:
    """
    Shared preprocessing utilities.

    NO FEATURE SELECTION is performed here.
    """

    def __init__(
        self,
        test_size: float = 0.20,
        random_state: int = 42
    ):

        self.test_size = test_size
        self.random_state = random_state

        self.id_cols: List[str] = []
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []

        self.feature_columns: List[str] = []
        self.output_columns: List[str] = []

        self.fitted = False

    # ------------------------------------------------------
    # DATA SPLITTING
    # ------------------------------------------------------

    def split_ids(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        self.id_cols = detect_id_columns(df)

        ids = df[
            self.id_cols
        ].copy() if self.id_cols else pd.DataFrame(
            index=df.index
        )

        X = df.drop(
            columns=self.id_cols,
            errors="ignore"
        ).copy()

        return X, ids

    # ------------------------------------------------------
    # FEATURE TYPE DETECTION
    # ------------------------------------------------------

    def detect_feature_types(
        self,
        X: pd.DataFrame
    ):

        self.numeric_cols = _safe_numeric_columns(X)
        self.categorical_cols = _safe_categorical_columns(X)

        self.feature_columns = X.columns.tolist()

    # ------------------------------------------------------
    # METADATA
    # ------------------------------------------------------

    def get_info(self) -> Dict:

        return {
            "id_columns": self.id_cols,
            "numeric_columns": self.numeric_cols,
            "categorical_columns": self.categorical_cols,
            "feature_count_before_processing":
                len(self.feature_columns),
            "final_feature_count":
                len(self.output_columns),
            "feature_selection": False,
            "feature_selection_removed": 0,
        }


# ==========================================================
# SUPERVISED PREPROCESSOR
# ==========================================================

class SupervisedPreprocessor(
    BasePreprocessor
):

    def __init__(
        self,
        target_col: Optional[str] = None,
        test_size: float = 0.20,
        random_state: int = 42,
        rare_min_frequency: float = 0.01,
        skew_threshold: float = 1.0
    ):

        super().__init__(
            test_size=test_size,
            random_state=random_state
        )

        self.target_col = target_col
        self.rare_min_frequency = rare_min_frequency
        self.skew_threshold = skew_threshold

        self.task = None

        self.numeric_pipeline = None
        self.categorical_pipeline = None

        self.encoder = None
        self.scaler = None

        self.numeric_after_log = []
        self.categorical_encoding = "target_encoding"

    # ------------------------------------------------------
    # TASK DETECTION
    # ------------------------------------------------------

    def detect_task(
        self,
        y: pd.Series
    ) -> str:

        if (
            pd.api.types.is_object_dtype(y)
            or pd.api.types.is_string_dtype(y)
            or pd.api.types.is_bool_dtype(y)
            or pd.api.types.is_categorical_dtype(y)
        ):
            return "classification"

        if pd.api.types.is_numeric_dtype(y):

            # Keep the notebook-style practical heuristic.
            return (
                "classification"
                if y.nunique(dropna=True) <= 20
                else "regression"
            )

        return "classification"

    # ------------------------------------------------------
    # FIT
    # ------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ):

        X = X.copy()
        y = pd.Series(y).reset_index(drop=True)
        X = X.reset_index(drop=True)

        self.detect_feature_types(X)
        self.task = self.detect_task(y)

        # -----------------------------
        # NUMERICAL
        # -----------------------------

        self.numeric_imputer = SimpleImputer(
            strategy="median"
        )

        X_num = pd.DataFrame(
            self.numeric_imputer.fit_transform(
                X[self.numeric_cols]
            ),
            columns=self.numeric_cols
        )

        self.log_transformer = SkewedLogTransformer(
            skew_threshold=self.skew_threshold
        )

        X_num = self.log_transformer.fit_transform(
            X_num
        )

        self.numeric_after_log = (
            self.log_transformer.log_columns_
        )

        # -----------------------------
        # CATEGORICAL
        # -----------------------------

        self.categorical_imputer = SimpleImputer(
            strategy="most_frequent"
        )

        if self.categorical_cols:

            X_cat = pd.DataFrame(
                self.categorical_imputer.fit_transform(
                    X[self.categorical_cols]
                ),
                columns=self.categorical_cols
            )

            self.rare_grouper = RareCategoryGrouper(
                min_frequency=self.rare_min_frequency
            )

            X_cat = self.rare_grouper.fit_transform(
                X_cat
            )

            # Target encoding is retained as the supervised
            # categorical feature-engineering method.
            #
            # For classification, encode target labels to numeric codes
            # before fitting the encoder.
            y_encoder = y.copy()

            if self.task == "classification":
                self.target_classes_ = pd.Index(
                    y_encoder.dropna().unique()
                )

                class_map = {
                    value: i
                    for i, value in enumerate(
                        self.target_classes_
                    )
                }

                y_encoder = y_encoder.map(
                    class_map
                )

            else:
                y_encoder = pd.to_numeric(
                    y_encoder,
                    errors="coerce"
                )

            self.encoder = LeaveOneOutTargetEncoder()

            self.encoder.fit(
                X_cat,
                y_encoder
            )

        # -----------------------------
        # SCALER
        # -----------------------------

        self.scaler = MinMaxScaler()

        transformed = self._transform_core(
            X,
            y=y,
            fit_training=True
        )

        self.output_columns = transformed.columns.tolist()

        self.fitted = True

        return self

    # ------------------------------------------------------
    # TRANSFORM
    # ------------------------------------------------------

    def _transform_core(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        fit_training: bool = False
    ):

        X = X.copy().reset_index(drop=True)

        parts = []

        # Numerical
        if self.numeric_cols:

            X_num = pd.DataFrame(
                self.numeric_imputer.transform(
                    X[self.numeric_cols]
                ),
                columns=self.numeric_cols
            )

            X_num = self.log_transformer.transform(
                X_num
            )

            parts.append(
                X_num.reset_index(drop=True)
            )

        # Categorical -> target encoded
        if self.categorical_cols:

            X_cat = pd.DataFrame(
                self.categorical_imputer.transform(
                    X[self.categorical_cols]
                ),
                columns=self.categorical_cols
            )

            X_cat = self.rare_grouper.transform(
                X_cat
            )

            if fit_training and y is not None:

                y_encoder = pd.Series(y).reset_index(
                    drop=True
                )

                if self.task == "classification":

                    class_map = {
                        value: i
                        for i, value in enumerate(
                            self.target_classes_
                        )
                    }

                    y_encoder = y_encoder.map(
                        class_map
                    )

                else:

                    y_encoder = pd.to_numeric(
                        y_encoder,
                        errors="coerce"
                    )

                X_cat_encoded = (
                    self.encoder.fit_transform(
                        X_cat,
                        y_encoder
                    )
                )

            else:

                X_cat_encoded = (
                    self.encoder.transform(
                        X_cat
                    )
                )

            X_cat_encoded.columns = [
                f"{col}__target_encoded"
                for col in self.categorical_cols
            ]

            parts.append(
                X_cat_encoded.reset_index(drop=True)
            )

        if parts:

            X_out = pd.concat(
                parts,
                axis=1
            )

        else:

            X_out = pd.DataFrame(
                index=X.index
            )

        # Fit scaler only on training data.
        if fit_training:

            scaled = self.scaler.fit_transform(
                X_out
            )

        else:

            scaled = self.scaler.transform(
                X_out
            )

        X_out = pd.DataFrame(
            scaled,
            columns=X_out.columns
        )

        return X_out

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ):

        X_no_ids, ids = self.split_ids(X)

        self.fit(
            X_no_ids,
            y
        )

        processed = self._transform_core(
            X_no_ids,
            y=y,
            fit_training=True
        )

        return processed, ids

    def transform(
        self,
        X: pd.DataFrame
    ):

        if not self.fitted:
            raise RuntimeError(
                "Preprocessor must be fitted before transform()."
            )

        X_no_ids = X.drop(
            columns=self.id_cols,
            errors="ignore"
        ).copy()

        ids = (
            X[self.id_cols].copy()
            if self.id_cols
            else pd.DataFrame(
                index=X.index
            )
        )

        processed = self._transform_core(
            X_no_ids,
            y=None,
            fit_training=False
        )

        return processed, ids

    def get_info(self):

        info = super().get_info()

        info.update({
            "task": self.task,
            "target_column": self.target_col,
            "numeric_log_transformed":
                self.numeric_after_log,
            "categorical_encoding":
                self.categorical_encoding,
            "feature_selection": False,
            "feature_selection_removed": 0,
            "final_feature_count":
                len(self.output_columns),
        })

        return info


# ==========================================================
# UNSUPERVISED PREPROCESSOR
# ==========================================================

class UnsupervisedPreprocessor(
    BasePreprocessor
):

    def __init__(
        self,
        test_size: float = 0.20,
        random_state: int = 42,
        rare_min_frequency: float = 0.01,
        skew_threshold: float = 1.0
    ):

        super().__init__(
            test_size=test_size,
            random_state=random_state
        )

        self.rare_min_frequency = rare_min_frequency
        self.skew_threshold = skew_threshold

        self.scaler = None
        self.encoder = None

    # ------------------------------------------------------
    # FIT
    # ------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame
    ):

        X = X.copy().reset_index(drop=True)

        self.detect_feature_types(X)

        # Numerical
        self.numeric_imputer = SimpleImputer(
            strategy="median"
        )

        X_num = pd.DataFrame(
            self.numeric_imputer.fit_transform(
                X[self.numeric_cols]
            ),
            columns=self.numeric_cols
        )

        self.log_transformer = SkewedLogTransformer(
            skew_threshold=self.skew_threshold
        )

        X_num = self.log_transformer.fit_transform(
            X_num
        )

        # Categorical
        if self.categorical_cols:

            self.categorical_imputer = SimpleImputer(
                strategy="most_frequent"
            )

            X_cat = pd.DataFrame(
                self.categorical_imputer.fit_transform(
                    X[self.categorical_cols]
                ),
                columns=self.categorical_cols
            )

            self.rare_grouper = RareCategoryGrouper(
                min_frequency=self.rare_min_frequency
            )

            X_cat = self.rare_grouper.fit_transform(
                X_cat
            )

            self.encoder = _make_ohe()

            self.encoder.fit(
                X_cat
            )

        # Fit scaler on the complete transformed training matrix.
        X_out = self._transform_core(
            X,
            fit_scaler=True
        )

        self.output_columns = X_out.columns.tolist()
        self.fitted = True

        return self

    # ------------------------------------------------------
    # TRANSFORM
    # ------------------------------------------------------

    def _transform_core(
        self,
        X: pd.DataFrame,
        fit_scaler: bool = False
    ):

        X = X.copy().reset_index(drop=True)

        parts = []

        if self.numeric_cols:

            X_num = pd.DataFrame(
                self.numeric_imputer.transform(
                    X[self.numeric_cols]
                ),
                columns=self.numeric_cols
            )

            X_num = self.log_transformer.transform(
                X_num
            )

            parts.append(
                X_num.reset_index(drop=True)
            )

        if self.categorical_cols:

            X_cat = pd.DataFrame(
                self.categorical_imputer.transform(
                    X[self.categorical_cols]
                ),
                columns=self.categorical_cols
            )

            X_cat = self.rare_grouper.transform(
                X_cat
            )

            encoded = self.encoder.transform(
                X_cat
            )

            try:
                names = (
                    self.encoder
                    .get_feature_names_out(
                        self.categorical_cols
                    )
                )
            except Exception:
                names = [
                    f"cat_{i}"
                    for i in range(
                        encoded.shape[1]
                    )
                ]

            X_cat_encoded = pd.DataFrame(
                encoded,
                columns=names
            )

            parts.append(
                X_cat_encoded.reset_index(drop=True)
            )

        if parts:

            X_out = pd.concat(
                parts,
                axis=1
            )

        else:

            X_out = pd.DataFrame(
                index=X.index
            )

        if fit_scaler:

            scaled = self.scaler.fit_transform(
                X_out
            )

        else:

            scaled = self.scaler.transform(
                X_out
            )

        return pd.DataFrame(
            scaled,
            columns=X_out.columns
        )

    def fit_transform(
        self,
        X: pd.DataFrame
    ):

        X_no_ids, ids = self.split_ids(X)

        self.fit(
            X_no_ids
        )

        processed = self._transform_core(
            X_no_ids,
            fit_scaler=True
        )

        return processed, ids

    def transform(
        self,
        X: pd.DataFrame
    ):

        if not self.fitted:
            raise RuntimeError(
                "Preprocessor must be fitted before transform()."
            )

        X_no_ids = X.drop(
            columns=self.id_cols,
            errors="ignore"
        ).copy()

        ids = (
            X[self.id_cols].copy()
            if self.id_cols
            else pd.DataFrame(
                index=X.index
            )
        )

        processed = self._transform_core(
            X_no_ids,
            fit_scaler=False
        )

        return processed, ids

    def get_info(self):

        info = super().get_info()

        info.update({
            "numeric_log_transformed":
                getattr(
                    self,
                    "log_transformer",
                    None
                ).log_columns_
                if hasattr(
                    getattr(
                        self,
                        "log_transformer",
                        None
                    ),
                    "log_columns_"
                )
                else [],
            "categorical_encoding":
                "one_hot_encoding",
            "feature_selection": False,
            "feature_selection_removed": 0,
            "final_feature_count":
                len(self.output_columns),
        })

        return info


# ==========================================================
# HIGH-LEVEL PROCESSING
# ==========================================================

def process_supervised_dataset(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.20,
    random_state: int = 42
) -> Dict:

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found."
        )

    from sklearn.model_selection import train_test_split

    X = df.drop(
        columns=[target_col]
    ).copy()

    y = df[target_col].copy()

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state
        )
    )

    processor = SupervisedPreprocessor(
        target_col=target_col,
        test_size=test_size,
        random_state=random_state
    )

    X_train_processed, train_ids = (
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

    # Preserve target in the processed supervised outputs.
    train_output = X_train_processed.copy()
    test_output = X_test_processed.copy()

    train_output[target_col] = (
        y_train.reset_index(drop=True)
    )

    # If test target exists in the supplied source, preserve it.
    test_output[target_col] = (
        y_test.reset_index(drop=True)
    )

    # Preserve detected IDs.
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

    info.update({
        "dataset_type": "Entire Dataset",
        "rows_processed": len(df),
        "train_rows": len(train_output),
        "test_rows": len(test_output),
    })

    return {
        "X_train": train_output,
        "X_test": test_output,
        "y_train": y_train,
        "y_test": y_test,
        "processor": processor,
        "task": processor.task,
        "info": info,
    }


def process_unsupervised_dataset(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42
) -> Dict:

    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state
    )

    processor = UnsupervisedPreprocessor(
        test_size=test_size,
        random_state=random_state
    )

    X_train, train_ids = (
        processor.fit_transform(
            train_df
        )
    )

    X_test, test_ids = (
        processor.transform(
            test_df
        )
    )

    if not train_ids.empty:
        X_train = pd.concat(
            [
                train_ids.reset_index(drop=True),
                X_train.reset_index(drop=True)
            ],
            axis=1
        )

    if not test_ids.empty:
        X_test = pd.concat(
            [
                test_ids.reset_index(drop=True),
                X_test.reset_index(drop=True)
            ],
            axis=1
        )

    info = processor.get_info()

    info.update({
        "task": "unsupervised",
        "dataset_type": "Entire Dataset",
        "rows_processed": len(df),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    })

    return {
        "X_train": X_train,
        "X_test": X_test,
        "processor": processor,
        "task": "unsupervised",
        "info": info,
    }


# ==========================================================
# BACKWARD-COMPATIBLE ALIASES
# ==========================================================

SupervisedPipeline = SupervisedPreprocessor
UnsupervisedPipeline = UnsupervisedPreprocessor
