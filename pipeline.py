"""
General-purpose preprocessing pipeline for the ML automation application.

Key fixes:
1. ID columns (e.g. CustomerID) are removed from processed model datasets.
   They are NOT appended back to train/test CSV outputs.
2. No house-price-specific logic:
   - no SalePrice
   - no LotFrontage
   - no LotArea
   - no 1stFlrSF
   - no GrLivArea
   - no YrSold / YearBuilt assumptions
3. Supervised categorical preprocessing is target-independent.
   This is important for cross-validation because target-mean encoding
   outside the CV folds can leak validation-fold target information.
4. All learned preprocessing state is fitted on training data only.
5. Train/test feature columns are frozen and aligned.
"""

import pandas as pd
import numpy as np

from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer


class DataPreprocessor:

    def __init__(
        self,
        target_col=None,
        task=None,
        rare_label_threshold=0.01,
        skew_threshold=0.75,
        lasso_alpha=0.0005,
        logistic_c=1.0,
        scale_features=True,
    ):
        self.target_col = target_col
        self.task = task

        self.rare_label_threshold = rare_label_threshold
        self.skew_threshold = skew_threshold

        # Kept for API compatibility with the existing application.
        self.lasso_alpha = lasso_alpha
        self.logistic_c = logistic_c

        self.scale_features = scale_features

        # --------------------------------------------------
        # General state
        # --------------------------------------------------
        self.id_cols = []

        self.features_with_nan = []
        self.features_nan_cat = []
        self.features_nan_num = []

        self.train_medians = {}
        self.frequent_labels = {}

        self.categorical_features = []

        self.skewed_features = []
        self.skew_shifts = {}

        self.feature_columns = []
        self.scalable_features = []
        self.scaler = None

        self.feature_count_before_processing = 0

        self.log_features = []
        self.outlier_bounds = {}

        self.near_zero_variance_cols = []
        self.multicollinearity_drop_cols = []

        self.target_log_enabled = False

        self.fitted = False
        self.selected_features = []

        # --------------------------------------------------
        # Target state
        # --------------------------------------------------
        self.target_classes = []
        self.target_class_mapping = {}

        self.target_encoder = None
        self.target_was_label_encoded = False

        # --------------------------------------------------
        # Imputation / categorical state
        # --------------------------------------------------
        self.numeric_imputer = None
        self.categorical_imputer = None

        self.numeric_imputer_features = []
        self.categorical_imputer_features = []

        # --------------------------------------------------
        # Unsupervised state
        # --------------------------------------------------
        self.one_hot_categories = {}
        self.unsupervised_feature_columns = []
        self.unsupervised_log_features = []
        self.unsupervised_outlier_bounds = {}
        self.unsupervised_near_zero_cols = []
        self.unsupervised_multicollinearity_drop_cols = []
        self.unsupervised_frequency_mappings = {}
        self.unsupervised_frequency_fallbacks = {}

    # ======================================================
    # BASIC HELPERS
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

    def _get_numeric_features(self, X):
        return [
            col
            for col in X.columns
            if pd.api.types.is_numeric_dtype(X[col])
        ]

    # ======================================================
    # TASK DETECTION
    # ======================================================

    def _detect_task(self, y):
        if (
            pd.api.types.is_object_dtype(y)
            or pd.api.types.is_string_dtype(y)
            or pd.api.types.is_bool_dtype(y)
            or pd.api.types.is_categorical_dtype(y)
        ):
            return "classification"

        if pd.api.types.is_numeric_dtype(y):
            return "classification" if y.nunique() < 10 else "regression"

        return "classification"

    # ======================================================
    # ID DETECTION
    # ======================================================

    def _detect_ids(self, X):
        """
        Detect identifier-like columns.

        A column is considered an ID when every row has a unique
        value. This catches CustomerID without hardcoding its name.
        """
        self.id_cols = []

        for col in X.columns:
            if X[col].nunique(dropna=False) == len(X):
                self.id_cols.append(col)

        return self.id_cols

    # ======================================================
    # MISSING VALUES
    # ======================================================

    def _detect_missing_features(self, X):
        self.features_with_nan = [
            col for col in X.columns if X[col].isna().any()
        ]

        self.features_nan_num = [
            col
            for col in self.features_with_nan
            if pd.api.types.is_numeric_dtype(X[col])
        ]

        self.features_nan_cat = [
            col
            for col in self.features_with_nan
            if col not in self.features_nan_num
        ]

    def _add_missing_indicators(self, X):
        X = X.copy()

        for col in self.features_with_nan:
            if col in X.columns:
                X[f"{col}__missing"] = X[col].isna().astype(int)

        return X

    def _fit_imputation(self, X):
        numeric_features = self._get_numeric_features(X)
        categorical_features = self._get_categorical_features(X)

        self.numeric_imputer_features = numeric_features
        self.categorical_imputer_features = categorical_features

        if numeric_features:
            self.numeric_imputer = SimpleImputer(strategy="median")
            self.numeric_imputer.fit(X[numeric_features])

            self.train_medians = {
                col: float(X[col].median())
                for col in numeric_features
            }
        else:
            self.numeric_imputer = None
            self.train_medians = {}

        if categorical_features:
            self.categorical_imputer = SimpleImputer(
                strategy="most_frequent"
            )
            self.categorical_imputer.fit(X[categorical_features])
        else:
            self.categorical_imputer = None

    def _apply_imputation(self, X):
        X = X.copy()

        if self.numeric_imputer is not None:
            cols = [
                c for c in self.numeric_imputer_features
                if c in X.columns
            ]
            if cols:
                X[cols] = self.numeric_imputer.transform(X[cols])

        if self.categorical_imputer is not None:
            cols = [
                c for c in self.categorical_imputer_features
                if c in X.columns
            ]
            if cols:
                X[cols] = self.categorical_imputer.transform(X[cols])

        return X

    # ======================================================
    # SKEW / LOG TRANSFORMATION
    # ======================================================

    def _fit_skewness(self, X):
        self.skewed_features = []
        self.skew_shifts = {}

        for feature in self._get_numeric_features(X):
            series = pd.to_numeric(
                X[feature],
                errors="coerce"
            ).dropna()

            if len(series) < 3:
                continue

            skew_value = stats.skew(series)

            if abs(skew_value) > self.skew_threshold:
                minimum = series.min()

                if minimum <= 0:
                    self.skew_shifts[feature] = float(abs(minimum) + 1)
                else:
                    self.skew_shifts[feature] = 0.0

                self.skewed_features.append(feature)

    def _apply_skewness(self, X):
        X = X.copy()

        for feature in self.skewed_features:
            if feature not in X.columns:
                continue

            shift = self.skew_shifts.get(feature, 0.0)

            values = pd.to_numeric(
                X[feature],
                errors="coerce"
            )

            X[feature] = np.log1p(values + shift)

        return X

    # Generalized notebook-compatible method.
    # It intentionally does NOT contain SalePrice or Ames housing columns.
    def _fit_log_features(self, X, y=None):
        self._fit_skewness(X)
        self.log_features = self.skewed_features.copy()

        # Target log is only relevant to regression and is kept
        # generalized. Classification targets are never log transformed.
        self.target_log_enabled = False

        if self.task == "regression" and y is not None:
            values = pd.to_numeric(
                pd.Series(y),
                errors="coerce"
            )
            if (
                len(values) > 0
                and not values.isna().any()
                and (values > 0).all()
            ):
                self.target_log_enabled = True

    def _apply_log_features(self, X):
        return self._apply_skewness(X)

    # ======================================================
    # OUTLIER HANDLING
    # ======================================================

    def _fit_outlier_bounds(self, X):
        self.outlier_bounds = {}

        # Fit bounds on all numerical predictors rather than only on
        # hardcoded housing variables.
        for feature in self._get_numeric_features(X):
            series = pd.to_numeric(
                X[feature],
                errors="coerce"
            ).dropna()

            if series.empty:
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            if iqr == 0:
                continue

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            self.outlier_bounds[feature] = (
                float(lower),
                float(upper)
            )

    def _apply_outlier_clipping(self, X):
        X = X.copy()

        for feature, (lower, upper) in self.outlier_bounds.items():
            if feature in X.columns:
                X[feature] = X[feature].clip(lower, upper)

        return X

    # ======================================================
    # RARE CATEGORIES
    # ======================================================

    def _fit_rare_labels(self, X):
        self.categorical_features = self._get_categorical_features(X)
        self.frequent_labels = {}

        for feature in self.categorical_features:
            frequencies = X[feature].value_counts(
                normalize=True,
                dropna=False
            )

            self.frequent_labels[feature] = (
                frequencies[
                    frequencies > self.rare_label_threshold
                ].index.tolist()
            )

    def _apply_rare_labels(self, X):
        X = X.copy()

        for feature, frequent in self.frequent_labels.items():
            if feature not in X.columns:
                continue

            X[feature] = np.where(
                X[feature].isin(frequent),
                X[feature],
                "Rare"
            )

        return X

    # ======================================================
    # NEAR-ZERO VARIANCE
    # ======================================================

    def _fit_near_zero_variance(self, X):
        self.near_zero_variance_cols = []

        for feature in X.columns:
            frequencies = X[feature].value_counts(
                normalize=True,
                dropna=False
            )

            if frequencies.empty:
                continue

            if frequencies.iloc[0] > 0.99:
                self.near_zero_variance_cols.append(feature)

    # ======================================================
    # MULTICOLLINEARITY
    # ======================================================

    def _fit_multicollinearity(self, X, y_processed=None):
        """
        Detect highly correlated numerical predictors.

        Target is deliberately NOT included in this calculation.
        This keeps feature removal independent of the target and avoids
        using target information during preprocessing.
        """
        self.multicollinearity_drop_cols = []

        numeric_features = self._get_numeric_features(X)

        if len(numeric_features) < 2:
            return

        corr_matrix = X[numeric_features].corr()

        threshold = 0.90
        drop_cols = set()

        columns = corr_matrix.columns.tolist()

        for i in range(len(columns)):
            for j in range(i):
                col_i = columns[i]
                col_j = columns[j]

                corr = corr_matrix.loc[col_i, col_j]

                if abs(corr) > threshold:
                    # Drop the later feature deterministically.
                    drop_cols.add(col_i)

        self.multicollinearity_drop_cols = sorted(drop_cols)

    def _apply_supervised_feature_drops(self, X):
        X = X.copy()

        drop_cols = set(
            self.near_zero_variance_cols
            + self.multicollinearity_drop_cols
        )

        if drop_cols:
            X = X.drop(
                columns=list(drop_cols),
                errors="ignore"
            )

        return X

    # ======================================================
    # TARGET ENCODING
    # ======================================================

    def _fit_target_encoder(self, y):
        """
        Encode only the target itself.

        Feature encoding is deliberately target-independent so this
        preprocessor can safely be used inside cross-validation.
        """
        self.target_encoder = None
        self.target_was_label_encoded = False
        self.target_classes = []
        self.target_class_mapping = {}

        is_categorical_target = (
            pd.api.types.is_object_dtype(y)
            or pd.api.types.is_string_dtype(y)
            or pd.api.types.is_bool_dtype(y)
            or pd.api.types.is_categorical_dtype(y)
        )

        if self.task == "classification" and is_categorical_target:
            y_series = pd.Series(y).reset_index(drop=True)

            if y_series.isna().any():
                raise ValueError(
                    "Target column contains missing values."
                )

            self.target_encoder = LabelEncoder()
            self.target_encoder.fit(y_series.astype(str))

            self.target_classes = (
                self.target_encoder.classes_.tolist()
            )

            self.target_class_mapping = {
                value: index
                for index, value
                in enumerate(self.target_classes)
            }

            self.target_was_label_encoded = True

    def encode_target(self, y):
        y_series = pd.Series(y).reset_index(drop=True)

        if self.target_encoder is not None:
            try:
                return self.target_encoder.transform(
                    y_series.astype(str).to_numpy()
                )
            except ValueError as exc:
                raise ValueError(
                    "The target contains a class that was not "
                    "present in the training target."
                ) from exc

        if self.task == "regression":
            values = pd.to_numeric(
                y_series,
                errors="coerce"
            )

            if values.isna().any():
                raise ValueError(
                    "Regression target contains missing or non-numeric values."
                )

            if self.target_log_enabled:
                if (values <= 0).any():
                    raise ValueError(
                        "Regression target contains zero/negative values "
                        "but target log transformation was enabled."
                    )
                return np.log(values.to_numpy())

            return values.to_numpy()

        return y_series.to_numpy()

    def get_target_classes(self):
        return self.target_classes

    # ======================================================
    # CATEGORICAL ENCODING
    # ======================================================

    def _fit_one_hot_encoding(self, X):
        """
        Fit one-hot categories on training data only.

        This is target-independent and therefore safe for CV.
        """
        self.one_hot_categories = {}

        for feature in self._get_categorical_features(X):
            categories = (
                X[feature]
                .astype(str)
                .drop_duplicates()
                .tolist()
            )

            self.one_hot_categories[feature] = categories

    def _apply_one_hot_encoding(self, X):
        X = X.copy()

        for feature, categories in self.one_hot_categories.items():
            if feature not in X.columns:
                for category in categories:
                    X[f"{feature}_{category}"] = 0
                continue

            values = X[feature].astype(str)

            for category in categories:
                X[f"{feature}_{category}"] = (
                    values == str(category)
                ).astype(int)

            X = X.drop(columns=[feature])

        return X

    # ======================================================
    # NUMERIC VALIDATION
    # ======================================================

    def _ensure_numeric(self, X):
        X = X.copy()

        boolean_features = [
            col
            for col in X.columns
            if pd.api.types.is_bool_dtype(X[col])
        ]

        for feature in boolean_features:
            X[feature] = X[feature].astype(int)

        non_numeric_features = [
            col
            for col in X.columns
            if not pd.api.types.is_numeric_dtype(X[col])
        ]

        if non_numeric_features:
            raise ValueError(
                "Non-numeric features remain before scaling: "
                f"{non_numeric_features}"
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

        extra = [
            col for col in X.columns
            if col not in self.feature_columns
        ]

        if extra:
            X = X.drop(columns=extra)

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

        if not self.scalable_features:
            self.scaler = None
            return

        self.scaler = MinMaxScaler()
        self.scaler.fit(X[self.scalable_features])

    def _apply_scaling(self, X):
        X = X.copy()

        if (
            not self.scale_features
            or self.scaler is None
            or not self.scalable_features
        ):
            return X

        X[self.scalable_features] = self.scaler.transform(
            X[self.scalable_features]
        )

        return X

    # ======================================================
    # SUPERVISED FIT
    # ======================================================

    def fit(self, X_train, y_train):
        if y_train is None:
            raise ValueError(
                "Target values are required for supervised preprocessing."
            )

        X_train = X_train.copy()
        y_train = pd.Series(
            y_train,
            index=X_train.index
        )

        self.task = self.task or self._detect_task(y_train)

        if self.task == "regression":
            y_train = pd.to_numeric(
                y_train,
                errors="coerce"
            )

            if y_train.isna().any():
                raise ValueError(
                    "Regression target contains missing/non-numeric values."
                )

        self.feature_count_before_processing = X_train.shape[1]

        # IDs are detected and removed BEFORE any feature processing.
        self._detect_ids(X_train)

        X = X_train.drop(
            columns=self.id_cols,
            errors="ignore"
        )

        self._detect_missing_features(X)

        # Missing handling state is fitted on TRAIN ONLY.
        self._fit_imputation(X)

        X = self._add_missing_indicators(X)
        X = self._apply_imputation(X)

        # General skew/log handling.
        self._fit_log_features(X, y_train)
        X = self._apply_log_features(X)

        # Outlier bounds are fitted on TRAIN ONLY.
        self._fit_outlier_bounds(X)
        X = self._apply_outlier_clipping(X)

        # Rare-category state is fitted on TRAIN ONLY.
        self._fit_rare_labels(X)
        X = self._apply_rare_labels(X)

        # Structural feature checks.
        self._fit_near_zero_variance(X)
        X = self._apply_supervised_feature_drops(X)

        self._fit_multicollinearity(X)
        X = self._apply_supervised_feature_drops(X)

        # IMPORTANT:
        # Categorical features are encoded without using y.
        self._fit_one_hot_encoding(X)
        X = self._apply_one_hot_encoding(X)

        X = self._ensure_numeric(X)

        self.feature_columns = X.columns.tolist()
        self.selected_features = self.feature_columns.copy()

        # Scaling is fitted on TRAIN ONLY.
        self._fit_scaler(X)

        self.fitted = True
        return self

    # ======================================================
    # SUPERVISED TRANSFORM
    # ======================================================

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError(
                "Pipeline has not been fitted yet."
            )

        X = X.copy()

        # IDs are dropped and never returned in processed model data.
        X = X.drop(
            columns=self.id_cols,
            errors="ignore"
        )

        X = self._add_missing_indicators(X)
        X = self._apply_imputation(X)

        X = self._apply_log_features(X)
        X = self._apply_outlier_clipping(X)

        X = self._apply_rare_labels(X)

        X = self._apply_supervised_feature_drops(X)

        X = self._apply_one_hot_encoding(X)

        X = self._align_features(X)
        X = self._ensure_numeric(X)
        X = self._apply_scaling(X)

        return X

    # ======================================================
    # SUPERVISED FIT TRANSFORM
    # ======================================================

    def fit_transform(self, X_train, y_train):
        self.fit(X_train, y_train)

        X = X_train.copy()

        X = X.drop(
            columns=self.id_cols,
            errors="ignore"
        )

        X = self._add_missing_indicators(X)
        X = self._apply_imputation(X)

        X = self._apply_log_features(X)
        X = self._apply_outlier_clipping(X)

        X = self._apply_rare_labels(X)
        X = self._apply_supervised_feature_drops(X)

        X = self._apply_one_hot_encoding(X)

        X = self._align_features(X)
        X = self._ensure_numeric(X)
        X = self._apply_scaling(X)

        return X

    # ======================================================
    # UNSUPERVISED METHODS
    # ======================================================

    def _fit_unsupervised_log_features(self, X):
        self.unsupervised_log_features = []

        for feature in self._get_numeric_features(X):
            series = pd.to_numeric(
                X[feature],
                errors="coerce"
            ).dropna()

            if len(series) < 3:
                continue

            skew_value = stats.skew(series)

            if abs(skew_value) > self.skew_threshold:
                minimum = series.min()
                self.unsupervised_log_features.append(feature)

                if minimum <= 0:
                    self.skew_shifts[feature] = float(abs(minimum) + 1)
                else:
                    self.skew_shifts[feature] = 0.0

    def _apply_unsupervised_log_features(self, X):
        X = X.copy()

        for feature in self.unsupervised_log_features:
            if feature not in X.columns:
                continue

            shift = self.skew_shifts.get(feature, 0.0)
            X[feature] = np.log1p(
                pd.to_numeric(X[feature], errors="coerce") + shift
            )

        return X

    def _fit_unsupervised_outliers(self, X):
        self.unsupervised_outlier_bounds = {}

        for feature in self._get_numeric_features(X):
            series = pd.to_numeric(
                X[feature],
                errors="coerce"
            ).dropna()

            if series.empty:
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            if iqr == 0:
                continue

            self.unsupervised_outlier_bounds[feature] = (
                float(q1 - 1.5 * iqr),
                float(q3 + 1.5 * iqr)
            )

    def _apply_unsupervised_outliers(self, X):
        X = X.copy()

        for feature, (lower, upper) in (
            self.unsupervised_outlier_bounds.items()
        ):
            if feature in X.columns:
                X[feature] = X[feature].clip(lower, upper)

        return X

    def _fit_unsupervised_nzv(self, X):
        self.unsupervised_near_zero_cols = []

        for feature in X.columns:
            frequencies = X[feature].value_counts(
                normalize=True,
                dropna=False
            )

            if not frequencies.empty and frequencies.iloc[0] > 0.99:
                self.unsupervised_near_zero_cols.append(feature)

    def _fit_unsupervised_multicollinearity(self, X):
        self.unsupervised_multicollinearity_drop_cols = []

        numeric_features = self._get_numeric_features(X)

        if len(numeric_features) < 2:
            return

        corr = X[numeric_features].corr()
        columns = corr.columns.tolist()
        drops = set()

        for i in range(len(columns)):
            for j in range(i):
                if abs(corr.iloc[i, j]) > 0.90:
                    drops.add(columns[i])

        self.unsupervised_multicollinearity_drop_cols = sorted(drops)

    def _fit_frequency_encoding(self, X):
        self.unsupervised_frequency_mappings = {}
        self.unsupervised_frequency_fallbacks = {}

        for feature in self._get_categorical_features(X):
            frequencies = X[feature].value_counts(normalize=True)

            ordered = frequencies.sort_values().index.tolist()

            mapping = {
                category: rank
                for rank, category in enumerate(ordered)
            }

            self.unsupervised_frequency_mappings[feature] = mapping

            self.unsupervised_frequency_fallbacks[feature] = (
                int(round((len(mapping) - 1) / 2))
                if mapping
                else 0
            )

    def _apply_frequency_encoding(self, X):
        X = X.copy()

        for feature, mapping in (
            self.unsupervised_frequency_mappings.items()
        ):
            if feature not in X.columns:
                continue

            fallback = self.unsupervised_frequency_fallbacks.get(
                feature,
                0
            )

            X[feature] = (
                X[feature]
                .map(mapping)
                .fillna(fallback)
                .astype(float)
            )

        return X

    def fit_unsupervised(self, X):
        X = X.copy()

        self.feature_count_before_processing = X.shape[1]

        self._detect_ids(X)

        X = X.drop(
            columns=self.id_cols,
            errors="ignore"
        )

        self._detect_missing_features(X)

        X = self._add_missing_indicators(X)

        self._fit_imputation(X)
        X = self._apply_imputation(X)

        self._fit_unsupervised_log_features(X)
        X = self._apply_unsupervised_log_features(X)

        self._fit_unsupervised_outliers(X)
        X = self._apply_unsupervised_outliers(X)

        self._fit_rare_labels(X)
        X = self._apply_rare_labels(X)

        self._fit_unsupervised_nzv(X)

        drops = (
            self.unsupervised_near_zero_cols
            + self.unsupervised_multicollinearity_drop_cols
        )

        X = X.drop(columns=drops, errors="ignore")

        self._fit_frequency_encoding(X)
        X = self._apply_frequency_encoding(X)

        X = self._ensure_numeric(X)

        self.feature_columns = X.columns.tolist()
        self.unsupervised_feature_columns = self.feature_columns.copy()
        self.selected_features = self.feature_columns.copy()

        self._fit_scaler(X)

        self.task = "unsupervised"
        self.fitted = True

        return self

    def transform_unsupervised(self, X):
        if not self.fitted:
            raise RuntimeError(
                "Unsupervised pipeline has not been fitted yet."
            )

        X = X.copy()

        X = X.drop(
            columns=self.id_cols,
            errors="ignore"
        )

        X = self._add_missing_indicators(X)
        X = self._apply_imputation(X)

        X = self._apply_unsupervised_log_features(X)
        X = self._apply_unsupervised_outliers(X)

        X = self._apply_rare_labels(X)

        drops = (
            self.unsupervised_near_zero_cols
            + self.unsupervised_multicollinearity_drop_cols
        )

        X = X.drop(columns=drops, errors="ignore")

        X = self._apply_frequency_encoding(X)

        X = self._align_features(X)
        X = self._ensure_numeric(X)
        X = self._apply_scaling(X)

        return X

    def fit_transform_unsupervised(self, X):
        self.fit_unsupervised(X)
        return self.transform_unsupervised(X)

    # ======================================================
    # INFO
    # ======================================================

    def get_info(self):
        if not self.fitted:
            raise RuntimeError(
                "Pipeline has not been fitted yet."
            )

        return {
            "task": self.task,
            "target": (
                self.target_col
                if self.target_col is not None
                else "None"
            ),
            "id_columns": self.id_cols,
            "missing_value_features": self.features_with_nan,
            "numeric_missing_features": self.features_nan_num,
            "categorical_missing_features": self.features_nan_cat,
            "skewed_features": self.log_features,
            "scaled_features": self.scalable_features,
            "original_feature_count": self.feature_count_before_processing,
            "selected_feature_count": len(self.feature_columns),
            "feature_count_before_processing": (
                self.feature_count_before_processing
            ),
            "final_feature_count": len(self.feature_columns),
            "selected_features": self.feature_columns.copy(),
            "feature_selection_method": "None",
            "near_zero_variance_removed": (
                self.near_zero_variance_cols
            ),
            "multicollinearity_removed": (
                self.multicollinearity_drop_cols
            ),
            "log_features": self.log_features,
            "target_log": self.target_log_enabled,
            "outlier_clipped_features": list(
                self.outlier_bounds.keys()
            ),
            "categorical_encoding": (
                "One-hot encoding"
                if self.one_hot_categories
                else "None"
            ),
            "target_label_encoding": (
                "LabelEncoder"
                if self.target_was_label_encoded
                else "None"
            ),
            "target_classes": self.get_target_classes(),
            "unsupervised_log_features": (
                self.unsupervised_log_features
            ),
            "unsupervised_outlier_clipped_features": list(
                self.unsupervised_outlier_bounds.keys()
            ),
            "unsupervised_near_zero_variance_removed": (
                self.unsupervised_near_zero_cols
            ),
            "unsupervised_multicollinearity_removed": (
                self.unsupervised_multicollinearity_drop_cols
            ),
            "unsupervised_categorical_encoding": (
                "Frequency-ordered encoding"
                if self.unsupervised_frequency_mappings
                else "None"
            ),
        }


# ==========================================================
# API COMPATIBILITY WRAPPERS
# ==========================================================

class SupervisedPreprocessor(DataPreprocessor):

    def detect_task(self, y):
        return self._detect_task(y)


class UnsupervisedPreprocessor(DataPreprocessor):

    def fit_transform(self, X):
        self.fit_unsupervised(X)
        return self.transform_unsupervised(X)

    def transform(self, X):
        return self.transform_unsupervised(X)


# ==========================================================
# SUPERVISED DATASET PROCESSOR
# ==========================================================

def process_supervised_dataset(
    df,
    target_col,
    test_size=0.20,
    random_state=42
):
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found."
        )

    df = df.copy()

    # Remove exact duplicate rows before splitting.
    df = df.drop_duplicates().reset_index(drop=True)

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    # Stratification is appropriate for classification and preserves
    # class proportions between train and test.
    temp_processor = SupervisedPreprocessor(
        target_col=target_col
    )

    detected_task = temp_processor.detect_task(y)

    stratify = y if detected_task == "classification" else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    processor = SupervisedPreprocessor(
        target_col=target_col,
        task=detected_task
    )

    X_train_processed = processor.fit_transform(
        X_train,
        y_train
    )

    X_test_processed = processor.transform(
        X_test
    )

    y_train_processed = processor.encode_target(y_train)
    y_test_processed = processor.encode_target(y_test)

    train_output = X_train_processed.copy()
    test_output = X_test_processed.copy()

    # IDs are intentionally NOT appended back.
    # The processed CSVs are model-ready datasets only.
    train_output[target_col] = y_train_processed
    test_output[target_col] = y_test_processed

    train_output = train_output.reset_index(drop=True)
    test_output = test_output.reset_index(drop=True)

    info = processor.get_info()

    info.update(
        {
            "task": processor.task,
            "target": target_col,
            "dataset_type": "Entire Dataset",
            "rows_processed": len(df),
            "feature_selection_method": "None",
            "ids_removed_from_model_data": processor.id_cols,
        }
    )

    return {
        "X_train": train_output,
        "X_test": test_output,
        "y_train": pd.Series(
            y_train_processed,
            name=target_col
        ),
        "y_test": pd.Series(
            y_test_processed,
            name=target_col
        ),
        "processor": processor,
        "task": processor.task,
        "info": info,
    }


# ==========================================================
# UNSUPERVISED DATASET PROCESSOR
# ==========================================================

def process_unsupervised_dataset(
    df,
    test_size=0.20,
    random_state=42
):
    df = df.copy()

    df = df.drop_duplicates().reset_index(drop=True)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state
    )

    processor = UnsupervisedPreprocessor()

    X_train_processed = processor.fit_transform(
        train_df
    )

    X_test_processed = processor.transform(
        test_df
    )

    # IDs are intentionally NOT appended back to processed outputs.
    train_output = X_train_processed.reset_index(drop=True)
    test_output = X_test_processed.reset_index(drop=True)

    info = processor.get_info()

    info.update(
        {
            "task": "unsupervised",
            "target": "None",
            "dataset_type": "Entire Dataset",
            "rows_processed": len(df),
            "feature_selection_method": "None",
            "ids_removed_from_model_data": processor.id_cols,
        }
    )

    return {
        "X_train": train_output,
        "X_test": test_output,
        "processor": processor,
        "task": "unsupervised",
        "info": info,
    }
