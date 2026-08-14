import pandas as pd
import numpy as np

from scipy import stats

from sklearn.preprocessing import MinMaxScaler


class DataPreprocessor:

    def __init__(
        self,
        target_col=None,
        task=None,
        rare_label_threshold=0.01,
        skew_threshold=0.75,
        lasso_alpha=0.0005,
        logistic_c=1.0,
        scale_features=True
    ):

        self.target_col = target_col
        self.task = task

        self.rare_label_threshold = rare_label_threshold
        self.skew_threshold = skew_threshold

        self.lasso_alpha = lasso_alpha
        self.logistic_c = logistic_c

        self.scale_features = scale_features

        # ==================================================
        # GENERAL
        # ==================================================

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

        self.fitted = False

        # ==================================================
        # SUPERVISED
        # ==================================================

        self.label_mappings = {}
        self.global_target_mean = None

        # Feature selection is intentionally disabled.
        self.selected_features = []

        # ==================================================
        # UNSUPERVISED
        # ==================================================

        self.one_hot_categories = {}
        self.unsupervised_feature_columns = []

    # ======================================================
    # CATEGORICAL FEATURES
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
    # TASK DETECTION
    # ======================================================

    def _detect_task(self, y):

        if (
            pd.api.types.is_object_dtype(y)
            or pd.api.types.is_string_dtype(y)
            or pd.api.types.is_bool_dtype(y)
        ):
            return "classification"

        if pd.api.types.is_numeric_dtype(y):

            if y.nunique() <= 20:
                return "classification"

            return "regression"

        return "classification"

    # ======================================================
    # ID DETECTION
    # ======================================================

    def _detect_ids(self, X):

        self.id_cols = []

        for col in X.columns:

            if X[col].nunique(dropna=False) == len(X):

                self.id_cols.append(col)

        return self.id_cols

    # ======================================================
    # MISSING VALUE DETECTION
    # ======================================================

    def _detect_missing_features(self, X):

        self.features_with_nan = [
            col
            for col in X.columns
            if X[col].isnull().sum() > 0
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

    # ======================================================
    # MISSING INDICATORS
    # ======================================================

    def _add_missing_indicators(self, X):

        X = X.copy()

        for feature in self.features_with_nan:

            if feature not in X.columns:
                continue

            X[f"{feature}_nan"] = (
                X[feature].isnull().astype(int)
            )

        return X

    # ======================================================
    # IMPUTATION
    # ======================================================

    def _fit_imputation(self, X):

        self.train_medians = {}

        for feature in self.features_nan_num:

            if feature not in X.columns:
                continue

            median_value = X[feature].median()

            if pd.isna(median_value):
                median_value = 0

            self.train_medians[feature] = median_value

    def _apply_imputation(self, X):

        X = X.copy()

        for feature in self.features_nan_num:

            if feature not in X.columns:
                continue

            value = self.train_medians.get(
                feature,
                0
            )

            X[feature] = X[feature].fillna(value)

        for feature in self.features_nan_cat:

            if feature not in X.columns:
                continue

            X[feature] = (
                X[feature]
                .fillna("Missing")
                .astype(str)
            )

        return X

    # ======================================================
    # RARE CATEGORIES
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

            frequent = frequencies[
                frequencies >= self.rare_label_threshold
            ].index.tolist()

            self.frequent_labels[feature] = frequent

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
    # SKEWNESS
    # ======================================================

    def _fit_skewness(self, X):

        self.skewed_features = []
        self.skew_shifts = {}

        numerical_features = [
            col
            for col in X.columns
            if pd.api.types.is_numeric_dtype(X[col])
        ]

        for feature in numerical_features:

            series = X[feature].dropna()

            if len(series) < 3:
                continue

            skew_value = stats.skew(series)

            if abs(skew_value) > self.skew_threshold:

                self.skewed_features.append(feature)

                minimum = series.min()

                if minimum < 0:
                    self.skew_shifts[feature] = (
                        abs(minimum) + 1
                    )
                else:
                    self.skew_shifts[feature] = 0

    def _apply_skewness(self, X):

        X = X.copy()

        for feature in self.skewed_features:

            if feature not in X.columns:
                continue

            shift = self.skew_shifts.get(
                feature,
                0
            )

            if shift > 0:

                X[feature] = np.log1p(
                    X[feature] + shift
                )

            else:

                X[feature] = np.log1p(
                    X[feature]
                )

        return X

    # ======================================================
    # SUPERVISED TARGET ENCODING
    # ======================================================

    def _fit_target_encoding(self, X, y):

        categorical_features = (
            self._get_categorical_features(X)
        )

        self.global_target_mean = y.mean()

        self.label_mappings = {}

        for feature in categorical_features:

            temp = pd.DataFrame({
                "category": X[feature].values,
                "target": y.values
            })

            grouped = (
                temp
                .groupby("category")["target"]
                .agg(["sum", "count"])
            )

            self.label_mappings[feature] = (
                grouped.to_dict("index")
            )

    def _apply_target_encoding_train(
        self,
        X,
        y
    ):

        X = X.copy()

        y_array = np.asarray(y)

        for feature in self.categorical_features:

            if feature not in X.columns:
                continue

            mapping = self.label_mappings[feature]

            encoded_values = []

            for i, category in enumerate(
                X[feature].values
            ):

                if category in mapping:

                    category_sum = mapping[
                        category
                    ]["sum"]

                    category_count = mapping[
                        category
                    ]["count"]

                    other_sum = (
                        category_sum - y_array[i]
                    )

                    other_count = (
                        category_count - 1
                    )

                    if other_count > 0:

                        value = (
                            other_sum / other_count
                        )

                    else:

                        value = self.global_target_mean

                else:

                    value = self.global_target_mean

                encoded_values.append(value)

            X[feature] = encoded_values

        return X

    def _apply_target_encoding(self, X):

        X = X.copy()

        for feature in self.categorical_features:

            if feature not in X.columns:
                continue

            mapping = self.label_mappings[feature]

            def encode_value(category):

                if category not in mapping:
                    return self.global_target_mean

                category_sum = mapping[
                    category
                ]["sum"]

                category_count = mapping[
                    category
                ]["count"]

                return category_sum / category_count

            X[feature] = (
                X[feature]
                .map(encode_value)
                .fillna(self.global_target_mean)
            )

        return X

    # ======================================================
    # UNSUPERVISED ONE-HOT ENCODING
    # ======================================================

    def _fit_one_hot_encoding(self, X):

        self.one_hot_categories = {}

        categorical_features = (
            self._get_categorical_features(X)
        )

        for feature in categorical_features:

            categories = (
                X[feature]
                .astype(str)
                .unique()
                .tolist()
            )

            self.one_hot_categories[feature] = categories

    def _apply_one_hot_encoding(self, X):

        X = X.copy()

        for feature, categories in (
            self.one_hot_categories.items()
        ):

            if feature not in X.columns:

                for category in categories:

                    X[
                        f"{feature}_{category}"
                    ] = 0

                continue

            values = X[feature].astype(str)

            for category in categories:

                X[
                    f"{feature}_{category}"
                ] = (
                    values == str(category)
                ).astype(int)

            X = X.drop(
                columns=[feature]
            )

        return X

    # ======================================================
    # NUMERIC VALIDATION
    # ======================================================

    def _ensure_numeric(self, X):

        X = X.copy()

        categorical_features = (
            self._get_categorical_features(X)
        )

        if categorical_features:

            raise ValueError(
                "Categorical features remain "
                "before scaling: "
                f"{categorical_features}"
            )

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
                "Non-numeric features remain "
                "before scaling: "
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

        X = X[self.feature_columns]

        return X

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

        if not self.scale_features:
            return X

        if self.scaler is None:
            return X

        if not self.scalable_features:
            return X

        X[self.scalable_features] = (
            self.scaler.transform(
                X[self.scalable_features]
            )
        )

        return X

    # ======================================================
    # SUPERVISED FIT
    # ======================================================

    def fit(
        self,
        X_train,
        y_train
    ):

        if y_train is None:

            raise ValueError(
                "Target values are required "
                "for supervised preprocessing."
            )

        X_train = X_train.copy()

        self._detect_ids(X_train)

        X = X_train.drop(
            columns=self.id_cols,
            errors="ignore"
        )

        self._detect_missing_features(X)

        self._fit_imputation(X)

        self._fit_rare_labels(X)

        X_temp = self._apply_imputation(X)

        X_temp = self._apply_rare_labels(
            X_temp
        )

        self._fit_skewness(X_temp)

        X_temp = self._apply_skewness(
            X_temp
        )

        self._fit_target_encoding(
            X_temp,
            y_train
        )

        X_temp = (
            self._apply_target_encoding_train(
                X_temp,
                y_train
            )
        )

        X_temp = self._ensure_numeric(
            X_temp
        )

        self.feature_columns = (
            X_temp.columns.tolist()
        )

        self._fit_scaler(X_temp)

        X_temp = self._apply_scaling(
            X_temp
        )

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

        ids = pd.DataFrame(
            index=X.index
        )

        for col in self.id_cols:

            if col in X.columns:
                ids[col] = X[col]

        X = X.drop(
            columns=self.id_cols,
            errors="ignore"
        )

        X = self._add_missing_indicators(X)

        X = self._apply_imputation(X)

        X = self._apply_rare_labels(X)

        X = self._apply_skewness(X)

        X = self._apply_target_encoding(X)

        X = self._align_features(X)

        X = self._ensure_numeric(X)

        X = self._apply_scaling(X)

        return X, ids

    # ======================================================
    # SUPERVISED FIT TRANSFORM
    # ======================================================

    def fit_transform(
        self,
        X_train,
        y_train
    ):

        self.fit(
            X_train,
            y_train
        )

        X = X_train.copy()

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
            y_train
        )

        X = self._align_features(X)

        X = self._ensure_numeric(X)

        X = self._apply_scaling(X)

        return X

    # ======================================================
    # UNSUPERVISED FIT
    # ======================================================

    def fit_unsupervised(self, X):

        X = X.copy()

        # ----------------------------------------------
        # ID detection
        # ----------------------------------------------

        self._detect_ids(X)

        X = X.drop(
            columns=self.id_cols,
            errors="ignore"
        )

        # ----------------------------------------------
        # Missing values
        # ----------------------------------------------

        self._detect_missing_features(X)

        self._fit_imputation(X)

        X = self._apply_imputation(X)

        # ----------------------------------------------
        # Rare categories
        # ----------------------------------------------

        self._fit_rare_labels(X)

        X = self._apply_rare_labels(X)

        # ----------------------------------------------
        # Skewness
        # ----------------------------------------------

        self._fit_skewness(X)

        X = self._apply_skewness(X)

        # ----------------------------------------------
        # One-hot encoding
        # ----------------------------------------------

        self._fit_one_hot_encoding(X)

        X = self._apply_one_hot_encoding(X)

        # ----------------------------------------------
        # Numeric validation
        # ----------------------------------------------

        X = self._ensure_numeric(X)

        # ----------------------------------------------
        # Save feature structure
        # ----------------------------------------------

        self.feature_columns = (
            X.columns.tolist()
        )

        self.unsupervised_feature_columns = (
            X.columns.tolist()
        )

        # ----------------------------------------------
        # Scaling
        # ----------------------------------------------

        self._fit_scaler(X)

        self.task = "unsupervised"

        # No target-based feature selection.
        self.selected_features = (
            self.feature_columns.copy()
        )

        self.fitted = True

        return self

    # ======================================================
    # UNSUPERVISED TRANSFORM
    # ======================================================

    def transform_unsupervised(self, X):

        if not self.fitted:

            raise RuntimeError(
                "Unsupervised pipeline has "
                "not been fitted yet."
            )

        X = X.copy()

        ids = pd.DataFrame(
            index=X.index
        )

        # ----------------------------------------------
        # Preserve IDs
        # ----------------------------------------------

        for col in self.id_cols:

            if col in X.columns:
                ids[col] = X[col]

        # ----------------------------------------------
        # Remove IDs
        # ----------------------------------------------

        X = X.drop(
            columns=self.id_cols,
            errors="ignore"
        )

        # ----------------------------------------------
        # Missing indicators
        # ----------------------------------------------

        X = self._add_missing_indicators(X)

        # ----------------------------------------------
        # Imputation
        # ----------------------------------------------

        X = self._apply_imputation(X)

        # ----------------------------------------------
        # Rare categories
        # ----------------------------------------------

        X = self._apply_rare_labels(X)

        # ----------------------------------------------
        # Skewness
        # ----------------------------------------------

        X = self._apply_skewness(X)

        # ----------------------------------------------
        # One-hot encoding
        # ----------------------------------------------

        X = self._apply_one_hot_encoding(X)

        # ----------------------------------------------
        # Align columns
        # ----------------------------------------------

        X = self._align_features(X)

        # ----------------------------------------------
        # Numeric validation
        # ----------------------------------------------

        X = self._ensure_numeric(X)

        # ----------------------------------------------
        # Scaling
        # ----------------------------------------------

        X = self._apply_scaling(X)

        return X, ids

    # ======================================================
    # UNSUPERVISED FIT TRANSFORM
    # ======================================================

    def fit_transform_unsupervised(self, X):

        self.fit_unsupervised(X)

        X_processed, _ = (
            self.transform_unsupervised(X)
        )

        return X_processed

    # ======================================================
    # PIPELINE INFORMATION
    # ======================================================

    def get_info(self):

        if not self.fitted:

            raise RuntimeError(
                "Pipeline has not been fitted yet."
            )

        return {

            "task":
                self.task,

            "target":
                (
                    self.target_col
                    if self.target_col is not None
                    else "None"
                ),

            "id_columns":
                self.id_cols,

            "missing_value_features":
                self.features_with_nan,

            "numeric_missing_features":
                self.features_nan_num,

            "categorical_missing_features":
                self.features_nan_cat,

            "skewed_features":
                self.skewed_features,

            "scaled_features":
                self.scalable_features,

            "original_feature_count":
                len(self.feature_columns),

            "selected_feature_count":
                len(self.feature_columns),

            "selected_features":
                self.feature_columns.copy(),

            "feature_selection_method":
                "None"
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


def process_supervised_dataset(df, target_col, test_size=0.20, random_state=42):

    from sklearn.model_selection import train_test_split

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    processor = SupervisedPreprocessor(target_col=target_col)
    X_train_processed = processor.fit_transform(X_train, y_train)
    X_test_processed, test_ids = processor.transform(X_test)

    train_output = X_train_processed.copy()
    test_output = X_test_processed.copy()

    train_ids = pd.DataFrame(index=X_train.index)
    for col in processor.id_cols:
        if col in X_train.columns:
            train_ids[col] = X_train[col]

    if not train_ids.empty:
        train_output = pd.concat([train_ids.reset_index(drop=True), train_output.reset_index(drop=True)], axis=1)
    if not test_ids.empty:
        test_output = pd.concat([test_ids.reset_index(drop=True), test_output.reset_index(drop=True)], axis=1)

    train_output[target_col] = y_train.reset_index(drop=True)
    test_output[target_col] = y_test.reset_index(drop=True)

    info = processor.get_info()
    info.update({"task": processor.task, "target": target_col, "dataset_type": "Entire Dataset", "rows_processed": len(df), "feature_selection_method": "None"})

    return {"X_train": train_output, "X_test": test_output, "y_train": y_train, "y_test": y_test, "processor": processor, "task": processor.task, "info": info}


def process_unsupervised_dataset(df, test_size=0.20, random_state=42):

    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    processor = UnsupervisedPreprocessor()
    X_train_processed, train_ids = processor.fit_transform(train_df)
    X_test_processed, test_ids = processor.transform(test_df)

    train_output = X_train_processed.copy()
    test_output = X_test_processed.copy()

    if not train_ids.empty:
        train_output = pd.concat([train_ids.reset_index(drop=True), train_output.reset_index(drop=True)], axis=1)
    if not test_ids.empty:
        test_output = pd.concat([test_ids.reset_index(drop=True), test_output.reset_index(drop=True)], axis=1)

    info = processor.get_info()
    info.update({"task": "unsupervised", "target": "None", "dataset_type": "Entire Dataset", "rows_processed": len(df), "feature_selection_method": "None"})

    return {"X_train": train_output, "X_test": test_output, "processor": processor, "task": "unsupervised", "info": info}
