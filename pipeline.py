import pandas as pd
import numpy as np

from scipy import stats

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split


# ==========================================================
# CORE DATA PREPROCESSOR
# ==========================================================

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
        # GENERAL STATE
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
        # SUPERVISED STATE
        # ==================================================

        self.label_mappings = {}
        self.global_target_mean = None

        self.feature_selector = None
        self.selected_features = []

        # ==================================================
        # UNSUPERVISED STATE
        # ==================================================

        self.one_hot_categories = {}
        self.unsupervised_feature_columns = []

        # ==================================================
        # FEATURE COUNTS
        # ==================================================

        self.feature_count_before_processing = 0
        self.feature_count_after_encoding = 0
        self.feature_count_after_selection = 0

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
                or isinstance(
                    X[col].dtype,
                    pd.CategoricalDtype
                )
            )
        ]

    # ======================================================
    # TASK DETECTION
    # ======================================================

    def detect_task(self, y):

        if (
            pd.api.types.is_object_dtype(y)
            or pd.api.types.is_string_dtype(y)
            or pd.api.types.is_bool_dtype(y)
        ):
            return "classification"

        if pd.api.types.is_numeric_dtype(y):

            if y.nunique(dropna=True) <= 20:
                return "classification"

            return "regression"

        return "classification"

    # Backward-compatible internal name.
    def _detect_task(self, y):
        return self.detect_task(y)

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
    # MISSING VALUE INDICATORS
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
    # RARE CATEGORY HANDLING
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

        for feature, frequent in (
            self.frequent_labels.items()
        ):

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

        self.categorical_features = (
            categorical_features
        )

        self.global_target_mean = y.mean()

        self.label_mappings = {}

        for feature in categorical_features:

            temp = pd.DataFrame({
                "category": X[feature].values,
                "target": np.asarray(y)
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

                    category_sum = (
                        mapping[category]["sum"]
                    )

                    category_count = (
                        mapping[category]["count"]
                    )

                    # Leave-one-out target encoding:
                    # the current row's target is excluded.
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

                        value = (
                            self.global_target_mean
                        )

                else:

                    value = (
                        self.global_target_mean
                    )

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

                category_sum = (
                    mapping[category]["sum"]
                )

                category_count = (
                    mapping[category]["count"]
                )

                return (
                    category_sum / category_count
                )

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

            self.one_hot_categories[feature] = (
                categories
            )

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
    # SUPERVISED FEATURE SELECTION
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

        self.feature_selector = SelectFromModel(
            estimator=estimator,
            threshold="mean"
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

            raise ValueError(
                "Feature selection removed all "
                "features. Try changing "
                "lasso_alpha or logistic_c."
            )

    def _apply_feature_selection(self, X):

        X = X.copy()

        for feature in self.selected_features:

            if feature not in X.columns:
                X[feature] = 0

        return X[self.selected_features]

    # ======================================================
    # SUPERVISED FIT
    # ======================================================

    def fit(self, X_train, y_train):

        if y_train is None:

            raise ValueError(
                "Target values are required "
                "for supervised preprocessing."
            )

        X_train = X_train.copy()

        self.feature_count_before_processing = (
            X_train.shape[1]
        )

        self.task = (
            self.task
            if self.task is not None
            else self.detect_task(y_train)
        )

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

        self.feature_count_after_encoding = (
            len(self.feature_columns)
        )

        self._fit_scaler(X_temp)

        X_temp = self._apply_scaling(
            X_temp
        )

        self._fit_feature_selector(
            X_temp,
            y_train
        )

        self.feature_count_after_selection = (
            len(self.selected_features)
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

        X = self._apply_feature_selection(X)

        return X, ids

    # ======================================================
    # SUPERVISED FIT TRANSFORM
    # ======================================================

    def fit_transform(self, X_train, y_train):

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

        X = self._apply_feature_selection(X)

        return X

    # ======================================================
    # UNSUPERVISED FIT
    # ======================================================

    def fit_unsupervised(self, X):

        X = X.copy()

        self.feature_count_before_processing = (
            X.shape[1]
        )

        self._detect_ids(X)

        X = X.drop(
            columns=self.id_cols,
            errors="ignore"
        )

        self._detect_missing_features(X)

        self._fit_imputation(X)

        X = self._apply_imputation(X)

        self._fit_rare_labels(X)

        X = self._apply_rare_labels(X)

        self._fit_skewness(X)

        X = self._apply_skewness(X)

        self._fit_one_hot_encoding(X)

        X = self._apply_one_hot_encoding(X)

        X = self._ensure_numeric(X)

        self.feature_columns = (
            X.columns.tolist()
        )

        self.unsupervised_feature_columns = (
            X.columns.tolist()
        )

        self.feature_count_after_encoding = (
            len(self.feature_columns)
        )

        self._fit_scaler(X)

        self.task = "unsupervised"

        self.selected_features = (
            self.feature_columns.copy()
        )

        self.feature_count_after_selection = (
            len(self.selected_features)
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

        X = self._apply_one_hot_encoding(X)

        X = self._align_features(X)

        X = self._ensure_numeric(X)

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

            "feature_count_before_processing":
                self.feature_count_before_processing,

            "feature_count_after_encoding":
                self.feature_count_after_encoding,

            "original_feature_count":
                self.feature_count_before_processing,

            "selected_feature_count":
                self.feature_count_after_selection,

            "final_feature_count":
                self.feature_count_after_selection,

            "selected_features":
                self.selected_features,

            "feature_selection_method":
                (
                    "L1-based feature selection"
                    if self.task in [
                        "classification",
                        "regression"
                    ]
                    else "Not applicable"
                )
        }


# ==========================================================
# COMPATIBILITY CLASSES
# ==========================================================

class SupervisedPreprocessor(DataPreprocessor):

    def __init__(
        self,
        target_col=None,
        **kwargs
    ):

        super().__init__(
            target_col=target_col,
            **kwargs
        )


class UnsupervisedPreprocessor(DataPreprocessor):

    def __init__(
        self,
        test_size=0.20,
        random_state=42,
        **kwargs
    ):

        super().__init__(
            target_col=None,
            task="unsupervised",
            **kwargs
        )

        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y=None):

        return self.fit_unsupervised(X)

    def transform(self, X):

        return self.transform_unsupervised(X)

    def fit_transform(self, X, y=None):

        self.fit_unsupervised(X)

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
            f"Target column '{target_col}' "
            "not found in dataset."
        )

    if not 0 < test_size < 1:

        raise ValueError(
            "test_size must be between 0 and 1."
        )

    X = df.drop(
        columns=[target_col]
    )

    y = df[target_col]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state
        )
    )

    processor = SupervisedPreprocessor(
        target_col=target_col
    )

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

    train_ids = pd.DataFrame(
        index=X_train.index
    )

    for col in processor.id_cols:

        if col in X_train.columns:
            train_ids[col] = X_train[col]

    train_output = (
        X_train_processed.reset_index(
            drop=True
        )
    )

    if not train_ids.empty:

        train_output = pd.concat(
            [
                train_ids.reset_index(
                    drop=True
                ),
                train_output
            ],
            axis=1
        )

    train_output[target_col] = (
        y_train.reset_index(
            drop=True
        )
    )

    test_output = (
        X_test_processed.reset_index(
            drop=True
        )
    )

    if not test_ids.empty:

        test_output = pd.concat(
            [
                test_ids.reset_index(
                    drop=True
                ),
                test_output
            ],
            axis=1
        )

    return {

        "X_train":
            train_output,

        "X_test":
            test_output,

        "y_train":
            y_train,

        "y_test":
            y_test,

        "processor":
            processor,

        "info":
            processor.get_info(),

        "task":
            processor.task
    }


# ==========================================================
# UNSUPERVISED DATASET PROCESSOR
# ==========================================================

def process_unsupervised_dataset(
    df,
    test_size=0.20,
    random_state=42
):

    if not 0 < test_size < 1:

        raise ValueError(
            "test_size must be between 0 and 1."
        )

    X_train, X_test = train_test_split(
        df.copy(),
        test_size=test_size,
        random_state=random_state
    )

    processor = UnsupervisedPreprocessor(
        test_size=test_size,
        random_state=random_state
    )

    X_train_processed, train_ids = (
        processor.fit_transform(
            X_train
        )
    )

    X_test_processed, test_ids = (
        processor.transform(
            X_test
        )
    )

    train_output = (
        X_train_processed.reset_index(
            drop=True
        )
    )

    if not train_ids.empty:

        train_output = pd.concat(
            [
                train_ids.reset_index(
                    drop=True
                ),
                train_output
            ],
            axis=1
        )

    test_output = (
        X_test_processed.reset_index(
            drop=True
        )
    )

    if not test_ids.empty:

        test_output = pd.concat(
            [
                test_ids.reset_index(
                    drop=True
                ),
                test_output
            ],
            axis=1
        )

    return {

        "X_train":
            train_output,

        "X_test":
            test_output,

        "processor":
            processor,

        "info":
            processor.get_info()
    }
