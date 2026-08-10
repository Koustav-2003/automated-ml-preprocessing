import pandas as pd
import numpy as np

from scipy import stats

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel


class DataPreprocessor:
    """
    Universal supervised preprocessing pipeline.

    Pipeline:
        1. Detect ID-like columns
        2. Create missing-value indicators
        3. Handle numerical missing values
        4. Handle categorical missing values
        5. Group rare categorical labels
        6. Handle skewed numerical features
        7. Target-guided categorical encoding
        8. MinMax scaling
        9. L1-based feature selection

    Supported:
        - Regression
        - Classification

    IMPORTANT:
        All learned preprocessing parameters come ONLY from
        training data.

        Test/new data is transformed using those learned
        parameters and never used to fit anything.
    """

    def __init__(
        self,
        target_col,
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
        # LEARNED PARAMETERS
        # ==================================================

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

        self.feature_selector = None

        self.selected_features = []

        self.scaler = None

        self.fitted = False

    # ======================================================
    # CATEGORICAL DETECTION
    # ======================================================

    def _get_categorical_features(self, X):
        """
        Universal categorical-column detection.

        Handles:
            - object
            - string
            - category
        """

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

        """
        Automatically determine regression vs classification.

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

            if y.nunique() <= 20:
                return "classification"

            return "regression"

        return "classification"

    # ======================================================
    # ID DETECTION
    # ======================================================

    def _detect_ids(self, X):

        """
        Detect columns where every value is unique.
        """

        self.id_cols = []

        for col in X.columns:

            if X[col].nunique(dropna=False) == len(X):

                self.id_cols.append(col)

        return self.id_cols

    # ======================================================
    # MISSING VALUE INDICATORS
    # ======================================================

    def _fit_missing_indicators(self, X):

        self.features_with_nan = [
            col
            for col in X.columns
            if X[col].isnull().sum() > 0
        ]

        categorical_features = (
            self._get_categorical_features(X)
        )

        self.features_nan_cat = [
            col
            for col in self.features_with_nan
            if col in categorical_features
        ]

        self.features_nan_num = [
            col
            for col in self.features_with_nan
            if pd.api.types.is_numeric_dtype(X[col])
        ]

    def _add_missing_indicators(self, X):

        X = X.copy()

        for feature in self.features_with_nan:

            if feature in X.columns:

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

            self.train_medians[feature] = (
                X[feature].median()
            )

    def _apply_imputation(self, X):

        X = X.copy()

        # ------------------------------------------
        # Categorical
        # ------------------------------------------

        for feature in self.features_nan_cat:

            if feature in X.columns:

                X[feature] = X[feature].fillna(
                    "Missing"
                )

        # ------------------------------------------
        # Numerical
        # ------------------------------------------

        for feature, median in self.train_medians.items():

            if feature in X.columns:

                X[feature] = X[feature].fillna(
                    median
                )

        return X

    # ======================================================
    # RARE LABEL GROUPING
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
    # TARGET ENCODING
    # ======================================================

    def _fit_target_encoding(self, X, y):

        """
        Learn target means from TRAINING DATA ONLY.
        """

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

        """
        Leave-one-out target encoding for training data.
        """

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
                            other_sum /
                            other_count
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

        """
        Target encoding for test/new data.

        Uses only mappings learned from training data.
        """

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

                return (
                    category_sum /
                    category_count
                )

            X[feature] = (
                X[feature]
                .map(encode_value)
                .fillna(
                    self.global_target_mean
                )
            )

        return X

    # ======================================================
    # FINAL CATEGORICAL SAFETY
    # ======================================================

    def _ensure_numeric(self, X):

        """
        Final safety check.

        No categorical/string column is allowed to reach
        scaling or feature selection.
        """

        categorical_features = (
            self._get_categorical_features(X)
        )

        if categorical_features:

            raise ValueError(
                "Categorical features remain after "
                "target encoding: "
                f"{categorical_features}"
            )

        # Convert boolean columns to integers
        boolean_features = [
            col
            for col in X.columns
            if pd.api.types.is_bool_dtype(X[col])
        ]

        for feature in boolean_features:

            X[feature] = (
                X[feature].astype(int)
            )

        # Final numeric validation
        non_numeric_features = [
            col
            for col in X.columns
            if not pd.api.types.is_numeric_dtype(X[col])
        ]

        if non_numeric_features:

            raise ValueError(
                "Non-numeric features remain before "
                "scaling: "
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

        X = X[
            self.feature_columns
        ]

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

        X[
            self.scalable_features
        ] = self.scaler.transform(
            X[self.scalable_features]
        )

        return X

    # ======================================================
    # FEATURE SELECTION
    # ======================================================

    def _fit_feature_selector(
        self,
        X,
        y
    ):

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

        if len(self.selected_features) == 0:

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

        return X[
            self.selected_features
        ]

    # ======================================================
    # FIT
    # ======================================================

    def fit(
        self,
        X_train,
        y_train
    ):

        """
        FIT ONLY on training data.
        """

        X_train = X_train.copy()

        y_train = pd.Series(
            y_train,
            index=X_train.index
        )

        if len(X_train) != len(y_train):

            raise ValueError(
                "X_train and y_train must have "
                "the same number of rows."
            )

        # ------------------------------------------
        # Task
        # ------------------------------------------

        if self.task is None:

            self.task = self._detect_task(
                y_train
            )

        # ------------------------------------------
        # IDs
        # ------------------------------------------

        self._detect_ids(
            X_train
        )

        X = X_train.drop(
            columns=self.id_cols,
            errors="ignore"
        )

        # ------------------------------------------
        # Missing indicators
        # ------------------------------------------

        self._fit_missing_indicators(X)

        X = self._add_missing_indicators(X)

        # ------------------------------------------
        # Imputation
        # ------------------------------------------

        self._fit_imputation(X)

        X = self._apply_imputation(X)

        # ------------------------------------------
        # Rare labels
        # ------------------------------------------

        self._fit_rare_labels(X)

        X = self._apply_rare_labels(X)

        # ------------------------------------------
        # Skew
        # ------------------------------------------

        self._fit_skewness(X)

        X = self._apply_skewness(X)

        # ------------------------------------------
        # Target encoding
        # ------------------------------------------

        self._fit_target_encoding(
            X,
            y_train
        )

        X = self._apply_target_encoding_train(
            X,
            y_train
        )

        # ------------------------------------------
        # Final numeric conversion/check
        # ------------------------------------------

        X = self._ensure_numeric(X)

        # ------------------------------------------
        # Save feature structure
        # ------------------------------------------

        self.feature_columns = (
            X.columns.tolist()
        )

        # ------------------------------------------
        # Scaling
        # ------------------------------------------

        self._fit_scaler(X)

        X = self._apply_scaling(X)

        # ------------------------------------------
        # Feature selection
        # ------------------------------------------

        self._fit_feature_selector(
            X,
            y_train
        )

        self.fitted = True

        return self

    # ======================================================
    # TRANSFORM
    # ======================================================

    def transform(
        self,
        X
    ):

        if not self.fitted:

            raise RuntimeError(
                "Pipeline has not been fitted. "
                "Call fit(X_train, y_train) first."
            )

        X = X.copy()

        # ------------------------------------------
        # Save IDs
        # ------------------------------------------

        ids = pd.DataFrame(
            index=X.index
        )

        for col in self.id_cols:

            if col in X.columns:

                ids[col] = X[col]

        # ------------------------------------------
        # Remove IDs
        # ------------------------------------------

        X = X.drop(
            columns=self.id_cols,
            errors="ignore"
        )

        # ------------------------------------------
        # Missing indicators
        # ------------------------------------------

        X = self._add_missing_indicators(X)

        # ------------------------------------------
        # Imputation
        # ------------------------------------------

        X = self._apply_imputation(X)

        # ------------------------------------------
        # Rare labels
        # ------------------------------------------

        X = self._apply_rare_labels(X)

        # ------------------------------------------
        # Skew
        # ------------------------------------------

        X = self._apply_skewness(X)

        # ------------------------------------------
        # Target encoding
        # ------------------------------------------

        X = self._apply_target_encoding(X)

        # ------------------------------------------
        # Align
        # ------------------------------------------

        X = self._align_features(X)

        # ------------------------------------------
        # Numeric safety
        # ------------------------------------------

        X = self._ensure_numeric(X)

        # ------------------------------------------
        # Scaling
        # ------------------------------------------

        X = self._apply_scaling(X)

        # ------------------------------------------
        # Feature selection
        # ------------------------------------------

        X = self._apply_feature_selection(X)

        return X, ids

    # ======================================================
    # FIT TRANSFORM
    # ======================================================

    def fit_transform(
        self,
        X_train,
        y_train
    ):

        """
        Fit pipeline on training data and return
        processed training data.

        Uses leave-one-out target encoding.
        """

        self.fit(
            X_train,
            y_train
        )

        X = X_train.copy()

        # ------------------------------------------
        # Remove IDs
        # ------------------------------------------

        X = X.drop(
            columns=self.id_cols,
            errors="ignore"
        )

        # ------------------------------------------
        # Missing indicators
        # ------------------------------------------

        X = self._add_missing_indicators(X)

        # ------------------------------------------
        # Imputation
        # ------------------------------------------

        X = self._apply_imputation(X)

        # ------------------------------------------
        # Rare labels
        # ------------------------------------------

        X = self._apply_rare_labels(X)

        # ------------------------------------------
        # Skew
        # ------------------------------------------

        X = self._apply_skewness(X)

        # ------------------------------------------
        # Target encoding
        # ------------------------------------------

        X = self._apply_target_encoding_train(
            X,
            y_train
        )

        # ------------------------------------------
        # Align
        # ------------------------------------------

        X = self._align_features(X)

        # ------------------------------------------
        # Numeric safety
        # ------------------------------------------

        X = self._ensure_numeric(X)

        # ------------------------------------------
        # Scaling
        # ------------------------------------------

        X = self._apply_scaling(X)

        # ------------------------------------------
        # Feature selection
        # ------------------------------------------

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

            "task":
                self.task,

            "target":
                self.target_col,

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
                len(self.selected_features),

            "selected_features":
                self.selected_features
        }


# ==========================================================
# LOCAL TEST
# ==========================================================

if __name__ == "__main__":

    from sklearn.model_selection import train_test_split

    INPUT_FILE = "train.csv"

    TARGET_COL = "SalePrice"

    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Dataset shape: {df.shape}"
    )

    X = df.drop(
        columns=[TARGET_COL]
    )

    y = df[TARGET_COL]

    processor = DataPreprocessor(
        target_col=TARGET_COL
    )

    task = processor._detect_task(y)

    print(
        f"Detected task: {task}"
    )

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

    print(
        f"X_train: {X_train.shape}"
    )

    print(
        f"X_test : {X_test.shape}"
    )

    # ------------------------------------------------------
    # FIT + TRANSFORM TRAIN
    # ------------------------------------------------------

    print("\n" + "=" * 60)
    print("FITTING PIPELINE ON TRAINING DATA")
    print("=" * 60)

    X_train_processed = (
        processor.fit_transform(
            X_train,
            y_train
        )
    )

    # ------------------------------------------------------
    # TRANSFORM TEST
    # ------------------------------------------------------

    print("\n" + "=" * 60)
    print("TRANSFORMING TEST DATA")
    print("=" * 60)

    X_test_processed, test_ids = (
        processor.transform(
            X_test
        )
    )

    # ------------------------------------------------------
    # ADD TARGET
    # ------------------------------------------------------

    train_output = (
        X_train_processed.copy()
    )

    train_output[TARGET_COL] = (
        y_train.values
    )

    # ------------------------------------------------------
    # ADD IDS
    # ------------------------------------------------------

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

    test_output = X_test_processed.copy()

    if not test_ids.empty:

        test_output = pd.concat(
            [
                test_ids.reset_index(drop=True),
                test_output.reset_index(drop=True)
            ],
            axis=1
        )

    # ------------------------------------------------------
    # SAVE
    # ------------------------------------------------------

    train_output.to_csv(
        "X_train.csv",
        index=False
    )

    test_output.to_csv(
        "X_test.csv",
        index=False
    )

    # ------------------------------------------------------
    # INFO
    # ------------------------------------------------------

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)

    info = processor.get_info()

    print(
        f"Task: {info['task']}"
    )

    print(
        f"Detected IDs: {info['id_columns']}"
    )

    print(
        f"Original features: "
        f"{info['original_feature_count']}"
    )

    print(
        f"Selected features: "
        f"{info['selected_feature_count']}"
    )

    print(
        f"Skewed features: "
        f"{len(info['skewed_features'])}"
    )

    print("\nSelected features:")

    for feature in info["selected_features"]:

        print(
            f"  - {feature}"
        )

    print("\nFiles created:")

    print("X_train.csv")
    print("X_test.csv")