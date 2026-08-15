import pandas as pd
import numpy as np

from scipy import stats

from sklearn.preprocessing import MinMaxScaler, LabelEncoder


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

        # --------------------------------------------------
        # Notebook-based supervised feature engineering state
        # --------------------------------------------------

        self.feature_count_before_processing = 0

        self.log_features = []
        self.log_target = False

        self.outlier_bounds = {}

        self.near_zero_variance_cols = []
        self.multicollinearity_drop_cols = []

        self.categorical_order_mappings = {}
        self.categorical_order_fallback = {}

        self.target_log_enabled = False

        self.fitted = False

        # ==================================================
        # SUPERVISED
        # ==================================================

        self.label_mappings = {}
        self.global_target_mean = None

        # Target encoding state for classification.
        self.target_classes = []
        self.target_class_mapping = {}
        self.global_target_priors = {}

        # Label encoder for categorical classification targets.
        # Fitted on training labels only.
        self.target_encoder = None
        self.target_was_label_encoded = False

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

            # Numeric target with fewer than 10 unique values
            # is treated as classification; otherwise regression.
            if y.nunique() < 10:
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
                frequencies > self.rare_label_threshold
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
    # TARGET LABEL ENCODING
    # ======================================================

    def _fit_target_encoder(self, y):

        self.target_encoder = None
        self.target_was_label_encoded = False

        is_categorical_target = (
            pd.api.types.is_object_dtype(y)
            or pd.api.types.is_string_dtype(y)
            or pd.api.types.is_categorical_dtype(y)
            or pd.api.types.is_bool_dtype(y)
        )

        if (
            self.task == "classification"
            and is_categorical_target
        ):

            y_series = pd.Series(
                y
            ).reset_index(drop=True)

            if y_series.isna().any():
                raise ValueError(
                    "Target column contains missing values. "
                    "Categorical classification targets must be "
                    "complete before label encoding."
                )

            self.target_encoder = LabelEncoder()
            self.target_encoder.fit(
                y_series.astype(str)
            )

            self.target_classes = (
                self.target_encoder.classes_.tolist()
            )

            self.target_class_mapping = {
                class_value: index
                for index, class_value
                in enumerate(self.target_classes)
            }

            self.target_was_label_encoded = True

    def encode_target(self, y):
        """Return the target in the representation used by the model."""

        y_series = pd.Series(
            y
        ).reset_index(drop=True)

        if self.target_encoder is None:
            return y_series.to_numpy()

        try:
            return self.target_encoder.transform(
                y_series.astype(str).to_numpy()
            )

        except ValueError as exc:
            raise ValueError(
                "The target contains a class that was not present "
                "in the training target."
            ) from exc

    def get_target_classes(self):

        if self.target_encoder is None:
            return []

        return self.target_encoder.classes_.tolist()

    # ======================================================
    # TARGET LABEL ENCODING
    # ======================================================

    def _fit_target_encoder(self, y):

        self.target_encoder = None
        self.target_was_label_encoded = False
        self.target_classes = []
        self.target_class_mapping = {}

        is_categorical_target = (
            pd.api.types.is_object_dtype(y)
            or pd.api.types.is_string_dtype(y)
            or pd.api.types.is_categorical_dtype(y)
            or pd.api.types.is_bool_dtype(y)
        )

        if (
            self.task == "classification"
            and is_categorical_target
        ):

            y_series = pd.Series(
                y
            ).reset_index(drop=True)

            if y_series.isna().any():

                raise ValueError(
                    "Target column contains missing values. "
                    "Categorical classification targets must "
                    "be complete before label encoding."
                )

            self.target_encoder = LabelEncoder()

            self.target_encoder.fit(
                y_series.astype(str)
            )

            self.target_classes = (
                self.target_encoder.classes_.tolist()
            )

            self.target_class_mapping = {
                class_value: class_index
                for class_index, class_value
                in enumerate(self.target_classes)
            }

            self.target_was_label_encoded = True

    def encode_target(self, y):

        """Return the final target representation used by the notebook flow."""

        y_series = pd.Series(
            y
        ).reset_index(drop=True)

        # Classification categorical target -> LabelEncoder.
        if self.target_encoder is not None:

            try:

                return self.target_encoder.transform(
                    y_series.astype(str).to_numpy()
                )

            except ValueError as exc:

                raise ValueError(
                    "The target contains a class that was not present "
                    "in the training target."
                ) from exc

        # Regression -> notebook's log transformation of the target
        # whenever the training target was strictly positive.
        if (
            self.task == "regression"
            and self.target_log_enabled
        ):

            values = pd.to_numeric(
                y_series,
                errors="coerce"
            )

            if values.isna().any():

                raise ValueError(
                    "Regression target contains non-numeric or "
                    "missing values."
                )

            if (values <= 0).any():

                raise ValueError(
                    "The regression target contains zero or negative "
                    "values, but the notebook target-log transformation "
                    "was fitted on strictly positive training values."
                )

            return np.log(
                values.to_numpy()
            )

        values = pd.to_numeric(
            y_series,
            errors="coerce"
        )

        if values.isna().any():

            raise ValueError(
                "Numeric regression target contains missing or "
                "non-numeric values."
            )

        return values.to_numpy()

    def get_target_classes(self):

        if self.target_encoder is None:
            return []

        return self.target_encoder.classes_.tolist()

    # ======================================================
    # NOTEBOOK-BASED SUPERVISED FEATURES
    # ======================================================

    def _fit_log_features(self, X, y):

        notebook_numeric_features = [
            "LotFrontage",
            "LotArea",
            "1stFlrSF",
            "GrLivArea"
        ]

        self.log_features = []

        for feature in notebook_numeric_features:

            if feature not in X.columns:
                continue

            series = X[feature].dropna()

            if series.empty:
                continue

            # Notebook skips a feature when 0 is present.
            # Guard against negatives as well so np.log never creates
            # invalid values in a generalized dataset.
            if (series <= 0).any():
                continue

            self.log_features.append(feature)

        # The notebook also logs SalePrice. For generalized regression,
        # do the same to any strictly-positive numeric target.
        self.target_log_enabled = False

        if self.task == "regression":

            y_numeric = pd.to_numeric(
                y,
                errors="coerce"
            )

            if (
                not y_numeric.isna().any()
                and len(y_numeric) > 0
                and (y_numeric > 0).all()
            ):
                self.target_log_enabled = True

    def _apply_log_features(self, X):

        X = X.copy()

        for feature in self.log_features:

            if feature not in X.columns:
                continue

            X[feature] = np.log(
                X[feature]
            )

        return X

    def _fit_outlier_bounds(self, X):

        self.outlier_bounds = {}

        for feature in self.log_features:

            series = X[feature].dropna()

            if series.empty:
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            lower = (
                q1
                -
                1.5 * iqr
            )

            upper = (
                q3
                +
                1.5 * iqr
            )

            self.outlier_bounds[feature] = (
                float(lower),
                float(upper)
            )

    def _apply_outlier_clipping(self, X):

        X = X.copy()

        for feature, (lower, upper) in (
            self.outlier_bounds.items()
        ):

            if feature not in X.columns:
                continue

            X[feature] = X[feature].clip(
                lower,
                upper
            )

        return X

    def _fit_near_zero_variance(self, X):

        self.near_zero_variance_cols = []

        for feature in X.columns:

            top_frequency = (
                X[feature]
                .value_counts(
                    normalize=True,
                    dropna=False
                )
            )

            if top_frequency.empty:
                continue

            if (
                top_frequency.iloc[0]
                >
                0.99
            ):

                self.near_zero_variance_cols.append(
                    feature
                )

    def _fit_multicollinearity(
        self,
        X,
        y_processed
    ):

        self.multicollinearity_drop_cols = []

        # Recreate the notebook's correlation matrix:
        # numerical features + target.
        corr_data = X.copy()

        corr_data["__target_for_corr__"] = (
            y_processed
        )

        corr_matrix = corr_data.corr(
            numeric_only=True
        )

        if corr_matrix.empty:
            return

        threshold = 0.90

        high_corr_pairs = []

        columns = corr_matrix.columns.tolist()

        for i in range(len(columns)):

            for j in range(i):

                col_i = columns[i]
                col_j = columns[j]

                if (
                    abs(
                        corr_matrix.iloc[i, j]
                    )
                    >
                    threshold
                ):

                    high_corr_pairs.append(
                        (
                            col_i,
                            col_j,
                            corr_matrix.iloc[i, j]
                        )
                    )

        drop_cols = set()

        for col_i, col_j, _ in high_corr_pairs:

            # Never remove the actual target from X.
            if col_i == "__target_for_corr__":

                drop_candidate = col_j

            elif col_j == "__target_for_corr__":

                drop_candidate = col_i

            else:

                corr_i_target = abs(
                    corr_matrix.loc[
                        col_i,
                        "__target_for_corr__"
                    ]
                )

                corr_j_target = abs(
                    corr_matrix.loc[
                        col_j,
                        "__target_for_corr__"
                    ]
                )

                if corr_i_target < corr_j_target:

                    drop_candidate = col_i

                else:

                    drop_candidate = col_j

            if drop_candidate in X.columns:

                drop_cols.add(
                    drop_candidate
                )

        self.multicollinearity_drop_cols = (
            sorted(drop_cols)
        )

    def _apply_supervised_feature_drops(self, X):

        X = X.copy()

        drop_cols = set(
            self.near_zero_variance_cols
            +
            self.multicollinearity_drop_cols
        )

        if drop_cols:

            X = X.drop(
                columns=list(drop_cols),
                errors="ignore"
            )

        return X

    def _fit_target_order_encoding(
        self,
        X,
        y_processed
    ):

        self.categorical_order_mappings = {}
        self.categorical_order_fallback = {}

        categorical_features = (
            self._get_categorical_features(X)
        )

        y_series = pd.Series(
            y_processed
        ).reset_index(drop=True)

        for feature in categorical_features:

            temp = pd.DataFrame({
                "category":
                    X[feature].reset_index(
                        drop=True
                    ),

                "target":
                    y_series
            })

            # EXACT NOTEBOOK IDEA:
            # category -> mean target -> sorted -> ordinal rank.
            means = (
                temp
                .groupby("category")["target"]
                .mean()
                .sort_values()
            )

            mapping = {
                category: rank
                for rank, category
                in enumerate(
                    means.index
                )
            }

            self.categorical_order_mappings[
                feature
            ] = mapping

            # Unseen categories in test data do not occur in the
            # notebook. Use a deterministic fallback rather than
            # generating NaN/non-numeric values.
            if mapping:

                self.categorical_order_fallback[
                    feature
                ] = int(
                    round(
                        (len(mapping) - 1) / 2
                    )
                )

            else:

                self.categorical_order_fallback[
                    feature
                ] = 0

    def _apply_target_order_encoding(self, X):

        X = X.copy()

        for feature, mapping in (
            self.categorical_order_mappings.items()
        ):

            if feature not in X.columns:
                continue

            fallback = (
                self.categorical_order_fallback.get(
                    feature,
                    0
                )
            )

            X[feature] = (
                X[feature]
                .map(mapping)
                .fillna(fallback)
                .astype(float)
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

        y_train = pd.Series(
            y_train,
            index=X_train.index
        )

        self.task = self._detect_task(
            y_train
        )

        if self.task == "regression":

            y_numeric = pd.to_numeric(
                y_train,
                errors="coerce"
            )

            if y_numeric.isna().any():

                raise ValueError(
                    "Regression target contains missing or "
                    "non-numeric values."
                )

            y_train = y_numeric

        self._fit_target_encoder(
            y_train
        )

        # y_processed is calculated after the notebook's target-log
        # decision is made so multicollinearity uses the same target
        # representation as the notebook.
        y_processed = None

        # Feature count is measured on the input feature matrix,
        # excluding the target.
        self.feature_count_before_processing = (
            X_train.shape[1]
        )

        # Notebook flow: ID detection -> missing handling.
        self._detect_ids(X_train)

        X = X_train.drop(
            columns=self.id_cols,
            errors="ignore"
        )

        self._detect_missing_features(X)

        self._fit_imputation(X)

        X = self._add_missing_indicators(
            X
        )

        X = self._apply_imputation(
            X
        )

        # Notebook flow: temporal feature engineering.
        X = self._apply_temporal_features(
            X,
            fit=True
        )

        # Notebook flow: selected log transforms.
        self._fit_log_features(
            X,
            y_train
        )

        y_processed = self.encode_target(
            y_train
        )

        X = self._apply_log_features(
            X
        )

        # Notebook flow: IQR clipping.
        self._fit_outlier_bounds(
            X
        )

        X = self._apply_outlier_clipping(
            X
        )

        # Notebook flow: rare categories.
        self._fit_rare_labels(
            X
        )

        X = self._apply_rare_labels(
            X
        )

        # Notebook flow: near-zero variance.
        self._fit_near_zero_variance(
            X
        )

        X = self._apply_supervised_feature_drops(
            X
        )

        # Notebook flow: multicollinearity.
        self._fit_multicollinearity(
            X,
            y_processed
        )

        X = self._apply_supervised_feature_drops(
            X
        )

        # After numerical feature removal, rebuild the categorical
        # feature list exactly as the notebook does.
        self.categorical_features = (
            self._get_categorical_features(X)
        )

        # Notebook's final categorical encoding:
        # ordered target mean -> integer rank.
        self._fit_target_order_encoding(
            X,
            y_processed
        )

        X = self._apply_target_order_encoding(
            X
        )

        X = self._ensure_numeric(
            X
        )

        # Final feature matrix structure is now frozen.
        self.feature_columns = (
            X.columns.tolist()
        )

        self.selected_features = (
            self.feature_columns.copy()
        )

        # Notebook uses MinMaxScaler on every feature except target.
        self._fit_scaler(
            X
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

        # Missing indicators use exactly the features that were
        # detected during training.
        X = self._add_missing_indicators(
            X
        )

        X = self._apply_imputation(
            X
        )

        X = self._apply_temporal_features(
            X,
            fit=False
        )

        X = self._apply_log_features(
            X
        )

        X = self._apply_outlier_clipping(
            X
        )

        X = self._apply_rare_labels(
            X
        )

        X = self._apply_supervised_feature_drops(
            X
        )

        X = self._apply_target_order_encoding(
            X
        )

        X = self._align_features(
            X
        )

        X = self._ensure_numeric(
            X
        )

        X = self._apply_scaling(
            X
        )

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

        X = self._add_missing_indicators(
            X
        )

        X = self._apply_imputation(
            X
        )

        X = self._apply_temporal_features(
            X,
            fit=False
        )

        X = self._apply_log_features(
            X
        )

        X = self._apply_outlier_clipping(
            X
        )

        X = self._apply_rare_labels(
            X
        )

        X = self._apply_supervised_feature_drops(
            X
        )

        X = self._apply_target_order_encoding(
            X
        )

        X = self._align_features(
            X
        )

        X = self._ensure_numeric(
            X
        )

        X = self._apply_scaling(
            X
        )

        return X

    # ======================================================
    # SUPERVISED TEMPORAL FEATURES
    # ======================================================

    def _apply_temporal_features(
        self,
        X,
        fit=False
    ):

        X = X.copy()

        if "YrSold" not in X.columns:
            return X

        for feature in [
            "YearBuilt",
            "YearRemodAdd",
            "GarageYrBlt"
        ]:

            if feature in X.columns:

                X[feature] = (
                    X["YrSold"]
                    -
                    X[feature]
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
    # UNSUPERVISED NOTEBOOK FLOW
    # ======================================================

    def _fit_unsupervised_log_features(self, X):

        notebook_numeric_features = [
            "LotFrontage",
            "LotArea",
            "1stFlrSF",
            "GrLivArea"
        ]

        self.unsupervised_log_features = []

        for feature in notebook_numeric_features:

            if feature not in X.columns:
                continue

            series = X[feature].dropna()

            if series.empty:
                continue

            # Match notebook: skip if a zero exists.
            if (series == 0).any():
                continue

            # General safety for arbitrary numeric tables.
            if (series <= 0).any():
                continue

            self.unsupervised_log_features.append(
                feature
            )

    def _apply_unsupervised_log_features(self, X):

        X = X.copy()

        for feature in getattr(
            self,
            "unsupervised_log_features",
            []
        ):

            if feature not in X.columns:
                continue

            X[feature] = np.log(
                X[feature]
            )

        return X

    def _fit_unsupervised_outliers(self, X):

        self.unsupervised_outlier_bounds = {}

        for feature in getattr(
            self,
            "unsupervised_log_features",
            []
        ):

            series = X[feature].dropna()

            if series.empty:
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)

            iqr = q3 - q1

            lower = (
                q1
                -
                1.5 * iqr
            )

            upper = (
                q3
                +
                1.5 * iqr
            )

            self.unsupervised_outlier_bounds[
                feature
            ] = (
                float(lower),
                float(upper)
            )

    def _apply_unsupervised_outliers(self, X):

        X = X.copy()

        for feature, (lower, upper) in (
            getattr(
                self,
                "unsupervised_outlier_bounds",
                {}
            ).items()
        ):

            if feature not in X.columns:
                continue

            X[feature] = X[feature].clip(
                lower,
                upper
            )

        return X

    def _fit_unsupervised_nzv(self, X):

        self.unsupervised_near_zero_cols = []

        for feature in X.columns:

            frequencies = (
                X[feature]
                .value_counts(
                    normalize=True,
                    dropna=False
                )
            )

            if frequencies.empty:
                continue

            # Exact notebook rule: strictly greater than 0.99.
            if frequencies.iloc[0] > 0.99:

                self.unsupervised_near_zero_cols.append(
                    feature
                )

    def _fit_unsupervised_multicollinearity(self, X):

        self.unsupervised_multicollinearity_drop_cols = []

        corr_matrix = X.corr(
            numeric_only=True
        )

        if corr_matrix.empty:
            return

        threshold = 0.90

        columns = corr_matrix.columns.tolist()

        for i in range(len(columns)):

            for j in range(i):

                if (
                    abs(
                        corr_matrix.iloc[i, j]
                    )
                    >
                    threshold
                ):

                    # EXACT NOTEBOOK BEHAVIOUR:
                    # cols_to_drop.add(colname_j)
                    # so the earlier/first column is kept
                    # and the later/second column is dropped.
                    column_to_drop = columns[j]

                    if (
                        column_to_drop
                        not in
                        self.unsupervised_multicollinearity_drop_cols
                    ):

                        self.unsupervised_multicollinearity_drop_cols.append(
                            column_to_drop
                        )

    def _apply_unsupervised_structure_drops(self, X):

        X = X.copy()

        drop_cols = (
            list(
                getattr(
                    self,
                    "unsupervised_near_zero_cols",
                    []
                )
            )
            +
            list(
                getattr(
                    self,
                    "unsupervised_multicollinearity_drop_cols",
                    []
                )
            )
        )

        if drop_cols:

            X = X.drop(
                columns=drop_cols,
                errors="ignore"
            )

        return X

    def _fit_frequency_encoding(self, X):

        self.unsupervised_frequency_mappings = {}
        self.unsupervised_frequency_fallbacks = {}

        categorical_features = (
            self._get_categorical_features(X)
        )

        for feature in categorical_features:

            # Exact notebook idea:
            # sort categories by descending frequency,
            # then assign 0, 1, 2, ...
            ordered_categories = (
                X[feature]
                .value_counts()
                .sort_values(
                    ascending=False
                )
                .index
            )

            mapping = {
                category: rank
                for rank, category
                in enumerate(
                    ordered_categories,
                    start=0
                )
            }

            self.unsupervised_frequency_mappings[
                feature
            ] = mapping

            if mapping:

                self.unsupervised_frequency_fallbacks[
                    feature
                ] = int(
                    round(
                        (len(mapping) - 1) / 2
                    )
                )

            else:

                self.unsupervised_frequency_fallbacks[
                    feature
                ] = 0

    def _apply_frequency_encoding(self, X):

        X = X.copy()

        for feature, mapping in (
            self.unsupervised_frequency_mappings.items()
        ):

            if feature not in X.columns:
                continue

            fallback = (
                self.unsupervised_frequency_fallbacks.get(
                    feature,
                    0
                )
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

        # Original notebook starts by identifying duplicate rows.
        # In the application, this is handled on the complete uploaded
        # dataframe at the workflow boundary. We retain the pipeline's
        # fit/transform contract here and do not silently alter test
        # row counts inside transform.

        self.feature_count_before_processing = (
            X.shape[1]
        )

        # --------------------------------------------------
        # ID detection
        # --------------------------------------------------

        self._detect_ids(X)

        X = X.drop(
            columns=self.id_cols,
            errors="ignore"
        )

        # --------------------------------------------------
        # Missing indicators + missing values
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
        # Temporal feature engineering
        # --------------------------------------------------

        X = self._apply_temporal_features(
            X,
            fit=True
        )

        # --------------------------------------------------
        # Selected notebook log features
        # --------------------------------------------------

        self._fit_unsupervised_log_features(
            X
        )

        X = self._apply_unsupervised_log_features(
            X
        )

        # --------------------------------------------------
        # IQR clipping
        # --------------------------------------------------

        self._fit_unsupervised_outliers(
            X
        )

        X = self._apply_unsupervised_outliers(
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
        # Near-zero variance
        # --------------------------------------------------

        self._fit_unsupervised_nzv(
            X
        )

        X = self._apply_unsupervised_structure_drops(
            X
        )

        # --------------------------------------------------
        # Multicollinearity
        # --------------------------------------------------

        self._fit_unsupervised_multicollinearity(
            X
        )

        X = self._apply_unsupervised_structure_drops(
            X
        )

        # --------------------------------------------------
        # Frequency-ordered categorical encoding
        # --------------------------------------------------

        self.categorical_features = (
            self._get_categorical_features(X)
        )

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

        # --------------------------------------------------
        # Freeze final feature structure
        # --------------------------------------------------

        self.feature_columns = (
            X.columns.tolist()
        )

        self.unsupervised_feature_columns = (
            self.feature_columns.copy()
        )

        self.selected_features = (
            self.feature_columns.copy()
        )

        # --------------------------------------------------
        # Scaling
        # --------------------------------------------------

        self._fit_scaler(
            X
        )

        self.task = "unsupervised"

        self.fitted = True

        return self

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

        # --------------------------------------------------
        # Preserve IDs
        # --------------------------------------------------

        for col in self.id_cols:

            if col in X.columns:

                ids[col] = X[col]

        X = X.drop(
            columns=self.id_cols,
            errors="ignore"
        )

        # --------------------------------------------------
        # Missing indicators / imputation
        # --------------------------------------------------

        X = self._add_missing_indicators(
            X
        )

        X = self._apply_imputation(
            X
        )

        # --------------------------------------------------
        # Temporal features
        # --------------------------------------------------

        X = self._apply_temporal_features(
            X,
            fit=False
        )

        # --------------------------------------------------
        # Log + outliers
        # --------------------------------------------------

        X = self._apply_unsupervised_log_features(
            X
        )

        X = self._apply_unsupervised_outliers(
            X
        )

        # --------------------------------------------------
        # Rare categories
        # --------------------------------------------------

        X = self._apply_rare_labels(
            X
        )

        # --------------------------------------------------
        # Structural drops
        # --------------------------------------------------

        X = self._apply_unsupervised_structure_drops(
            X
        )

        # --------------------------------------------------
        # Frequency encoding
        # --------------------------------------------------

        X = self._apply_frequency_encoding(
            X
        )

        # --------------------------------------------------
        # Align / numeric / scale
        # --------------------------------------------------

        X = self._align_features(
            X
        )

        X = self._ensure_numeric(
            X
        )

        X = self._apply_scaling(
            X
        )

        return X, ids

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
                self.log_features,

            "scaled_features":
                self.scalable_features,

            "original_feature_count":
                self.feature_count_before_processing,

            "selected_feature_count":
                len(self.feature_columns),

            "feature_count_before_processing":
                self.feature_count_before_processing,

            "final_feature_count":
                len(self.feature_columns),

            "selected_features":
                self.feature_columns.copy(),

            # Keep the existing application's wording; the actual
            # notebook-based reductions are recorded separately below.
            "feature_selection_method":
                "None",

            "notebook_near_zero_variance_removed":
                self.near_zero_variance_cols,

            "notebook_multicollinearity_removed":
                self.multicollinearity_drop_cols,

            "notebook_log_features":
                self.log_features,

            "notebook_target_log":
                self.target_log_enabled,

            "notebook_outlier_clipped_features":
                list(
                    self.outlier_bounds.keys()
                ),

            "notebook_target_encoding":
                (
                    "Target-mean ordered categorical encoding"
                    if self.categorical_order_mappings
                    else "None"
                ),

            "target_label_encoding":
                (
                    "LabelEncoder"
                    if self.target_was_label_encoded
                    else "None"
                ),

            "target_classes":
                self.get_target_classes(),

            "unsupervised_log_features":
                getattr(
                    self,
                    "unsupervised_log_features",
                    []
                ),

            "unsupervised_outlier_clipped_features":
                list(
                    getattr(
                        self,
                        "unsupervised_outlier_bounds",
                        {}
                    ).keys()
                ),

            "unsupervised_near_zero_variance_removed":
                getattr(
                    self,
                    "unsupervised_near_zero_cols",
                    []
                ),

            "unsupervised_multicollinearity_removed":
                getattr(
                    self,
                    "unsupervised_multicollinearity_drop_cols",
                    []
                ),

            "unsupervised_categorical_encoding":
                (
                    "Frequency-ordered encoding"
                    if getattr(
                        self,
                        "unsupervised_frequency_mappings",
                        {}
                    )
                    else "None"
                )
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


def process_supervised_dataset(
    df,
    target_col,
    test_size=0.20,
    random_state=42
):

    from sklearn.model_selection import train_test_split

    if target_col not in df.columns:

        raise ValueError(
            f"Target column '{target_col}' not found."
        )

    df = df.copy()

    # Notebook flow begins by removing duplicate rows.
    df = df.drop_duplicates().reset_index(drop=True)

    X = df.drop(
        columns=[target_col]
    ).copy()

    y = df[target_col].copy()

    # Split BEFORE fitting any learned preprocessing state.
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

    train_output = (
        X_train_processed.copy()
    )

    test_output = (
        X_test_processed.copy()
    )

    # Preserve the same ID behavior already expected by the app:
    # IDs are removed from model features but returned to the user.
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

    # Targets are never used as X features.
    # They are returned in processed form:
    # - LabelEncoder for categorical classification
    # - notebook log transform for positive regression targets
    train_output[target_col] = (
        processor.encode_target(
            y_train
        )
    )

    test_output[target_col] = (
        processor.encode_target(
            y_test
        )
    )

    # Put target last, matching the application's current output
    # convention while preserving all processed feature columns.
    train_output = train_output[
        [
            c for c in train_output.columns
            if c != target_col
        ]
        + [target_col]
    ]

    test_output = test_output[
        [
            c for c in test_output.columns
            if c != target_col
        ]
        + [target_col]
    ]

    info = processor.get_info()

    info.update(
        {
            "task":
                processor.task,

            "target":
                target_col,

            "dataset_type":
                "Entire Dataset",

            "rows_processed":
                len(df),

            "feature_selection_method":
                "None"
        }
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

        "task":
            processor.task,

        "info":
            info
    }



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