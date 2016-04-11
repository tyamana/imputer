import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNC, KNeighborsRegressor as KNR
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error as MSE
from math import sqrt
from copy import copy
from sklearn import svm
from sklearn.linear_model import LogisticRegression as lr


__author__ = 'Artem Zhirokhov'


class Imputer:
    """
    Imputation transformer for completing missing values.

    :param missing_values: The placeholder for the missing values. All occurrences of missing_values will be imputed.
        For missing values encoded as np.nan, use the string value “NaN”;
    :type missing_values: integer or “NaN”; optional (default=”NaN”)
    :param strategy: The imputation strategy.
        If strategy=mean, then impute mean value of feature.
        If strategy=class_mean, then impute mean value along class.
        If strategy=class_mean, then impute median value along class.
        If strategy=knn, then impute values, according to knn model with the best parameters, that would be
            defined during fitting.
        If strategy=svm, then impute values, according to svm model.
        If strategy=logistic_regr, then impute values, according to logistic regression model.
        [not implemented]If strategy=knn_auto, then impute values, according to knn model with the best parameters, that would be
            defined during fitting.
        [not implemented]If strategy=knn_custom, then impute values, according to knn model with user's parameters
    :type strategy: string, optional (default=”mean”)
    :param axis: The axis along which to impute.
        If axis=0, then impute along columns.
        If axis=1, then impute along rows.
    :type axis: integer, optional (default=0)
    :param verbose: Controls the verbosity of the imputer.
    :type verbose: integer, optional (default=0)
    :param copy: If True, a copy of X will be created. If False, imputation will be done in-place whenever possible.
    :type copy: boolean, optional (default=True)
    """

    def __init__(self, missing_values='NaN', strategy='mean', axis=0, verbose=0, copy=True):
        self.missing_values = missing_values
        self.strategy = strategy
        self.axis = axis
        self.verbose = verbose
        self.copy = copy

        self._is_fitted = False
        self._mask = []
        self._clr = None
        self._rgr = None
        self._devided_features = None

    """
    Main methods
    """

    def fit(self, X, y=None):
        if self.strategy == 'mean':
            self._mean_fit(X, y=y)
        elif self.strategy == 'class_mean':
            self._class_mean_fit(X, y=y)
        elif self.strategy == 'class_median':
            self._class_median_fit(X, y=y)

        self._devided_features = self._devide_features_to_classifiable_and_regressiable(X, 5)

        if self.strategy == 'knn':
            self._knn_fit(X, y=y)
        elif self.strategy == 'svm':
            self._svm_fit(X, y=y)
        elif self.strategy == 'logistic_regr':
            self._lr_fit(X, y=y)

        self._is_fitted = True

        return self

    def fit_transform(self, X, y=None):
        self.fit(X, y=y)
        return self.transform(X, y)

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        :return dict
        """

        params = {}
        params['missing_values'] = self.missing_values
        params['strategy'] = self.strategy
        params['axis'] = self.axis
        params['verbose'] = self.verbose
        params['copy'] = self.copy

        return params

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        :return self
        """
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)

        return self

    def transform(self, X, y=None):
        assert self._is_fitted, 'fit the estimator before transform'

        if self.strategy == 'mean':
            assert X.shape[1] == len(self._mask),\
                'inappropriate dimension of data - {0} instead of {1}'.format(X.shape[1], len(self._mask))

            dict_of_values_to_set = self._get_dict_of_values_to_set(X)

            return X.fillna(dict_of_values_to_set)

        X_new = X.copy()

        if self.strategy.startswith('class'):
            assert y is not None, 'y must be given for class mean strategy'

            nan_values = X.isnull()
            for column_name in X_new.columns.values:

                indices = list(nan_values[nan_values[column_name] == True].index.values)
                for i in indices:
                    class_value = y.loc[i]
                    if class_value in self._mean_values_mask.keys():
                        X_new.set_value(i, column_name, self._mean_values_mask[class_value][column_name])

        self._devided_features = self._devide_features_to_classifiable_and_regressiable(X, 5)

        if self.strategy == 'knn' or self.strategy == 'svm' or self.strategy == 'logistic_regr':

            for column_name in self._devided_features['class']:
                current_X_columns = copy(list(X.columns.values))
                current_X_columns.remove(column_name)

                current_X, _, _, X_test = self._get_X_and_y_by_column_name_with_imputs(X,
                                                                                       current_X_columns,
                                                                                       column_name)

                self._use_clr_to_fill_na(current_X, X_test, column_name, X_new)

            for column_name in self._devided_features['regr']:
                current_X_columns = copy(list(X.columns.values))
                current_X_columns.remove(column_name)

                current_X, _, _, X_test = self._get_X_and_y_by_column_name_with_imputs(X,
                                                                                       current_X_columns,
                                                                                       column_name)

                self._use_rgr_to_fill_na(current_X, X_test, column_name, X_new)

        return X_new

    """
    Fit methods
    """

    def _mean_fit(self, X, y=None):
        for column in X.columns.values:
            self._mask.append(X[column].mean())

    def _class_mean_fit(self, X, y=None):
        assert y is not None, 'y must be given for class mean strategy'
        classes_dataframes = self._get_dataframes_per_class(X, y)
        self._mean_values_mask = self._get_mask_of_values_per_class(X, y, classes_dataframes, 'mean')

    def _class_median_fit(self, X, y=None):
        assert y is not None, 'y must be given for class median strategy'
        classes_dataframes = self._get_dataframes_per_class(X, y)
        self._mean_values_mask = self._get_mask_of_values_per_class(X, y, classes_dataframes, 'median')

    def _svm_fit(self, X, y=None):
        for column_name in self._devided_features['class']:
            current_X_columns = list(X.columns)
            current_X_columns.remove(column_name)

            current_X, X_train, y_train, _ = self._get_X_and_y_by_column_name_with_imputs(X,
                                                                                          current_X_columns,
                                                                                          column_name)

            scaler = StandardScaler().fit(current_X)
            self._clr = svm.SVC().fit(scaler.transform(X_train), y_train)

        for column_name in self._devided_features['regr']:
            current_X_columns = list(X.columns)
            current_X_columns.remove(column_name)

            current_X, X_train, y_train, _ = self._get_X_and_y_by_column_name_with_imputs(X,
                                                                                          current_X_columns,
                                                                                          column_name)

            scaler = StandardScaler().fit(current_X)
            self._rgr = svm.SVR().fit(scaler.transform(X_train), y_train)

    def _knn_fit(self, X, y=None):

        for column_name in self._devided_features['class']:
            current_X_columns = copy(list(X.columns.values))
            current_X_columns.remove(column_name)

            current_X, X_train, y_train, X_test = self._get_X_and_y_by_column_name_with_imputs(X,
                                                                                               current_X_columns,
                                                                                               column_name)

            max_acc = 0
            best_neigh_num = 0
            epsilon = 0.005

            for neighbors_num in range(3, 100, 3):
                knn = KNC(metric='manhattan', n_neighbors=neighbors_num)
                X_scaled = StandardScaler().fit_transform(X_train)
                mean_acc_score = cross_val_score(knn, X_scaled, y_train, cv=5).mean()

                if mean_acc_score > max_acc + epsilon:
                    max_acc = mean_acc_score
                    best_neigh_num = neighbors_num

            if self.verbose is True:
                print(column_name, max_acc, best_neigh_num)

            self._clr = KNC(metric='manhattan', n_neighbors=best_neigh_num).fit(StandardScaler().fit_transform(X_train),
                                                                                y_train)

        for column_name in self._devided_features['regr']:
            current_X_columns = copy(list(X.columns.values))
            current_X_columns.remove(column_name)

            current_X, X_train, y_train, X_test = self._get_X_and_y_by_column_name_with_imputs(X,
                                                                                               current_X_columns,
                                                                                               column_name)

            X_tr, X_te, y_tr, y_te = train_test_split(X_train, y_train, test_size=0.33, random_state=22)

            min_mse = 0.0
            best_neigh_num = 0
            best_weights = ''
            epsilon = 0.005

            for neighbors_num in range(3, min(100, int(len(current_X.index.values)*0.67)), 3):
                knn_uniform = KNR(n_neighbors=neighbors_num)
                knn_distance = KNR(weights='distance', n_neighbors=neighbors_num)

                scaler = StandardScaler().fit(current_X)

                X_tr_scaled = scaler.transform(X_tr)
                knn_uniform.fit(X_tr_scaled, y_tr)
                knn_distance.fit(X_tr_scaled, y_tr)

                X_te_scaled = scaler.transform(X_te)
                y_pred = knn_uniform.predict(X_te_scaled)
                mse_un = sqrt(MSE(y_te, y_pred))
                y_pred = knn_distance.predict(X_te_scaled)
                mse_di = sqrt(MSE(y_te, y_pred))

                if min(mse_di, mse_un) < min_mse - epsilon or neighbors_num == 3:
                    min_mse = min(mse_di, mse_un)
                    if mse_di < mse_un:
                        best_weights = 'distance'
                    else:
                        best_weights = 'uniform'
                    best_neigh_num = neighbors_num

            if self.verbose is True:
                print(column_name, min_mse, best_neigh_num, best_weights)

            self._rgr = KNR(weights=best_weights,
                            n_neighbors=best_neigh_num).fit(StandardScaler().fit_transform(X_train), y_train)

    def _lr_fit(self, X, y=None):

        for column_name in self._devided_features['class']:
            current_X_columns = list(X.columns)
            current_X_columns.remove(column_name)

            current_X, X_train, y_train, _ = self._get_X_and_y_by_column_name_with_imputs(X,
                                                                                          current_X_columns,
                                                                                          column_name)

            scaler = StandardScaler().fit(current_X)
            scaled_X = scaler.fit_transform(X_train)

            newton_acc = cross_val_score(lr(solver='newton-cg'), scaled_X, y_train, cv=5).mean()
            lbfgs_acc = cross_val_score(lr(solver='lbfgs'), scaled_X, y_train, cv=5).mean()
            liblinear_acc = cross_val_score(lr(solver='liblinear'), scaled_X, y_train, cv=5).mean()
            sag_acc = cross_val_score(lr(solver='sag'), scaled_X, y_train, cv=5).mean()

            acc_map = {newton_acc: 'newton-cg', lbfgs_acc: 'lbfgs', liblinear_acc: 'liblinear', sag_acc: 'sag'}
            max_acc = max(newton_acc, lbfgs_acc, liblinear_acc, sag_acc)

            self._clr = lr(solver=acc_map[max_acc]).fit(scaler.transform(X_train), y_train)

        for column_name in self._devided_features['regr']:
            current_X_columns = list(X.columns)
            current_X_columns.remove(column_name)

            current_X, X_train, y_train, _ = self._get_X_and_y_by_column_name_with_imputs(X,
                                                                                          current_X_columns,
                                                                                          column_name)

            scaler = StandardScaler().fit(current_X)
            self._rgr = lr(solver='sag').fit(scaler.transform(X_train), y_train)

    """
    Supporting methods
    """

    def _get_X_and_y_by_column_name_with_imputs(self, X, current_X_columns, column_name):
        current_X = pd.DataFrame(Imputer(strategy='mean').fit_transform(X[current_X_columns]),
                                 columns=current_X_columns)

        current_y_nan_values = X[column_name].isnull()

        y_train = X[column_name][current_y_nan_values == False]
        X_train = current_X[current_X_columns][current_y_nan_values == False]
        X_test = current_X[current_X_columns][current_y_nan_values == True]

        return (current_X, X_train, y_train, X_test)

    def _get_mask_of_values_per_class(self, X, y, classes_dataframes, strategy):
        classes = y.value_counts().index.values.tolist()

        mean_values_mask = {}
        for column_name in X.columns.values:
            for class_value in classes:

                column_class_value = float
                if strategy == 'mean':
                    column_class_value = classes_dataframes[class_value][column_name].dropna().mean()
                elif strategy == 'median':
                    column_class_value = classes_dataframes[class_value][column_name].dropna().median()

                if column_class_value != np.nan: #учесть случай когда все сэмплы со значением класса пустые

                    if class_value not in mean_values_mask.keys():
                        mean_values_mask[class_value] = {column_name: column_class_value}
                    else:
                        mean_values_mask[class_value][column_name] = column_class_value

        return mean_values_mask

    def _get_dataframes_per_class(self, X, y):
        classes = y.value_counts().index.values.tolist()

        data = X.copy()
        data['target'] = y

        classes_dataframes = {}
        for class_value in classes:
            classes_dataframes[class_value] = data[data['target'] == class_value][X.columns]

        return classes_dataframes

    def _use_clr_to_fill_na(self, current_X, X_test, column_name, X_new):
        if X_test.empty is False:
            scaler = StandardScaler().fit(current_X)
            y_test = self._clr.predict(scaler.transform(X_test))
            X_test_indices = list(X_test.index.values)
            counter = 0
            for index in X_test_indices:
                X_new.set_value(index, column_name, y_test[counter])
                counter += 1

    def _use_rgr_to_fill_na(self, current_X, X_test, column_name, X_new):
        if X_test.empty is False:
            scaler = StandardScaler().fit(current_X)
            y_test = self._rgr.predict(scaler.transform(X_test))
            X_test_indices = list(X_test.index.values)
            counter = 0
            for index in X_test_indices:
                X_new.set_value(index, column_name, y_test[counter])
                counter += 1

    def _get_dict_of_values_to_set(self, X):
        dict_of_values = {}
        index_in_mask = 0

        for column in X.columns.values:
            dict_of_values[column] = self._mask[index_in_mask]
            index_in_mask += 1

        return dict_of_values

    def _devide_features_to_classifiable_and_regressiable(self, df, max_number_of_values_per_class):
        columns = list(df.columns.values)
        devided_features = {'class': [], 'regr': []}
        for column in columns:
            if len(df[column].value_counts().index) < max_number_of_values_per_class:
                devided_features['class'].append(column)
            else:
                devided_features['regr'].append(column)
        return devided_features