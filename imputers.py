import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNC, KNeighborsRegressor as KNR
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error as MSE
from math import sqrt
from copy import copy
from sklearn import svm
from sklearn.linear_model import LogisticRegression, Ridge
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


__author__ = 'Artem Zhirokhov'

def getMissingDataRate(df):

    N = df.shape[0] * df.shape[1]
    naValues = 0

    for label in df.columns:
        nanValInCol = df[label].isnull()
        if True in nanValInCol.value_counts().index:
            naValues += nanValInCol.value_counts()[True]

    return float(naValues) / N

class Imputer:
    """
    Imputation transformer for completing missing values.

    :param [not_implemented]missing_values: The placeholder for the missing values. All occurrences of missing_values will be imputed.
        For missing values encoded as np.nan, use the string value “NaN”;
    :type missing_values: integer or “NaN”; optional (default=”NaN”)
    :param strategy: The imputation strategy.
        If strategy=mean, then impute mean value of feature.
        If strategy=class_mean, then impute mean value along class.
        If strategy=class_median, then impute median value along class.
        If strategy=knn, then impute values, according to knn model with the best parameters, that would be
            defined during fitting.
        If strategy=svm, then impute values, according to svm model.
        If strategy=xgboost, then imputer uses xgboost model to imput, best parameters are tuned during fitting.
        If strategy=kmeans, then imputer uses K-means strategy to imput. Use **kwargs parameter of 'fit' method
            to set the paramters (same as sklearn.cluster.KMeans estimator).
        If strategy=random_forest then ...
        If strategy=class_knn then knn method will be used per each class (target array is required)
        If strategy=class_svm then svm method will be used per each class (target array is required)
        If strategy=class_xgboost then xgboost method will be used per each class (target array is required)
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
        self._classifiers = {}
        self._regressors = {}
        self._devided_features = None
        #label encoder for logistic regression in xgboost
        self._label_encoders = {}
        #for regression in xgboost
        self._label_scalers = {}

    """
    Main methods
    """

    def fit(self, X, y=None, **kwargs):

        self._devided_features = self._devide_features_to_classifiable_and_regressiable(X, 5)

        if self.strategy == 'mean':
            self._mean_fit(X, y=y)
        elif self.strategy == 'class_mean':
            self._class_mean_fit(X, y=y)
        elif self.strategy == 'class_median':
            self._class_median_fit(X, y=y)
        elif self.strategy == 'knn':
            self._knn_fit(X, y=y)
        elif self.strategy == 'svm':
            self._svm_fit(X, y=y)
        elif self.strategy == 'logistic_regr':
            self._lr_fit(X, y=y)
        elif self.strategy == 'random_forest':
            self._rf_fit(X, y=y)
        elif self.strategy == 'xgboost':
            self._xgb_fit(X, y=y)
        elif self.strategy == 'kmeans':
            self._kmeans_fit(X, y=y, **kwargs)
        elif self.strategy == 'class_knn':
            self._class_knn_svm_xgboost_fit(X, y=y)
        elif self.strategy == 'class_svm':
            self._class_knn_svm_xgboost_fit(X, y=y)
        elif self.strategy == 'class_xgboost':
            self._class_knn_svm_xgboost_fit(X, y=y)
        elif self.strategy == 'class_kmeans':
            self._class_kmeans_fit(X, y=y, **kwargs)


        self._is_fitted = True

        return self

    def fit_transform(self, X, y=None, **kwargs):
        self.fit(X, y=y, **kwargs)
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

        X_new = X.copy()
        self._devided_features = self._devide_features_to_classifiable_and_regressiable(X, 5)

        if self.strategy == 'mean':
            assert X.shape[1] == len(self._mask),\
                'inappropriate dimension of data - {0} instead of {1}'.format(X.shape[1], len(self._mask))

            dict_of_values_to_set = self._get_dict_of_values_to_set(X)

            return X.fillna(dict_of_values_to_set)

        elif self.strategy == 'class_knn' or self.strategy == 'class_svm' or self.strategy == 'class_xgboost' or\
             self.strategy == 'class_kmeans':
            X_new = self._class_knn_svm_xgboost_kmeans_transform(X, X_new, y)
        elif self.strategy.startswith('class'):
            self._class_transform(X_new, y)
        elif self.strategy == 'kmeans':
            X_new = self._kmeans_transform(X_new, y)

        elif self.strategy == 'knn' or self.strategy == 'svm' or self.strategy == 'logistic_regr' or\
             self.strategy == 'random_forest':
            self._knn_svm_lr_transform(X, X_new, y)
        elif self.strategy == 'xgboost':
            self._xgboost_transform(X, X_new, y)

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
            y_train = self._prevent_float_labeling(y_train, column_name)

            self._classifiers[column_name] = svm.SVC().fit(scaler.transform(X_train), y_train)

        for column_name in self._devided_features['regr']:
            current_X_columns = list(X.columns)
            current_X_columns.remove(column_name)

            current_X, X_train, y_train, _ = self._get_X_and_y_by_column_name_with_imputs(X,
                                                                                          current_X_columns,
                                                                                          column_name)

            scaler = StandardScaler().fit(current_X)
            self._regressors[column_name] = svm.SVR().fit(scaler.transform(X_train), y_train)

    def _knn_fit(self, X, y=None):

        for column_name in self._devided_features['class']:
            current_X_columns = copy(list(X.columns.values))
            current_X_columns.remove(column_name)

            current_X, X_train, y_train, _ = self._get_X_and_y_by_column_name_with_imputs(X,
                                                                                               current_X_columns,
                                                                                               column_name)

            scaler = StandardScaler().fit(current_X)
            max_acc = 0
            best_neigh_num = 0
            epsilon = 0.005

            top_border_of_neigh_num = min(50, int(X_train.shape[0] / 5)) # 5 = cv constant in cross validation

            for neighbors_num in range(5, top_border_of_neigh_num, 3):
                knn = KNC(metric='manhattan', n_neighbors=neighbors_num)
                X_scaled = scaler.transform(X_train)

                #to prevent the case when class values are not integers
                y_train = self._prevent_float_labeling(y_train, column_name)

                mean_acc_score = cross_val_score(knn, X_scaled, y_train, cv=5).mean()

                if mean_acc_score > max_acc + epsilon:
                    max_acc = mean_acc_score
                    best_neigh_num = neighbors_num

            if self.verbose == 1:
                print(column_name, max_acc, best_neigh_num)

            self._classifiers[column_name] = KNC(metric='manhattan',
                                                 n_neighbors=best_neigh_num).fit(scaler.transform(X_train),
                                                                                 y_train)

        for column_name in self._devided_features['regr']:
            current_X_columns = copy(list(X.columns.values))
            current_X_columns.remove(column_name)

            current_X, X_train, y_train, _ = self._get_X_and_y_by_column_name_with_imputs(X,
                                                                                               current_X_columns,
                                                                                               column_name)

            scaler = StandardScaler().fit(current_X)
            max_r2 = -1
            best_neigh_num = 0
            best_weights = ''
            epsilon = 0.005

            # FIXME problems with number of neighbors
            #for neighbors_num in range(3, min(100, int(len(current_X.index.values)*0.67)), 3):
            for neighbors_num in range(5, min(50, int(len(X_train.index.values)*0.67)), 3):
                knn_uniform = KNR(n_neighbors=neighbors_num)
                knn_distance = KNR(weights='distance', n_neighbors=neighbors_num)

                X_scaled = scaler.transform(X_train)
                r2_un = cross_val_score(knn_uniform, X_scaled, y_train, cv=5, scoring='r2').mean()
                r2_dis = cross_val_score(knn_distance, X_scaled, y_train, cv=5, scoring='r2').mean()

                if max(r2_dis, r2_un) > max_r2 or neighbors_num == 5:
                    max_r2 = max(r2_dis, r2_un)
                    if r2_dis > r2_un:
                        best_weights = 'distance'
                    else:
                        best_weights = 'uniform'
                    best_neigh_num = neighbors_num

            if self.verbose == 1:
                print(column_name, max_r2, best_neigh_num, best_weights)

            self._regressors[column_name] = KNR(weights=best_weights,
                                                n_neighbors=best_neigh_num).fit(scaler.transform(X_train),
                                                                                y_train)

    def _lr_fit(self, X, y=None):

        for column_name in self._devided_features['class']:
            current_X_columns = list(X.columns)
            current_X_columns.remove(column_name)

            current_X, X_train, y_train, _ = self._get_X_and_y_by_column_name_with_imputs(X,
                                                                                          current_X_columns,
                                                                                          column_name)

            scaler = StandardScaler().fit(current_X)
            self._label_encoders[column_name] = LabelEncoder().fit(y_train)
            y_train = self._label_encoders[column_name].transform(y_train)
            scaled_X = scaler.fit_transform(X_train)

            max_acc = 0
            best_c = 0
            epsilon = 0.0005

            #FIXME maybe this magic numbers should be tunable
            for c in [0.1 + 0.1*i for i in range(15)]:
                clf = LogisticRegression(random_state=510, C=c)

                acc_score = cross_val_score(clf, scaled_X, y_train, cv=5)
                mean_acc_score = acc_score.mean()

                if mean_acc_score > max_acc + epsilon:
                    max_acc = mean_acc_score
                    best_c = c

            if self.verbose > 0:
                print(max_acc, best_c)

            self._classifiers[column_name] = LogisticRegression(random_state=510,
                                                                C=c).fit(scaler.transform(X_train), y_train)

        for column_name in self._devided_features['regr']:
            current_X_columns = list(X.columns)
            current_X_columns.remove(column_name)

            current_X, X_train, y_train, _ = self._get_X_and_y_by_column_name_with_imputs(X,
                                                                                          current_X_columns,
                                                                                          column_name)

            scaler = StandardScaler().fit(current_X)

            max_r2 = -1
            best_alpha = 0
            epsilon = 0.0005

            #FIXME maybe this magic numbers should be tunable
            #for alpha in [10e-20*i for i in range()]:
            for alpha in [1e-20*(10 ** i) for i in range(21)]:
                rgr = Ridge(random_state=510, alpha=alpha, normalize=True)

                #X_scaled = StandardScaler().fit_transform(X)
                r2_score = cross_val_score(rgr, X_train, y_train, cv=5, scoring='r2')
                mean_r2_score = r2_score.mean()

                if mean_r2_score > max_r2 + epsilon:
                    max_r2 = mean_r2_score
                    best_alpha = alpha

            if self.verbose > 0:
                print(max_r2, best_alpha)

            self._regressors[column_name] = Ridge(random_state=510,
                                                  alpha=alpha, normalize=True).fit(scaler.transform(X_train), y_train)

    def _rf_fit(self, X, y=None):

        for column_name in self._devided_features['class']:
            current_X_columns = copy(list(X.columns.values))
            current_X_columns.remove(column_name)

            current_X, X_train, y_train, _ = self._get_X_and_y_by_column_name_with_imputs(X,
                                                                                          current_X_columns,
                                                                                          column_name)

            scaler = StandardScaler().fit(current_X)
            max_acc = 0
            best_param = 0
            epsilon = 0.0005

            for forest_num in range(5, 50, 3):
                clf = RandomForestClassifier(n_estimators=forest_num, random_state=510)
                X_scaled = scaler.transform(X_train)

                #to prevent the case when class values are not integers
                y_train = self._prevent_float_labeling(y_train, column_name)

                mean_acc_score = cross_val_score(clf, X_scaled, y_train, cv=5).mean()

                if mean_acc_score > max_acc + epsilon:
                    max_acc = mean_acc_score
                    best_param = forest_num

            if self.verbose == 1:
                print(column_name, max_acc, best_param)

            self._classifiers[column_name] = RandomForestClassifier(n_estimators=best_param,
                                                                    random_state=510).fit(scaler.transform(X_train),
                                                                                          y_train)

        for column_name in self._devided_features['regr']:
            current_X_columns = copy(list(X.columns.values))
            current_X_columns.remove(column_name)

            current_X, X_train, y_train, _ = self._get_X_and_y_by_column_name_with_imputs(X,
                                                                                          current_X_columns,
                                                                                          column_name)

            scaler = StandardScaler().fit(current_X)
            max_r2 = -1
            best_param = 0
            epsilon = 0.0005

            for forest_num in range(50, 53, 3):
                rgr = RandomForestRegressor(n_estimators=forest_num, random_state=510)

                mean_r2 = cross_val_score(rgr, scaler.transform(X_train), y_train, cv=5, scoring='r2').mean()

                if mean_r2 > max_r2 + epsilon or forest_num == 50:
                    best_param = forest_num
                    max_r2 = mean_r2

            if self.verbose == 1:
                print(column_name, best_param, max_r2)

            self._regressors[column_name] = RandomForestRegressor(n_estimators=best_param,
                                                                  random_state=510).fit(scaler.transform(X_train),
                                                                                        y_train)

    def _xgb_fit(self, X, y=None):
        for column_name in self._devided_features['class']:
            current_X_columns = list(X.columns)
            current_X_columns.remove(column_name)

            current_X, X_train, y_train, _ = self._get_X_and_y_by_column_name_with_imputs(X,
                                                                                          current_X_columns,
                                                                                          column_name)

            scaler = StandardScaler().fit(current_X)
            self._label_encoders[column_name] = LabelEncoder().fit(X[column_name].dropna())
            dtrain = self._get_dmatrix(scaler.transform(X_train), self._label_encoders[column_name].transform(y_train))

            param = {'silent': 1, 'nthread': 4}
            metric = ''
            #how much classes do we classify?
            num_class = y_train.value_counts().shape[0]
            if num_class == 2:
                param['objective'] = 'binary:logistic'
                metric = 'error'
            else:
                metric = 'merror'
                param['objective'] = 'multi:softmax'
                param['num_class'] = num_class

            #tune the best parameters
            best_param = {'1-error': 0}
            epsilon = 0.001

            converged = False
            #these magic numbers used below probably should be changed
            for eta in [0.4, 0.7, 1.0]:

                if converged is True:
                    break

                for max_depth in range(2, 9, 3):

                    if converged is True:
                        break

                    for num_round in range(10, 20, 3):

                        param['bst:max_depth'] = max_depth
                        param['bst:eta'] = eta
                        errors_df = xgb.cv(param, dtrain, num_round, nfold=5, metrics={metric})

                        test_mean_error = errors_df.iloc[-1][0]

                        if test_mean_error > best_param['1-error'] + epsilon or test_mean_error < epsilon:
                            best_param['1-error'] = test_mean_error
                            best_param['max_depth'] = max_depth
                            best_param['eta'] = eta
                            best_param['num_round'] = num_round

                            if test_mean_error == 0.0:
                                converged = True
                                break

            if self.verbose == 1:
                print(best_param)

            param['bst:max_depth'] = best_param['max_depth']
            param['bst:eta'] = best_param['eta']
            self._classifiers[column_name] = xgb.train(param, dtrain, best_param['num_round'])

        for column_name in self._devided_features['regr']:
            current_X_columns = list(X.columns)
            current_X_columns.remove(column_name)

            current_X, X_train, y_train, _ = self._get_X_and_y_by_column_name_with_imputs(X,
                                                                                          current_X_columns,
                                                                                          column_name)

            scaler = StandardScaler().fit(current_X)
            self._label_scalers[column_name] = MinMaxScaler().fit(y_train)
            dtrain = self._get_dmatrix(scaler.transform(X_train), self._label_scalers[column_name].transform(y_train))

            param = {'silent': 1, 'nthread': 4}
            param['objective'] = 'reg:logistic'

            #tune the best parameters
            best_param = {'error': 0}
            error_initialized = False
            #these magic numbers used below probably should be changed
            #for eta in [0.4 + i*0.2 for i in range(4)]:
            #    for max_depth in range(2, 11, 2):
            for num_round in range(10, 20):

                param['bst:max_depth'] = 2
                param['bst:eta'] = 1
                errors_df = xgb.cv(param, dtrain, num_round, nfold=5, metrics={'rmse'})

                test_mean_error = errors_df.iloc[-1][0]
                if error_initialized is False:
                    error_initialized = True
                    best_param['error'] = test_mean_error
                    best_param['num_round'] = num_round

                elif test_mean_error < best_param['error']:
                    best_param['error'] = test_mean_error
                    best_param['num_round'] = num_round

            if self.verbose == 1:
                print(best_param)

            param['bst:max_depth'] = 2
            param['bst:eta'] = 1
            self._regressors[column_name] = xgb.train(param, dtrain, best_param['num_round'])

    def _kmeans_fit(self, X, y=None, **kwargs):

        data = X.copy()
        if y is not None:
            data[y.name] = y
#        else:
#            raise ValueError('y should be given to apply \'kmeans\' method')

        self._classifiers = KMeans(**kwargs)
        self._classifiers.fit(Imputer().fit_transform(data))

    def _class_knn_svm_xgboost_fit(self, X, y=None):
        assert y is not None, 'y must be given for class knn strategy'

        classes_dataframes = self._get_dataframes_per_class(X, y)

        classifiers_for_class_value = {}
        regressors_for_class_value = {}
        if self.strategy == 'class_xgboost': #not sure of implementation
            encoders_for_class_value = {}
            scalers_for_class_value = {}

        for key in classes_dataframes.keys():

            if self.strategy == 'class_knn':
                self._knn_fit(classes_dataframes[key])
            elif self.strategy == 'class_svm':
                self._svm_fit(classes_dataframes[key])
            elif self.strategy == 'class_xgboost':
                self._xgb_fit(classes_dataframes[key])
                encoders_for_class_value[key] = copy(self._label_encoders)
                scalers_for_class_value[key] = copy(self._label_scalers)

            classifiers_for_class_value[key] = copy(self._classifiers)
            regressors_for_class_value[key] = copy(self._regressors)

        self._classifiers = classifiers_for_class_value
        self._regressors = regressors_for_class_value
        if self.strategy == 'class_xgboost':
            self._label_encoders = encoders_for_class_value
            self._label_scalers = scalers_for_class_value

    def _class_kmeans_fit(self, X, y=None, **kwargs):
        assert y is not None, 'y must be given for class knn strategy'

        classes_dataframes = self._get_dataframes_per_class(X, y)

        classifiers_for_class_value = {}

        for class_value in classes_dataframes.keys():
            class_value_df = classes_dataframes[class_value]

            self._kmeans_fit(class_value_df,
                             pd.Series([class_value] * class_value_df.shape[0],
                                       index=class_value_df.index),
                             **kwargs)

            classifiers_for_class_value[class_value] = copy(self._classifiers)

        self._classifiers = classifiers_for_class_value

    """
    Transform methods
    """

    def _class_transform(self, X_new, y=None):
        assert y is not None, 'y must be given for class mean strategy'

        nan_values = X_new.isnull()
        for column_name in X_new.columns.values:

            indices = list(nan_values[nan_values[column_name] == True].index.values)
            for i in indices:
                class_value = y.loc[i]
                if class_value in self._mean_values_mask.keys():
                    X_new.set_value(i, column_name, self._mean_values_mask[class_value][column_name])

    def _kmeans_transform(self, X_new, y=None):

        X_temp = X_new.copy()
        if y is not None:
            X_temp[y.name] = y


        estr = self._classifiers
        clusters_pred = estr.predict(Imputer().fit_transform(X_temp))
        cluster_centers = estr.cluster_centers_

        absolute_index = 0

        for index in X_new.index:

            row = X_new.loc[index].isnull()

            if row.any():
                for column_index in range(row.size):
                    if row.iloc[column_index] == True:

                        # FIXME MAGIC! set_value doesn't work on dataset and works on another one
                        X_new.set_value(index,
                                        X_new.columns[column_index],
                                        cluster_centers[clusters_pred[absolute_index], column_index])

            absolute_index += 1

        return X_new

    def _knn_svm_lr_transform(self, X, X_new, y=None):
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

    def _xgboost_transform(self, X, X_new, y=None):
        for column_name in self._devided_features['class']:

            current_X_columns = copy(list(X.columns.values))
            current_X_columns.remove(column_name)

            current_X, _, _, X_test = self._get_X_and_y_by_column_name_with_imputs(X,
                                                                                   current_X_columns,
                                                                                   column_name)

            if X_test.empty is False:
                scaler = StandardScaler().fit(current_X)
                y_pred = self._classifiers[column_name].predict(xgb.DMatrix(scaler.transform(X_test)))
                y_pred = self._label_encoders[column_name].inverse_transform(y_pred.astype(int))

                self._set_pred_values_to_df(list(X_test.index.values), X_new, y_pred, column_name)

        for column_name in self._devided_features['regr']:

            current_X_columns = copy(list(X.columns.values))
            current_X_columns.remove(column_name)

            current_X, _, _, X_test = self._get_X_and_y_by_column_name_with_imputs(X,
                                                                                   current_X_columns,
                                                                                   column_name)

            if X_test.empty is False:
                scaler = StandardScaler().fit(current_X)
                y_pred = self._regressors[column_name].predict(xgb.DMatrix(scaler.transform(X_test)))
                y_pred = self._label_scalers[column_name].inverse_transform(y_pred)

                self._set_pred_values_to_df(list(X_test.index.values), X_new, y_pred, column_name)

    def _class_knn_svm_xgboost_kmeans_transform(self, X, X_new, y=None):

        classes_dataframes = self._get_dataframes_per_class(X, y)
        classifiers_for_class_value = copy(self._classifiers)
        if self.strategy == 'class_svm' or self.strategy == 'class_knn' or self.strategy == 'class_xgboost':
            regressors_for_class_value = copy(self._regressors)
        if self.strategy == 'class_xgboost':
            encoders_for_class_value = copy(self._label_encoders)
            scalers_for_class_value = copy(self._label_scalers)
        X_new = pd.DataFrame(columns=X.columns.values)

        for (class_value, value_df) in classes_dataframes.items():

            self._classifiers = classifiers_for_class_value[class_value]
            if self.strategy == 'class_svm' or self.strategy == 'class_knn' or self.strategy == 'class_xgboost':
                self._regressors = regressors_for_class_value[class_value]
            if self.strategy == 'class_xgboost':
                self._label_encoders = encoders_for_class_value[class_value]
                self._label_scalers = scalers_for_class_value[class_value]

            new_df = value_df.copy()
            if self.strategy == 'class_xgboost':
                self._xgboost_transform(value_df, new_df)
            elif self.strategy == 'class_svm' or self.strategy == 'class_knn':
                self._knn_svm_lr_transform(value_df, new_df)
            else:
                self._kmeans_transform(new_df, pd.Series([class_value] * new_df.shape[0],
                                                         index=new_df.index))
            X_new = pd.concat([X_new, new_df])

        return X_new

    """
    Supporting methods
    """

    def _get_dmatrix(self, X, y):
        #xgb_X = X.as_matrix()
        xgb_y = np.asarray(y)

        return xgb.DMatrix(X, label=xgb_y)

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
            y_test = self._classifiers[column_name].predict(scaler.transform(X_test))
            if column_name in self._label_encoders.keys():
                y_test = self._label_encoders[column_name].inverse_transform(y_test)
            self._set_pred_values_to_df(list(X_test.index.values), X_new, y_test, column_name)

    def _use_rgr_to_fill_na(self, current_X, X_test, column_name, X_new):
        if X_test.empty is False:
            scaler = StandardScaler().fit(current_X)
            y_test = self._regressors[column_name].predict(scaler.transform(X_test))
            self._set_pred_values_to_df(list(X_test.index.values), X_new, y_test, column_name)

    def _get_dict_of_values_to_set(self, X):
        dict_of_values = {}
        index_in_mask = 0

        for column in X.columns.values:
            dict_of_values[column] = self._mask[index_in_mask]
            index_in_mask += 1

        return dict_of_values

    def _set_pred_values_to_df(self, indices, X_new, y_pred, column_name):
        counter = 0
        for index in indices:
            X_new.set_value(index, column_name, y_pred[counter])
            counter += 1

    def _devide_features_to_classifiable_and_regressiable(self, df, max_number_of_values_per_class):
        columns = list(df.columns.values)
        devided_features = {'class': [], 'regr': []}
        for column in columns:
            if len(df[column].value_counts().index) < max_number_of_values_per_class:
                devided_features['class'].append(column)
            else:
                devided_features['regr'].append(column)
        return devided_features

    def _prevent_float_labeling(self, y, column_name):
        try:
            if type(y.value_counts().index.values[0]) is np.float64:
                self._label_encoders[column_name] = LabelEncoder().fit(y)
                y = self._label_encoders[column_name].transform(y)
        except AttributeError:
            #y could be not pd.Series, so...
            pass

        return y