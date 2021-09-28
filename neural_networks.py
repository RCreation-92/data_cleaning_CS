
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import metrics
from sklearn import preprocessing
from collections import Counter
from imblearn.over_sampling import SMOTE
import statsmodels.api as sm

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import minmax_scale # single column normalization
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


logging.basicConfig(filename='report.log', filemode='w', format='%(message)s', level=logging.INFO)


df = pd.read_csv(r'dataset_train_1.csv')
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     print(df)
test_df = pd.read_csv(r'dataset_test_1.csv')


# Find the missing values in dataset
print('data_types', df.dtypes)
print('dataframe_info', df.info())
print('df_null', df.isnull().sum())
print('df.corr', df.corr())


# Aggressively drop the rows with NA
aggressive_drop = df.dropna()
print('aggressive_drop', aggressive_drop)


class ChainedAssignent:
    def __init__(self, chained=None):
        acceptable = [None, 'warn', 'raise']
        assert chained in acceptable, "chained must be in " + str(acceptable)
        self.swcw = chained

    def __enter__(self):
        self.saved_swcw = pd.options.mode.chained_assignment
        pd.options.mode.chained_assignment = self.swcw
        return self

    def __exit__(self, *args):
        pd.options.mode.chained_assignment = self.saved_swcw


def cleaning_data(any_dataframe):

    print('any_df', any_dataframe)
    df = any_dataframe
    df_imputed = df.select_dtypes(include=np.number).fillna(df.mean())
    print('df_imputed', df_imputed)
    # df1 = any_dataframe.select_dtypes(include=np.number)
    #
    # df1_imputed = df1.fillna(df1.mean())
    # print('df1_imputed', df1_imputed)

    merged = pd.merge(df, df_imputed, how='left', on='ID')
    print('merged', merged)

    merged_2 = merged[merged.columns.drop(list(merged.filter(regex='_x')))]
    # print(merged_2)

    df_imputed_dropped = merged_2.dropna()
    print('imputed+dropped df', df_imputed_dropped)
    # print(df_imputed_dropped.isnull().sum())
    # print(df.isnull().sum())

    df_imputed_dropped_object = df.select_dtypes(include=object)
    print('object', df.select_dtypes(include=object))
    for column in df_imputed_dropped_object:
        with ChainedAssignent():
            df_imputed_dropped.loc[:, column] = pd.Categorical(df_imputed_dropped[column])
            df_imputed_dropped.loc[:, column + str('_Code')] = df_imputed_dropped[column].cat.codes

    print('df_imputed_dropped[Var1_Code]', df_imputed_dropped['Var1_Code'])
    df_imputed_dropped_2 = df_imputed_dropped.drop(df_imputed_dropped_object, axis=1)
    print('df_imputed_dropped_2', df_imputed_dropped_2)
    print(df_imputed_dropped_2.isnull().sum())

    df_imputed_dropped_3 = df_imputed_dropped_2.drop('ID', axis=1)

    min_max_scaler = preprocessing.MinMaxScaler()
    data_minmax = min_max_scaler.fit_transform(df_imputed_dropped_3)

    new_dataframe = pd.DataFrame(data_minmax)

    new_dataframe.columns = df_imputed_dropped_3.columns.values
    # print('scaled', new_dataframe)
    return new_dataframe


df_cleaned = cleaning_data(df)
# print(df_cleaned.isnull().sum())
# print('df_cleaned', df_cleaned)

test_df_cleaned = cleaning_data(test_df)


# print('test_df_cleaned', test_df_cleaned)


def logistic_regression(train, test):

    train_feature_list = train.drop(['Target_Code'], axis=1)
    test_feature_list = test.drop(['Target_Code'], axis=1)
    train_target = train['Target_Code']
    test_target = test['Target_Code']

    logis_regression = LogisticRegression()
    fitted_model = logis_regression.fit(train_feature_list, train_target)
    print("Training set score: {:.3f}".format(fitted_model.score(train_feature_list, train_target)))
    print("Test set score: {:.3f}".format(fitted_model.score(test_feature_list, test_target)))

    logit_model = sm.Logit(train_target, train_feature_list)
    result = logit_model.fit()
    print(result.summary())

    prediction = logis_regression.predict(test_feature_list)

    confusion_matrix = metrics.confusion_matrix(test_target, prediction)
    # print('coef+intercept', fitted_model.coef_, fitted_model.intercept_)

    # print('confusion_matrix', confusion_matrix)

    tn, fp, fn, tp = confusion_matrix.ravel()

    specificity = tn / (fp + tn)
    sensitivity = tp / (tp + fn)
    type1_error = fp / (fp + tn)
    type2_error = fn / (tp + fn)

    accuracy_logregression = (tn + tp) / (tn + fp + fn + tp)

    fpr, tpr, threshold = metrics.roc_curve(test_target, prediction)
    roc_auc = metrics.auc(fpr, tpr)

    AIC = result.aic
    print('AIC', AIC)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    # print('specificity_logreg', specificity)
    # print('sensitivity_logreg', sensitivity)
    # print('confusion_matrix_logreg', confusion_matrix)
    # print('accuracy_logreg', accuracy_logreg)

    return confusion_matrix, accuracy_logregression, specificity, sensitivity, type1_error, type2_error, AIC


def logistic_regression_smt(train, test):
    train_feature_list = train.drop(['Target_Code'], axis=1)
    test_feature_list = test.drop(['Target_Code'], axis=1)
    train_target = train['Target_Code']
    test_target = test['Target_Code']

    # Using SMOTE package

    counter = Counter(train_target)
    print('Before', counter)

    smt = SMOTE(random_state=100)
    train_feature_list_smt, train_target_smt = \
        smt.fit_resample(train_feature_list, train_target)
    counter = Counter(train_target_smt)
    print('After', counter)

    logis_regression = LogisticRegression()
    fitted_model = logis_regression.fit(train_feature_list_smt, train_target_smt)
    print("Training set score: {:.3f}".format(fitted_model.score(train_feature_list_smt, train_target_smt)))
    print("Test set score: {:.3f}".format(fitted_model.score(test_feature_list, test_target)))

    logit_model = sm.Logit(train_target_smt, train_feature_list_smt)
    result = logit_model.fit()
    print(result.summary())
    prediction = logis_regression.predict(test_feature_list)

    confusion_matrix = metrics.confusion_matrix(test_target, prediction)

    AIC = result.aic
    print('AIC', AIC)

    tn, fp, fn, tp = confusion_matrix.ravel()

    specificity = tn / (fp + tn)
    sensitivity = tp / (tp + fn)
    accuracy_SMOTE_ = (tn + tp) / (tn + fp + fn + tp)
    type1_error = fp / (fp + tn)
    type2_error = fn / (tp + fn)

    fpr, tpr, threshold = metrics.roc_curve(test_target, prediction)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    # print('specificity_SMOTE', specificity)
    # print('sensitivity_SMOTE', sensitivity)
    # print('confusion_matrix_logreg', confusion_matrix)
    # print('accuracy_SMOTE', accuracy_SMOTE)

    return confusion_matrix, accuracy_SMOTE_, specificity, sensitivity, type1_error, type2_error, AIC


def optimized_data_for_logreg(any_dataframe):
    df1 = any_dataframe.select_dtypes(include=np.number)

    df1_imputed = df1.fillna(df1.mean())
    # print(df1_imputed)

    merged = pd.merge(df, df1_imputed, how='left', on='ID')
    # print(merged)

    merged_2 = merged[merged.columns.drop(list(merged.filter(regex='_x')))]
    # print(merged_2)

    df_imputed_dropped = merged_2.dropna()
    # print('imputed+dropped df', df_imputed_dropped)
    # print(df_imputed_dropped.isnull().sum())

    df_imputed_dropped_object = df.select_dtypes(include=object)
    for column in df_imputed_dropped_object:
        with ChainedAssignent():
            df_imputed_dropped.loc[:, column] = pd.Categorical(df_imputed_dropped[column])
            df_imputed_dropped.loc[:, column + str('_Code')] = df_imputed_dropped[column].cat.codes

    df_imputed_dropped_2 = df_imputed_dropped.drop(df_imputed_dropped_object, axis=1)
    # print('df_imputed_dropped_2', df_imputed_dropped_2)
    # print(df_imputed_dropped_2.isnull().sum())

    df_imputed_dropped_3 = df_imputed_dropped_2.drop('ID', axis=1)
    print('df_imputed_dropped_3', df_imputed_dropped_3)
    min_max_scaler = preprocessing.MinMaxScaler()
    data_minmax = min_max_scaler.fit_transform(df_imputed_dropped_3)

    new_dataframe = pd.DataFrame(data_minmax)

    new_dataframe.columns = df_imputed_dropped_3.columns.values

    new_dataframe_2 = new_dataframe[["Var9_y", "Var12_y", "Var15_y", "Var18_y",
                                     "Var20_y", "Var5_Code", "Var6_Code", "Target_Code"]]

    return new_dataframe_2


def optimized_data_for_SMOTE(any_dataframe):
    df1 = any_dataframe.select_dtypes(include=np.number)

    df1_imputed = df1.fillna(df1.mean())
    # print(df1_imputed)

    merged = pd.merge(df, df1_imputed, how='left', on='ID')
    # print(merged)

    merged_2 = merged[merged.columns.drop(list(merged.filter(regex='_x')))]
    # print(merged_2)

    df_imputed_dropped = merged_2.dropna()
    # print('imputed+dropped df', df_imputed_dropped)
    # print(df_imputed_dropped.isnull().sum())

    df_imputed_dropped_object = df.select_dtypes(include=object)
    for column in df_imputed_dropped_object:
        with ChainedAssignent():
            df_imputed_dropped.loc[:, column] = pd.Categorical(df_imputed_dropped[column])
            df_imputed_dropped.loc[:, column + str('_Code')] = df_imputed_dropped[column].cat.codes

    df_imputed_dropped_2 = df_imputed_dropped.drop(df_imputed_dropped_object, axis=1)
    # print('df_imputed_dropped_2', df_imputed_dropped_2)
    # print(df_imputed_dropped_2.isnull().sum())

    df_imputed_dropped_3 = df_imputed_dropped_2.drop('ID', axis=1)
    print('df_imputed_dropped_3', df_imputed_dropped_3)
    min_max_scaler = preprocessing.MinMaxScaler()
    data_minmax = min_max_scaler.fit_transform(df_imputed_dropped_3)

    new_dataframe = pd.DataFrame(data_minmax)

    new_dataframe.columns = df_imputed_dropped_3.columns.values

    new_dataframe_3 = new_dataframe[["Var7_y", "Var9_y", "Var11_y", "Var12_y", "Var15_y", "Var18_y",
                                     "Var20_y", "Var4_Code", "Var5_Code", "Var6_Code", "Target_Code"]]

    # print('scaled', new_dataframe)
    return new_dataframe_3


df_cleaned_logreg = optimized_data_for_logreg(df)
# print(df_cleaned.isnull().sum())
# print('df_cleaned_2', df_cleaned_2)


test_df_cleaned_logreg = optimized_data_for_logreg(test_df)
# print('test_df_cleaned', test_df_cleaned)


df_cleaned_SMOTE = optimized_data_for_SMOTE(df)
# print(df_cleaned.isnull().sum())
# print('df_cleaned_2', df_cleaned_2)

test_df_cleaned_SMOTE = optimized_data_for_SMOTE(test_df)
# print('test_df_cleaned', test_df_cleaned)


# confusion_matrix_logreg, accuracy_logreg, specificity_logreg, sensitivity_logreg, \
# type1_logreg, type2_logreg, AIC_logreg = logistic_regression(df_cleaned, test_df_cleaned)
# # confusion_matrix, model_accuracy = logistic_regression(df_cleaned, test_df_cleaned)
# print('results for logistic regression')
# print('confusion_matrix_logreg', confusion_matrix_logreg)
# print('accuracy_logreg', accuracy_logreg)
# print('specificity_logreg', specificity_logreg)
# print('sensitivity_logreg', sensitivity_logreg)
# print('type1_logreg', type1_logreg)
# print('type2_logreg', type2_logreg)
#
# confusion_matrix_SMOTE, accuracy_SMOTE, specificity_SMOTE, sensitivity_SMOTE, type1_SMOTE, type2_SMOTE, AIC_SMOTE \
#     = logistic_regression_smt(df_cleaned, test_df_cleaned)
# print('results for SMOTE data + logistic regression')
# print('confusion_matrix_SMOTE', confusion_matrix_SMOTE)
# print('accuracy_SMOTE', accuracy_SMOTE)
# print('specificity_SMOTE', specificity_SMOTE)
# print('sensitivity_SMOTE', sensitivity_SMOTE)
# print('type1_SMOTE', type1_SMOTE)
# print('type2_SMOTE', type2_SMOTE)

# opt_confusion_matrix_logreg, opt_accuracy_logreg, opt_specificity_logreg, opt_sensitivity_logreg, \
# opt_type1_logreg, opt_type2_logreg, opt_AIC_logreg = logistic_regression(df_cleaned_logreg, test_df_cleaned_logreg)
# confusion_matrix, model_accuracy = logistic_regression(df_cleaned, test_df_cleaned)
# print('results for optimized dataset + logistic regression')
# print('opt_confusion_matrix_logreg', opt_confusion_matrix_logreg)
# print('opt_accuracy_logreg', opt_accuracy_logreg)
# print('opt_specificity_logreg', opt_specificity_logreg)
# print('opt_sensitivity_logreg', opt_sensitivity_logreg)
# print('opt_type1_logreg', opt_type1_logreg)
# print('opt_type2_logreg', opt_type2_logreg)
#
# opt_confusion_matrix_SMOTE, opt_accuracy_SMOTE, opt_specificity_SMOTE, opt_sensitivity_SMOTE, \
# opt_type1_SMOTE, opt_type2_SMOTE, opt_AIC_SMOTE = logistic_regression_smt(df_cleaned_SMOTE, test_df_cleaned_SMOTE)
# print('results for optimized SMOTE data + logistic regression')
# print('opt_confusion_matrix_SMOTE', opt_confusion_matrix_SMOTE)
# print('opt_accuracy_SMOTE', opt_accuracy_SMOTE)
# print('opt_specificity_SMOTE', opt_specificity_SMOTE)
# print('opt_sensitivity_SMOTE', opt_sensitivity_SMOTE)
# print('opt_type1_SMOTE', opt_type1_SMOTE)
# print('opt_type2_SMOTE', opt_type2_SMOTE)


# print('optimized_results\n', logistic_regression(df_cleaned_2, test_df_cleaned_2))
# print('optimized_results_SMOTE\n', logistic_regression_smt(df_cleaned_SMOTE, test_df_cleaned_SMOTE))
# print('optimized_results_ADA\n', logistic_regression_adasyn(df_cleaned_ADASYN, test_df_cleaned_ADASYN))


def RandomForest(train, test):
    train_feature_list = train.drop(['Target_Code'], axis=1)
    test_feature_list = test.drop(['Target_Code'], axis=1)
    train_target = train['Target_Code']
    test_target = test['Target_Code']

    counter = Counter(train_target)
    print('Before', counter)

    smt = SMOTE(random_state=100)
    train_feature_list_smt, train_target_smt = \
        smt.fit_resample(train_feature_list, train_target)
    counter = Counter(train_target_smt)
    print('After', counter)

    rf = RandomForestClassifier(max_depth=3, random_state=100)
    fitted_model = rf.fit(train_feature_list_smt, train_target_smt)
    # print("Training set score: {:.3f}".format(fitted_model.score(train_feature_list, train_target)))
    # print("Test set score: {:.3f}".format(fitted_model.score(test_feature_list, test_target)))

    logit_model = sm.Logit(train_target_smt, train_feature_list_smt)
    result = logit_model.fit()
    print(result.summary())

    prediction = rf.predict(test_feature_list)

    confusion_matrix = metrics.confusion_matrix(test_target, prediction)
    # print('coef+intercept', fitted_model.coef_, fitted_model.intercept_)

    # print('confusion_matrix', confusion_matrix)

    tn, fp, fn, tp = confusion_matrix.ravel()

    specificity = tn / (fp + tn)
    sensitivity = tp / (tp + fn)
    type1_error = fp / (fp + tn)
    type2_error = fn / (tp + fn)

    accuracy_logregression = (tn + tp) / (tn + fp + fn + tp)

    fpr, tpr, threshold = metrics.roc_curve(test_target, prediction)
    roc_auc = metrics.auc(fpr, tpr)

    AIC = result.aic
    print('AIC', AIC)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    # print('specificity_logreg', specificity)
    # print('sensitivity_logreg', sensitivity)
    # print('confusion_matrix_logreg', confusion_matrix)
    # print('accuracy_logreg', accuracy_logreg)

    return confusion_matrix, accuracy_logregression, specificity, sensitivity, type1_error, type2_error, AIC

#
# confusion_matrix_RF, accuracy_RF, specificity_RF, sensitivity_RF, type1_RF, \
# type2_RF, AIC_RF = RandomForest(df_cleaned, test_df_cleaned)
# print('results for optimized RF data + logistic regression')
# print('confusion_matrix_RF', confusion_matrix_RF)
# print('accuracy_RF', accuracy_RF)
# print('specificity_RF', specificity_RF)
# print('sensitivity_RF', sensitivity_RF)
# print('type1_RF', type1_RF)
# print('type2_RF', type2_RF)
# print('AIC_RF', AIC_RF)
#
# opt_confusion_matrix_RF, opt_accuracy_RF, opt_specificity_RF, opt_sensitivity_RF, opt_type1_RF, \
# opt_type2_RF, opt_AIC_RF = RandomForest(df_cleaned_SMOTE, test_df_cleaned_SMOTE)
# print('results for optimized RF data + logistic regression')
# print('opt_confusion_matrix_RF', opt_confusion_matrix_RF)
# print('opt_accuracy_RF', opt_accuracy_RF)
# print('opt_specificity_RF', opt_specificity_RF)
# print('opt_sensitivity_RF', opt_sensitivity_RF)
# print('opt_type1_RF', opt_type1_RF)
# print('opt_type2_RF', opt_type2_RF)
# print('opt_AIC_RF', opt_AIC_RF)


def Adaboost(train, test):
    train_feature_list = train.drop(['Target_Code'], axis=1)
    test_feature_list = test.drop(['Target_Code'], axis=1)
    train_target = train['Target_Code']
    test_target = test['Target_Code']

    counter = Counter(train_target)
    print('Before', counter)

    smt = SMOTE(random_state=100)
    train_feature_list_smt, train_target_smt = \
        smt.fit_resample(train_feature_list, train_target)
    counter = Counter(train_target_smt)
    print('After', counter)

    adaboost = AdaBoostClassifier(n_estimators=100, random_state=100)
    fitted_model = adaboost.fit(train_feature_list_smt, train_target_smt)
    # print("Training set score: {:.3f}".format(fitted_model.score(train_feature_list, train_target)))
    # print("Test set score: {:.3f}".format(fitted_model.score(test_feature_list, test_target)))

    logit_model = sm.Logit(train_target_smt, train_feature_list_smt)
    result = logit_model.fit()
    print(result.summary())

    prediction = adaboost.predict(test_feature_list)

    confusion_matrix = metrics.confusion_matrix(test_target, prediction)
    # print('coef+intercept', fitted_model.coef_, fitted_model.intercept_)

    # print('confusion_matrix', confusion_matrix)

    tn, fp, fn, tp = confusion_matrix.ravel()

    specificity = tn / (fp + tn)
    sensitivity = tp / (tp + fn)
    type1_error = fp / (fp + tn)
    type2_error = fn / (tp + fn)

    accuracy_logregression = (tn + tp) / (tn + fp + fn + tp)

    fpr, tpr, threshold = metrics.roc_curve(test_target, prediction)
    roc_auc = metrics.auc(fpr, tpr)

    AIC = result.aic
    print('AIC', AIC)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    # print('specificity_logreg', specificity)
    # print('sensitivity_logreg', sensitivity)
    # print('confusion_matrix_logreg', confusion_matrix)
    # print('accuracy_logreg', accuracy_logreg)

    return confusion_matrix, accuracy_logregression, specificity, sensitivity, type1_error, type2_error, AIC


# confusion_matrix_ADA, accuracy_ADA, specificity_ADA, sensitivity_ADA, type1_ADA, \
# type2_ADA, AIC_ADA = Adaboost(df_cleaned, test_df_cleaned)
# print('results for optimized ADA data + logistic regression')
# print('confusion_matrix_ADA', confusion_matrix_ADA)
# print('accuracy_ADA', accuracy_ADA)
# print('specificity_ADA', specificity_ADA)
# print('sensitivity_ADA', sensitivity_ADA)
# print('type1_ADA', type1_ADA)
# print('type2_ADA', type2_ADA)
# print('AIC_ADA', AIC_ADA)
#
# opt_confusion_matrix_ADA, opt_accuracy_ADA, opt_specificity_ADA, opt_sensitivity_ADA, opt_type1_ADA, \
# opt_type2_ADA, opt_AIC_ADA = Adaboost(df_cleaned_SMOTE, test_df_cleaned_SMOTE)
# print('results for optimized ADA data + logistic regression')
# print('opt_confusion_matrix_ADA', opt_confusion_matrix_ADA)
# print('opt_accuracy_ADA', opt_accuracy_ADA)
# print('opt_specificity_ADA', opt_specificity_ADA)
# print('opt_sensitivity_ADA', opt_sensitivity_ADA)
# print('opt_type1_ADA', opt_type1_ADA)
# print('opt_type2_ADA', opt_type2_ADA)
# print('opt_AIC_ADA', opt_AIC_ADA)


########### Code below here is uses Neural Networks which is a similar technique to SMOTE for handling the minority class########
##

def neural_network_test(train_dataframe, test_dataframe):

    train_data = train_dataframe.values
    test_data = test_dataframe.values
    print('train', train_dataframe)
    print('test', test_dataframe)

    train_data_preprocess = []
    for i in range(train_data.shape[1]):
        data_temp = []
        if (i == train_data.shape[1]-1):  # first column 'class'
            # normalize the numeric data
            catBinarizer = LabelBinarizer()
            data_temp = catBinarizer.fit_transform(train_data[:, i])
        else:
            data_temp = minmax_scale(train_data[:, i].astype(float))
            data_temp = np.reshape(data_temp, (len(data_temp), 1))
        if len(train_data_preprocess) == 0:
            train_data_preprocess = data_temp
        else:
            train_data_preprocess = np.hstack([train_data_preprocess, data_temp])

    print("train_data_preprocess shape:", train_data_preprocess.shape)

    test_data_preprocess = []
    for i in range(test_data.shape[1]):
        data_temp = []
        if (i == test_data.shape[1]-1):  # first column 'class'
            # normalize the numeric data
            catBinarizer = LabelBinarizer()
            print('test_data_1', test_data[:, i])
            data_temp = catBinarizer.fit_transform(test_data[:, i])
            print('test_data_catbinarizer', catBinarizer.fit_transform(test_data[:, i]))
        else:
            data_temp = minmax_scale(test_data[:, i].astype(float))
            data_temp = np.reshape(data_temp, (len(data_temp), 1))
        if len(test_data_preprocess) == 0:
            test_data_preprocess = data_temp
        else:
            test_data_preprocess = np.hstack([test_data_preprocess, data_temp])

    print("test_data_preprocess", test_data_preprocess)

    x_train = train_data_preprocess[:,:-1]
    y_train = train_data_preprocess[:,-1:]
    x_test = test_data_preprocess[:,:-1]
    y_test = test_data_preprocess[:,-1:]

    print("Training data:", x_train.shape, y_train.shape)
    print("Test data:", x_test.shape, y_test.shape)

    model = Sequential()
    model.add(Dense(10, input_dim=x_train.shape[1], activation="relu"))
    model.add(Dense(4, activation="relu"))
    model.add(Dense(4, activation="gelu"))
    model.add(Dense(1, activation="relu"))
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.SGD(learning_rate=0.001))

    class_weights = {0: len(y_train) / np.sum(y_train == 0),
                     1: len(y_train) / np.sum(y_train == 1)}

    print("Class weights:", class_weights)
    hist = model.fit(x_train, y_train, epochs=100, class_weight=class_weights, verbose=0)

    y_predict_class = (model.predict(x_test) > 0.5).astype("int32")
    prediction_results = print(pd.DataFrame(confusion_matrix(y_test, y_predict_class), index=['true:0', 'true:1'], columns=['pred:0', 'pred:1']))

    return prediction_results

print('neural_networks', neural_network_test(df_cleaned, test_df_cleaned))
###########Code below here is uses ADASYN which is a similar technique to SMOTE for handling the minority class########

# def optimized_data_for_ADASYN(any_dataframe):
#
#     df1 = any_dataframe.select_dtypes(include=np.number)
#
#     df1_imputed = df1.fillna(df1.mean())
#     # print(df1_imputed)
#
#     merged = pd.merge(df, df1_imputed, how='left', on='ID')
#     # print(merged)
#
#     merged_2 = merged[merged.columns.drop(list(merged.filter(regex='_x')))]
#     # print(merged_2)
#
#     df_imputed_dropped = merged_2.dropna()
#     # print('imputed+dropped df', df_imputed_dropped)
#     # print(df_imputed_dropped.isnull().sum())
#
#     df_imputed_dropped_object = df.select_dtypes(include=object)
#     for column in df_imputed_dropped_object:
#         with ChainedAssignent():
#             df_imputed_dropped.loc[:, column] = pd.Categorical(df_imputed_dropped[column])
#             df_imputed_dropped.loc[:, column + str('_Code')] = df_imputed_dropped[column].cat.codes
#
#     df_imputed_dropped_2 = df_imputed_dropped.drop(df_imputed_dropped_object, axis=1)
#     # print('df_imputed_dropped_2', df_imputed_dropped_2)
#     # print(df_imputed_dropped_2.isnull().sum())
#
#     df_imputed_dropped_3 = df_imputed_dropped_2.drop('ID', axis=1)
#     print('df_imputed_dropped_3', df_imputed_dropped_3)
#     min_max_scaler = preprocessing.MinMaxScaler()
#     data_minmax = min_max_scaler.fit_transform(df_imputed_dropped_3)
#
#     new_dataframe = pd.DataFrame(data_minmax)
#
#     new_dataframe.columns = df_imputed_dropped_3.columns.values
#
#     new_dataframe_4 = new_dataframe[["Var7_y", "Var8_y", "Var9_y", "Var11_y", "Var12_y", "Var13_y",
#                                      "Var15_y", "Var18_y",
#                                     "Var20_y", "Var4_Code", "Var5_Code", "Var6_Code", "Target_Code"]]
#
#     # print('scaled', new_dataframe)
#     return new_dataframe_4
#
#


# def logistic_regression_adasyn(train, test):
#
#     train_feature_list = train.drop(['Target_Code'], axis=1)
#     test_feature_list = test.drop(['Target_Code'], axis=1)
#     train_target = train['Target_Code']
#     test_target = test['Target_Code']
#
#     # Using ADASYN from SMOTE
#
#     counter = Counter(train_target)
#     print('Before', counter)
#
#     ada = ADASYN(random_state=3)
#     train_feature_list_ada, train_target_ada = ada.fit_resample(train_feature_list, train_target)
#     counter = Counter(train_target_ada)
#     print('After', counter)
#
#     logis_regression = LogisticRegression()
#     logis_regression.fit(train_feature_list_ada, train_target_ada)
#     fitted_model = logis_regression.fit(train_feature_list_ada, train_target_ada)
#     print("Training set score: {:.3f}".format(fitted_model.score(train_feature_list_ada, train_target_ada)))
#     print("Test set score: {:.3f}".format(fitted_model.score(test_feature_list, test_target)))
#
#     logit_model = sm.Logit(train_target_ada, train_feature_list_ada)
#     result = logit_model.fit()
#     print(result.summary())
#
#     prediction = logis_regression.predict(test_feature_list)
#
#     confusion_matrix = metrics.confusion_matrix(test_target, prediction)
#
#     tn, fp, fn, tp = confusion_matrix.ravel()
#
#     specificity = tn/(fp+tn)
#     sensitivity = tp / (tp + fn)
#     type1_error = fp / (fp + tn)
#     type2_error = fn / (tp + fn)
#
#     accuracy_ADASYN_ = (tn + tp) / (tn + fp + fn + tp)
#
#     fpr, tpr, threshold = metrics.roc_curve(test_target, prediction)
#     roc_auc = metrics.auc(fpr, tpr)
#
#     AIC = result.aic
#     print('AIC', AIC)
#
#     plt.title('Receiver Operating Characteristic')
#     plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
#     plt.legend(loc='lower right')
#     plt.plot([0, 1], [0, 1], 'r--')
#     plt.xlim([0, 1])
#     plt.ylim([0, 1])
#     plt.ylabel('True Positive Rate')
#     plt.xlabel('False Positive Rate')
#     plt.show()
#
#     # print('specificity_ADASYN', specificity)
#     # print('sensitivity_ADASYN', sensitivity)
#     # print('confusion_matrix_logreg', confusion_matrix)
#     # print('accuracy_ADASYN', accuracy_ADASYN_)
#
#     return confusion_matrix, accuracy_ADASYN_, specificity, sensitivity, type1_error, type2_error, AIC

# df_cleaned_ADASYN = optimized_data_for_ADASYN(df)
# # print(df_cleaned.isnull().sum())
# # print('df_cleaned_2', df_cleaned_2)
#
# test_df_cleaned_ADASYN = optimized_data_for_ADASYN(test_df)
# # print('test_df_cleaned', test_df_cleaned)

# confusion_matrix_ADASYN, accuracy_ADASYN, specificity_ADASYN, sensitivity_ADASYN, type1_ADASYN, type2_ADASYN,\
#     AIC_ADASYN = logistic_regression_adasyn(df_cleaned, test_df_cleaned)
# print('results for ADASYN data + logistic regression')
# print('confusion_matrix_ADASYN', confusion_matrix_ADASYN)
# print('accuracy_ADASYN', accuracy_ADASYN)
# print('specificity_ADASYN', specificity_ADASYN)
# print('sensitivity_ADASYN', sensitivity_ADASYN)
# print('type1_ADASYN', type1_ADASYN)
# print('type2_ADASYN', type2_ADASYN)

# opt_confusion_matrix_ADASYN, opt_accuracy_ADASYN, opt_specificity_ADASYN, opt_sensitivity_ADASYN, opt_type1_ADASYN, \
#     opt_type2_ADASYN, opt_AIC_ADASYN = logistic_regression_adasyn(df_cleaned_ADASYN, test_df_cleaned_ADASYN)
# print('results for optimized ADASYN data + logistic regression')
# print('opt_confusion_matrix_ADASYN', opt_confusion_matrix_ADASYN)
# print('opt_accuracy_ADASYN', opt_accuracy_ADASYN)
# print('opt_specificity_ADASYN', opt_specificity_ADASYN)
# print('opt_sensitivity_ADASYN', opt_sensitivity_ADASYN)
# print('opt_type1_ADASYN', opt_type1_ADASYN)
# print('opt_type2_ADASYN', opt_type2_ADASYN)
# print('opt_AIC_ADASYN', opt_AIC_ADASYN)

