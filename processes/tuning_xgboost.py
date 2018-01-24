import pandas as pd
import numpy as np
import time
import pickle
import os
import importlib.machinery
from collections import Counter
import sys
sys.path.append('/home/sac086/extrasensory/')
import extrasense as es
from sklearn.metrics import accuracy_score, make_scorer, roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, precision_score, f1_score, average_precision_score, recall_score

import xgboost as xgb

def tune_estimators1(X_train, y_train, X_test, y_test):
    alg_params = {}
    for l in y_train.unique():
        print("Training %s classifier" % l)
        alg = es.xgboost_modelfit(es.get_base_xgb(), X_train, y_train, l)
        
        print("Getting Test Predictions")
        y_pred = alg.predict(X_test)
        y_pred_proba = alg.predict_proba(X_test)[:,1]
        
        values = [1 if i == l else 0 for i in y_train.unique()]
        y_test_activity = y_test.replace(to_replace=y_train.unique(), value=values)
        print("Test Accuracy : %.4g" % accuracy_score(y_test_activity, y_pred))
        print("Test AUC Score (Train): %f\n" % roc_auc_score(y_test_activity, y_pred_proba))
        alg_params[l] = alg.get_params()
    return alg_params

def tune_max_depth_min_child_weight2(X_train, y_train, user_ids_train, X_test, y_test, step1_params):
    step2_results = {}

    param_test1 = {'max_depth':range(3,10,2),
              'min_child_weight': range(1,6,2)}

    # set new alg dictionary
    for activity, old_param in step1_params.items():
        new_alg = es.get_base_xgb()
        new_alg.n_estimators = old_param['n_estimators']
        new_alg.n_jobs = 12
        
        values = [1 if l == activity else 0 for l in y_train.unique()]
        y_train_activity = y_train.replace(to_replace=y_train.unique(), value=values)
        y_test_activity = y_test.replace(to_replace=y_test.unique(), value=values)

        print("Tuning Alg for %s" % activity)
        cv = list(GroupKFold(n_splits=3).split(X_train, y_train_activity, user_ids_train))
        gsv = GridSearchCV(estimator = new_alg, param_grid = param_test1,
                           scoring='roc_auc', n_jobs=1, iid=False, cv=cv, verbose=2, refit=True)
        gsv.fit(X_train, y_train_activity, user_ids_train)
        print("Training results:")
        print("best params : %s" % gsv.best_params_)
        print("best score : %s" % gsv.best_score_)

        y_pred = gsv.predict(X_test)
        y_pred_proba = gsv.predict_proba(X_test)[:,1]

        print("Test Accuracy : %.4g" % accuracy_score(y_test_activity, y_pred))
        print("Test AUC Score (Train): %f\n" % roc_auc_score(y_test_activity, y_pred_proba))

        step2_results[activity] = (gsv.cv_results_, gsv.best_params_, gsv.best_score_)
    return step2_results

if __name__ == "__main__":
    # loading and initializing the features
    print("Loading data...")
    features_df = es.get_impersonal_data(leave_users_out=[], data_type="activity", labeled_only=True)
    
    # remove nan rows
    no_label_indeces = features_df.label.isnull()
    features_df = features_df[~no_label_indeces]

    timestamps = features_df.pop('timestamp')
    label_source = features_df.pop("label_source")
    labels = features_df.pop("label")
    user_ids = features_df.pop("user_id")

    print("Creating training and validation sets")
    # hold out a test set
    splitter = GroupShuffleSplit(n_splits=10, test_size=12).split(features_df, labels, user_ids)
    X_train = []
    y_train = []

    X_test = []
    y_test = []
    y_test_counter = Counter()

    while np.sum([True if l in y_test_counter else False for l in labels.unique()]) != len(labels.unique()):
        train_ind, test_ind = next(splitter)

        X_train = features_df.iloc[train_ind]
        y_train = labels.iloc[train_ind]
        user_ids_train = user_ids.iloc[train_ind]

        X_test = features_df.iloc[test_ind]
        y_test = labels.iloc[test_ind]
        user_ids_test = user_ids.iloc[test_ind]

        # test that y_test has at least one of all labels
        y_test_counter = Counter(y_test)
        for l in labels.unique():
            print("%s : %s" % (l, y_test_counter[l]))

        #print([l for l in labels.unique()])
        #print([True if l in y_test_counter else False for l in labels.unique()])
        #print("%s are labels" % np.sum([True if l in y_test_counter else False for l in labels.unique()]))
        #print("There are %s labels" % len(labels.unique()))
        print("\n")

    # step 1 : figure out how many estimators are needed for each classifier
    step1_params_filename= "step1_params.pickle"
    if os.path.isfile(step1_params_filename):
        with open(step1_params_filename, "rb") as fIn:
            print("Loading previous Step 1 parameter tuning state...")
            step1_params = pickle.load(fIn)
            for activity, alg_param in step1_params.items():
                print("%s : %s" % (activity, alg_param['n_estimators']))
    else:
        print("Step 1 Tuning...")
        step1_params = tune_estimators1(X_train, y_train, X_test, y_test)
        for activity, alg_param in step1_params.items():
            print("%s : %s" % (activity, alg_param['n_estimators']))
        with open(step1_params_filename, "wb") as fOut:
            print("Saving Step 1 parameter state")
            pickle.dump(step1_params, fOut)

    # step 2 : tune max_depth and min_child_weight with the appropriate number of estimators
    step2_results_filename= "step2_params.pickle"
    if os.path.isfile(step2_results_filename):
        with open(step2_results_filename, "rb") as fIn:
            print("Loading previous Step 2 parameter tuning state...")
            step2_results = pickle.load(fIn)
            print(alg_params)
    else:
        print("Step 2 Tuning...")
        step2_results = tune_max_depth_min_child_weight2(X_train, y_train, user_ids_train, X_test, y_test, step1_params)
        with open(step2_results_filename, "wb") as fOut:
            print("Saving Step 2 results")
            pickle.dump(step2_results, fOut)
    # step 3 : tune the gamma parameter (optional : tune with estimators again)
    # step 4 : tune subsample and colsample_bytree
    # step 5 : tune regularization parameters
    # step 6 : reduce learning rate 

