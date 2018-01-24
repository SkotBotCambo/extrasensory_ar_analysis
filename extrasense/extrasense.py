import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from collections import Counter
from extrasense.labels import * 

data_dir = '/home/sac086/extrasensory/data/features_labels/'
file_names = os.listdir(data_dir)
user_ids = [fn.split(".")[0] for fn in file_names]

def weka_RF():
    # trees.RandomForest '-P 100 -I 100 -num-slots 1 
    #                     -K 0 -M 1.0 -V 0.001 -S 1' 1116839470751428698a
    clf = RandomForestClassifier(n_estimators=100, 
                                 max_features="log2",
                                 min_samples_leaf=1)
    return clf

def get_uids_from_es_folds():
    folds_dir = '/home/sac086/extrasensory/data/cv_5_folds/'
    folds = []
    for i in range(0,5):
        with open(folds_dir+"fold_"+str(i)+"_test_android_uuids.txt") as f_in:
            android_ids = [f.strip() for f in f_in.readlines()]
        with open(folds_dir+"fold_"+str(i)+"_test_iphone_uuids.txt") as f_in:
            iphone_ids = [f.strip() for f in f_in.readlines()]
        folds.append(android_ids+iphone_ids)
    return folds

def get_base_xgb():
    return xgb.XGBClassifier(
                            learning_rate =0.1,
                            n_estimators=3000,
                            max_depth=5,
                            min_child_weight=1,
                            gamma=0,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            objective= 'binary:logistic',
                            n_jobs=6,
                            scale_pos_weight=1,
                            seed=27)

def xgboost_modelfit(alg, features, labels, activity, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    values = [1 if l == activity else 0 for l in labels.unique()]
    activity_labels  = labels.replace(to_replace=labels.unique(), value=values)
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(features, label=activity_labels)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(features, activity_labels,eval_metric='auc')
        
    #Predict training set:
    train_predictions = alg.predict(features)
    train_predprob = alg.predict_proba(features)[:,1]
        
    #Print model report:
    print("Training Accuracy : %.4g" % accuracy_score(activity_labels, train_predictions))
    print("Training AUC Score (Train): %f" % roc_auc_score(activity_labels, train_predprob))
    
    return alg

def get_normalized_timestamps(df):
    min_timestamp_by_user = {}

    user_ids = df.user_id.unique()

    for user_id in user_ids:
        min_timestamp_by_user[user_id] = df[df['user_id'] == user_id]['timestamp'].min()

    # Add a timestamp column that is normalized by the users minimum timestamp
    # This lets us know how long into the user's participation an event occurred
    normalized_timestamps = []
    for ind, row in df.iterrows():
        norm_ts = (row['timestamp'] - min_timestamp_by_user[row['user_id']]) / (3600. * 24.) # division by 3600 gives the hour, multiply by 24 gives day
        normalized_timestamps.append(norm_ts)
    normalized_timestamps = pd.Series(normalized_timestamps)
    return normalized_timestamps

class ZeroR(object):
    """ZeroR replicates the functionality of a sci-kit learn model using the simple heuristic of predicting
    the modal label of the training data. Intended for baseline measurements only.
    
    Attributes:
        counter : this is a counter object (from the collections python module) 
                  which represents the counts from the training labels and is used for the prediction
    """
    
    def __init__(self):
        """Return a ZeroR classifier"""
        self.counter = None
    
    def fit(self, X, y):
        self.counter = Counter(y)
    
    def predict(self, X):
        input_len = len(X)
        return np.array([self.counter.most_common(1)[0][0]] * input_len)

    def get_params(self, *args, **kwargs):
        return {}


def clean_labels(labels_df):
    labels = []
    clean_labels_df = labels_df.iloc[:,:-1].idxmax(axis=1).str.replace("label:", "")
    return clean_labels_df

def clean_activity_features(features_df):
    null_ind = []

    for ind, row in features_df.iterrows():
        if row.isnull().sum() > 0:
            null_ind.append(ind)
    return features_df.drop(features_df.index[null_ind])

# def clean_data(df, sensors=None, label_type="activity", labeled_only=False):
#     sensors_vals = [sensor_key_dict[sensor] for sensor in sensors]
#     if sensors:
#         sensors_to_keep = [col for col in df.columns if col.split(":")[0] in sensor_vals]
#     if label_type is "activity":
#         features_df = df[df.columns.intersection(sensors_to_keep)]
#         labels_df = clean_labels(df[df.columns.intersection(acc_labels_name)])
#         df = pd.concat((features_df,labels_df,df['timestamp']), axis=1)
#         df = df.rename(columns={0:'label'})
#         df = df.dropna(subset=acc_col_name)
#     else:
#         feature_columns = [col for col in df.columns if col not in label_col_name]
#         features_df = df[df.columns.intersection(feature_columns)]
#         labels_df = clean_labels(df[df.columns.intersection(label_col_name)])
#         df = pd.concat((features_df,labels_df,df['timestamp']), axis=1)
#         df = df.rename(columns={0:'label'})
#         df = df.dropna(subset=feature_columns)
#     if labeled_only:
#         df = df[df.label.notnull()]
#     return df

def clean_data(df, sensors=None, label_type="activity", labeled_only=False, drop_nan_rows=True):
    #print(df.columns)
    # get features
    if sensors is not None:
        sensor_lists = [sensor_key_dict[sensor] for sensor in sensors]
        sensor_vals = [sensor for sensor_list in sensor_lists for sensor in sensor_list]
        feature_columns = [col for col in df.columns if col.split(":")[0] in sensor_vals]
    else:
        feature_columns = [col for col in df.columns if col not in label_col_name]

    features_df = df[df.columns.intersection(feature_columns)]

    # get labels
    if label_type:
        labels = clean_labels(df[df.columns.intersection(context_dict[label_type])])
    else:
        labels = clean_labels(df[df.columns.intersection(label_col_name)])


    labels.name = "label"
    #print(labels_df)

    # construct dataframe with timestamp and label_source
    df = pd.concat((features_df, labels, df['timestamp'],df['label_source']), axis=1)

    # handle nan rows
    if drop_nan_rows:    
        initial_row_count = len(df)
        # df = df[pd.notnull(df[feature_columns])]
        df.dropna(subset=feature_columns, inplace=True)
        dropped_count = initial_row_count - len(df)
        print("Dropped %s rows with nan values" % dropped_count)

    if labeled_only:
        #print(df.columns)
        df = df[df.label.notnull()]
    return df 

def get_data_from_user_id(user_id, label_type=None, labeled_only=False):
    user_df = pd.read_csv(data_dir+user_id+".features_labels.csv")
    user_df = clean_data(user_df, label_type=data_type, labeled_only=labeled_only)
    return user_df

def get_impersonal_data(leave_users_out=None, sensors=None, label_type=None,
                        labeled_only=False, consolidate_stairs=True, include_stairs=True,
                         drop_nan_rows=True):
    uids = []
    user_dfs = []
    for uid in user_ids:
        if (leave_users_out is None) or (uid not in leave_users_out):
            raw_df = pd.read_csv(data_dir+uid+".features_labels.csv")
            user_df = clean_data(raw_df,
                                 sensors=sensors,
                                 label_type=label_type, 
                                 labeled_only=labeled_only,
                                 drop_nan_rows=drop_nan_rows)
            user_df['user_id'] = [uid] * user_df.shape[0]
            user_dfs.append(user_df)
    impersonal_df = pd.concat(user_dfs, ignore_index=True)

    if include_stairs:
        if consolidate_stairs:
            impersonal_df['label'] = impersonal_df['label'].replace(to_replace=['STAIRS_-_GOING_UP', 'STAIRS_-_GOING_DOWN'], value='STAIRS')
        #else:

    return impersonal_df