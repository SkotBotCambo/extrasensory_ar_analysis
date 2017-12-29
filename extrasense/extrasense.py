import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier


acc_col_name = ['raw_acc:magnitude_stats:mean',
 'raw_acc:magnitude_stats:std',
 'raw_acc:magnitude_stats:moment3',
 'raw_acc:magnitude_stats:moment4',
 'raw_acc:magnitude_stats:percentile25',
 'raw_acc:magnitude_stats:percentile50',
 'raw_acc:magnitude_stats:percentile75',
 'raw_acc:magnitude_stats:value_entropy',
 'raw_acc:magnitude_stats:time_entropy',
 'raw_acc:magnitude_spectrum:log_energy_band0',
 'raw_acc:magnitude_spectrum:log_energy_band1',
 'raw_acc:magnitude_spectrum:log_energy_band2',
 'raw_acc:magnitude_spectrum:log_energy_band3',
 'raw_acc:magnitude_spectrum:log_energy_band4',
 'raw_acc:magnitude_spectrum:spectral_entropy',
 'raw_acc:magnitude_autocorrelation:period',
 'raw_acc:magnitude_autocorrelation:normalized_ac',
 'raw_acc:3d:mean_x',
 'raw_acc:3d:mean_y',
 'raw_acc:3d:mean_z',
 'raw_acc:3d:std_x',
 'raw_acc:3d:std_y',
 'raw_acc:3d:std_z',
 'raw_acc:3d:ro_xy',
 'raw_acc:3d:ro_xz',
 'raw_acc:3d:ro_yz']
gyro_col_name = ['proc_gyro:magnitude_stats:mean',
 'proc_gyro:magnitude_stats:std',
 'proc_gyro:magnitude_stats:moment3',
 'proc_gyro:magnitude_stats:moment4',
 'proc_gyro:magnitude_stats:percentile25',
 'proc_gyro:magnitude_stats:percentile50',
 'proc_gyro:magnitude_stats:percentile75',
 'proc_gyro:magnitude_stats:value_entropy',
 'proc_gyro:magnitude_stats:time_entropy',
 'proc_gyro:magnitude_spectrum:log_energy_band0',
 'proc_gyro:magnitude_spectrum:log_energy_band1',
 'proc_gyro:magnitude_spectrum:log_energy_band2',
 'proc_gyro:magnitude_spectrum:log_energy_band3',
 'proc_gyro:magnitude_spectrum:log_energy_band4',
 'proc_gyro:magnitude_spectrum:spectral_entropy',
 'proc_gyro:magnitude_autocorrelation:period',
 'proc_gyro:magnitude_autocorrelation:normalized_ac',
 'proc_gyro:3d:mean_x',
 'proc_gyro:3d:mean_y',
 'proc_gyro:3d:mean_z',
 'proc_gyro:3d:std_x',
 'proc_gyro:3d:std_y',
 'proc_gyro:3d:std_z',
 'proc_gyro:3d:ro_xy',
 'proc_gyro:3d:ro_xz',
 'proc_gyro:3d:ro_yz']
label_col_name = ['label:LYING_DOWN',
 'label:SITTING',
 'label:FIX_walking',
 'label:FIX_running',
 'label:BICYCLING',
 'label:SLEEPING',
 'label:LAB_WORK',
 'label:IN_CLASS',
 'label:IN_A_MEETING',
 'label:LOC_main_workplace',
 'label:OR_indoors',
 'label:OR_outside',
 'label:IN_A_CAR',
 'label:ON_A_BUS',
 'label:DRIVE_-_I_M_THE_DRIVER',
 'label:DRIVE_-_I_M_A_PASSENGER',
 'label:LOC_home',
 'label:FIX_restaurant',
 'label:PHONE_IN_POCKET',
 'label:OR_exercise',
 'label:COOKING',
 'label:SHOPPING',
 'label:STROLLING',
 'label:DRINKING__ALCOHOL_',
 'label:BATHING_-_SHOWER',
 'label:CLEANING',
 'label:DOING_LAUNDRY',
 'label:WASHING_DISHES',
 'label:WATCHING_TV',
 'label:SURFING_THE_INTERNET',
 'label:AT_A_PARTY',
 'label:AT_A_BAR',
 'label:LOC_beach',
 'label:SINGING',
 'label:TALKING',
 'label:COMPUTER_WORK',
 'label:EATING',
 'label:TOILET',
 'label:GROOMING',
 'label:DRESSING',
 'label:AT_THE_GYM',
 'label:STAIRS_-_GOING_UP',
 'label:STAIRS_-_GOING_DOWN',
 'label:ELEVATOR',
 'label:OR_standing',
 'label:AT_SCHOOL',
 'label:PHONE_IN_HAND',
 'label:PHONE_IN_BAG',
 'label:PHONE_ON_TABLE',
 'label:WITH_CO-WORKERS',
 'label:WITH_FRIENDS',
 'label_source']

acc_labels_name = ["label:LYING_DOWN", 'label:SITTING', 'label:STAIRS_-_GOING_UP', 
                        'label:STAIRS_-_GOING_DOWN', 'label:FIX_walking', 'label:FIX_running',
                        'label:BICYCLING', 'label_source']

data_dir = '../data/features_labels/'
file_names = os.listdir(data_dir)
user_ids = [fn.split(".")[0] for fn in file_names]

def weka_RF():
    # trees.RandomForest '-P 100 -I 100 -num-slots 1 
    #                     -K 0 -M 1.0 -V 0.001 -S 1' 1116839470751428698a
    clf = RandomForestClassifier(n_estimators=100, 
                                 max_features="log2",
                                 min_samples_leaf=1)
    return clf

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

def clean_labels(labels_df, include_label_source=True):
    labels = []
    clean_labels_df = labels_df.iloc[:,:-1].idxmax(axis=1).str.replace("label:", "")
    if include_label_source:
        clean_labels_df = pd.concat((clean_labels_df, labels_df['label_source']), axis=1)
    return clean_labels_df

def clean_activity_features(features_df):
    null_ind = []

    for ind, row in features_df.iterrows():
        if row.isnull().sum() > 0:
            null_ind.append(ind)
    return features_df.drop(features_df.index[null_ind])

def clean_data(df, data_type="activity", labeled_only=False):
    if data_type is "activity":
        features_df = df[df.columns.intersection(acc_col_name)]
        labels_df = clean_labels(df[df.columns.intersection(acc_labels_name)])
        df = pd.concat((features_df,labels_df,df['timestamp']), axis=1)
        df = df.rename(columns={0:'label'})
        df = df.dropna(subset=acc_col_name)
    else:
        feature_columns = [col for col in df.columns if col not in label_col_name]
        features_df = df[df.columns.intersection(feature_columns)]
        labels_df = clean_labels(df[df.columns.intersection(label_col_name)])
        df = pd.concat((features_df,labels_df,df['timestamp']), axis=1)
        df = df.rename(columns={0:'label'})
        df = df.dropna(subset=feature_columns)
    if labeled_only:
        df = df[df.label.notnull()]
    return df

def get_data_from_user_id(user_id, data_type=None, labeled_only=False):
    user_df = pd.read_csv(data_dir+user_id+".features_labels.csv")
    user_df = clean_data(user_df, data_type=data_type, labeled_only=labeled_only)
    return user_df

def get_impersonal_data(leave_users_out=None, data_type=None, labeled_only=False):
    uids = []
    user_dfs = []
    for uid in user_ids:
        if (leave_users_out is None) or (uid not in leave_users_out):
            user_df = clean_data(pd.read_csv(data_dir+uid+".features_labels.csv"), data_type=data_type, labeled_only=labeled_only)
            user_df['user_id'] = [uid] * user_df.shape[0]
            user_dfs.append(user_df)
    impersonal_df = pd.concat(user_dfs, ignore_index=True)
    return impersonal_df