import pandas as pd
import numpy as np
import os

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
                        'label:BICYCLING']

data_dir = '../data/features_labels/'
file_names = os.listdir(data_dir)
user_ids = [fn.split(".")[0] for fn in file_names]

def clean_labels(labels_df):
    labels = []
    for ind, row in labels_df.iterrows():
        #print(np.argmax(row))
        max_label = np.argmax(row)
        if np.isnan(np.max(row)):
            label = max_label
        else:
            if ":" in max_label:
                label = max_label.split(":")[1]
            else:
                label=max_label
        labels.append(label)
    return pd.Series(labels)

def clean_activity_features(features_df):
    null_ind = []

    for ind, row in features_df.iterrows():
        if row.isnull().sum() > 0:
            null_ind.append(ind)
    return features_df.drop(features_df.index[null_ind])

def get_data_from_user_id(user_id, data_type=None, labeled_only=False):
    user_df = pd.read_csv(data_dir+user_id+".features_labels.csv")
    if data_type is "activity":
        features_df = user_df[user_df.columns.intersection(acc_col_name)]
        labels_df = clean_labels(user_df[user_df.columns.intersection(acc_labels_name)])
        user_df = pd.concat((features_df,labels_df), axis=1)
        user_df = user_df.rename(columns={0:'label'})
        user_df = user_df.dropna(subset=acc_col_name)
    else:
        feature_columns = [col for col in user_df.columns if col not in label_col_name]
        features_df = user_df[user_df.columns.intersection(feature_columns)]
        labels_df = clean_labels(user_df[user_df.columns.intersection(label_col_name)])
        user_df = pd.concat((features_df,labels_df), axis=1)
        print(user_df.shape)
        user_df = user_df.rename(columns={0:'label'})
        user_df = user_df.dropna(subset=feature_columns)
    if labeled_only:
        null_mask = labels_df.notnull()
        user_df = user_df[null_mask]
    return user_df