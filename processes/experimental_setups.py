import pandas as pd
import numpy as np
import time
import importlib.machinery
es = importlib.machinery.SourceFileLoader('extrasense','/home/sac086/extrasensory/extrasense/extrasense.py').load_module()

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

def run_experiment1(user_id, training_size=100, learning_algo=es.weka_RF, stratified=False):
    #print("Getting personal data...")
    personal_df = es.get_data_from_user_id(user_id, data_type="activity", labeled_only=True)
    timestamps_personal = personal_df.pop('timestamp')
    label_sources_personal = personal_df.pop("label_source")
    personal_labels = personal_df.pop("label")
    
    # train impersonal model
    print("getting impersonal data...")
    impersonal_df = es.get_impersonal_data(leave_users_out=[user_id], data_type="activity", labeled_only=True)
    timestamps_impersonal = impersonal_df.pop('timestamp')
    impersonal_train_labels = impersonal_df.pop("label")
    impersonal_label_sources = impersonal_df.pop('label_source')
    impersonal_ids = impersonal_df.pop("user_id")
    #impersonal_df.dropna()
    
    #impersonal_train_labels = impersonal_train_labels.index[impersonal_df.index]
    
     # standard scale training
    print("Scaling Impersonal...")
    impersonal_scaler = StandardScaler().fit(impersonal_df)
    scaled_impersonal_df = impersonal_scaler.transform(impersonal_df)
    
    impersonal_clf = learning_algo()
    impersonal_clf.fit(scaled_impersonal_df, impersonal_train_labels)
    
    # setup sampler
    if stratified:
        rs = StratifiedShuffleSplit(n_splits=5, train_size=training_size, test_size=100)
        splitter =  rs.split(personal_df, personal_labels)
    else:
        rs = ShuffleSplit(n_splits=5, train_size=training_size, test_size=100)
        splitter = rs.split(personal_df)
    
    run_count = 1
    rows = []
    # run sampler
    for train_ind, test_ind in splitter:
        print("Starting run #%s" % run_count)
        personal_train_df = personal_df.iloc[train_ind]
        personal_train_labels = personal_labels.iloc[train_ind]
        
        val_df = personal_df.iloc[test_ind]
        val_labels = personal_labels.iloc[test_ind]
        
        
        #return personal_train_df, personal_train_labels, impersonal_df, impersonal_train_labels
        hybrid_train_df = pd.concat([impersonal_df, personal_train_df])
        #return impersonal_train_labels, personal_train_labels
        hybrid_train_labels = pd.concat([impersonal_train_labels, personal_train_labels])
        
        # scale 
        personal_scaler = StandardScaler().fit(personal_train_df)
        scaled_personal_df = personal_scaler.transform(personal_train_df)
        
        hybrid_scaler = StandardScaler().fit(hybrid_train_df)
        scaled_hybrid_df = hybrid_scaler.transform(hybrid_train_df)
        
        # build and predict personal model
        personal_clf = learning_algo()
        personal_clf.fit(scaled_personal_df, personal_train_labels)
        personal_scaled_val_df = personal_scaler.transform(val_df)
        personal_predictions = personal_clf.predict(personal_scaled_val_df)
        
        # build and predict hybrid model
        hybrid_clf = learning_algo()
        hybrid_clf.fit(scaled_hybrid_df, hybrid_train_labels)
        hybrid_scaled_val_df = hybrid_scaler.transform(val_df)
        hybrid_predictions = hybrid_clf.predict(hybrid_scaled_val_df)
        
        # impersonal predictions
        impersonal_scaled_val_df = impersonal_scaler.transform(val_df)
        impersonal_predictions = impersonal_clf.predict(impersonal_scaled_val_df)
        
        # validate models
        personal_score = accuracy_score(val_labels, personal_predictions)
        hybrid_score = accuracy_score(val_labels, hybrid_predictions)
        impersonal_score = accuracy_score(val_labels, impersonal_predictions)
        
        print("\tRun #%s" % run_count)
        print("\tpersonal : %s" % personal_score)
        print("\thybrid : %s" % hybrid_score)
        print("\timpersonal : %s" % impersonal_score)
        print("\n")
        
        personal_row = {"user_id" : user_id, 
                        "method":"personal", 
                        "run_num" : run_count,
                        "training_size" : training_size,
                        "accuracy" : personal_score}
        
        hybrid_row = {"user_id" : user_id, 
                        "method":"hybrid", 
                        "run_num" : run_count,
                        "training_size" : training_size,
                        "accuracy" : hybrid_score}
        
        impersonal_row = {"user_id" : user_id, 
                        "method":"impersonal", 
                        "run_num" : run_count,
                        "training_size" : training_size,
                        "accuracy" : impersonal_score}
        rows.append(personal_row)
        rows.append(hybrid_row)
        rows.append(impersonal_row)
        
        run_count += 1
    return rows