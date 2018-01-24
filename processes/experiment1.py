#!/usr/bin/python

print("Loading modules...")
import sys, getopt
import pandas as pd
import numpy as np
import time
import importlib.machinery
from datetime import datetime
es = importlib.machinery.SourceFileLoader('extrasense','/home/sac086/extrasensory/extrasense/extrasense.py').load_module()
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import ipyparallel as ipp
results_directory = '/home/sac086/extrasensory/results/experiment1/'
todays_date = datetime.now()

import_string = '''import pandas as pd
import numpy as np
import time
import importlib.machinery
es = importlib.machinery.SourceFileLoader('extrasense','/home/sac086/extrasensory/extrasense/extrasense.py').load_module()
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
exp = importlib.machinery.SourceFileLoader('extrasense','/home/sac086/extrasensory/processes/experimental_setups.py').load_module()
'''

def get_command(training_sizes=None, stratification=None, learning_algo=None):
    init_str = '''rows, errors = [], []'''
    if training_sizes == None:
        init_str +='''\ntraining_sizes=[5,10,20,30,40,50,60]'''
    else:
        init_str += "\n"+str(training_sizes)
    init_str+='''\nfor user_id in user_ids:
    for ts in training_sizes:
        try:
            user_rows = exp.run_experiment1(user_id, training_size=ts'''
    
    if learning_algo:
        init_str+=''', learning_algo='''+learning_algo
    if stratification:
        init_str+= ''', stratified=True'''
    init_str+=''')'''
    init_str+='''
        except ValueError as ve:
            errors.append(ve)
            continue
        rows += user_rows
'''
    return init_str

def main(argv):
    print("Initializing cores...")
    c = ipp.Client()
    dview = c[:]
    dview.block=True
    dview.scatter('user_ids', es.user_ids)

    print("Loading modules to cores...")
    asr = dview.execute(import_string)

    print("Running experiment without class stratification...")
    start = time.time()
    command_str = get_command(learning_algo='es.ZeroR')
    asr = dview.execute(command_str)
    print("finished running processes")
    rows = dview.gather('rows')
    finish = time.time()
    print("Took %.3f hours" % ((finish - start) / 3600))
    scores_df = pd.DataFrame(rows)
    file_out_name = '%s-%s-%s_%s_exp1_no_stratification.pickle' % (todays_date.year, todays_date.month, todays_date.day, todays_date.hour)
    file_out_loc = results_directory + file_out_name
    scores_df.to_pickle(file_out_loc)

    start = time.time()
    command_str = get_command(learning_algo='es.ZeroR', stratification=True)
    asr = dview.execute(command_str)
    print("finished running processes")
    rows = dview.gather('rows')
    finish = time.time()
    print("Took %.3f hours" % ((finish - start) / 3600))
    scores_df = pd.DataFrame(rows)
    file_out_name = '%s-%s-%s_%s_exp1_with_stratification.pickle' % (todays_date.year, todays_date.month, todays_date.day, todays_date.hour)
    file_out_loc = results_directory + file_out_name
    scores_df.to_pickle(file_out_loc)
    print("Finished experiments")

if __name__ == "__main__":
    main(None)
# # without class stratification
# print("Running experiment without class stratification...")
# command1 = '''
# rows = []
# errors = []
# training_sizes = [5,10,20,30,40,50,60]
# for user_id in user_ids:
#     for ts in training_sizes:
#         print("Getting scores for %s" % user_id)
#         start = time.time()
#         try:
#             user_rows = exp.run_experiment1(user_id, training_size=ts)
#         except ValueError as ve:
#             errors.append(ve)
#             continue
#         finish = time.time()
#         duration_in_minutes = (finish - start) / 60.
#         print("\ttook %s minutes" % (duration_in_minutes))
#         rows += user_rows
# '''
# start = time.time()
# asr = dview.execute(command1)
# print("finished running processes")
# rows = dview.gather('rows')
# finish = time.time()
# print("Took %.3f hours" % ((finish - start) / 3600))
# scores_df = pd.DataFrame(rows)
# file_out_name = '%s-%s-%s_%s_exp1_no_stratification.pickle' % (todays_date.year, todays_date.month, todays_date.day, todays_date.hour)
# file_out_loc = results_directory + file_out_name
# scores_df.to_pickle(file_out_loc)

# print("Running experiment with class stratification...")
# command2 = '''
# rows = []
# errors = []
# training_sizes = [5,10,20,30,40,50,60]
# for user_id in user_ids:
#     for ts in training_sizes:
#         print("Getting scores for %s" % user_id)
#         start = time.time()
#         try:
#             user_rows = exp.run_experiment1(user_id, training_size=ts, stratified=True)
#         except ValueError as ve:
#             errors.append(ve)
#             continue
#         finish = time.time()
#         duration_in_minutes = (finish - start) / 60.
#         print("\ttook %s minutes" % (duration_in_minutes))
#         rows += user_rows
# '''
# start = time.time()
# asr = dview.execute(command2)
# print("finished running processes")
# rows = dview.gather('rows')
# finish = time.time()
# print("Took %.3f hours" % ((finish - start) / 3600))
# scores_df = pd.DataFrame(rows)
# file_out_name = '%s-%s-%s_%s_exp1_with_stratification.pickle' % (todays_date.year, todays_date.month, todays_date.day, todays_date.hour)
# file_out_loc = results_directory + file_out_name
# scores_df.to_pickle(file_out_loc)
# print("Finished experiments")