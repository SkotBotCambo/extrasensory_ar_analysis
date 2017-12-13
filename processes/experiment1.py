print("Loading modules...")
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

print("Initializing cores...")

c = ipp.Client()

dview = c[:]

dview.block=True

dview.scatter('user_ids', es.user_ids)

results_directory = '/home/sac086/extrasensory/results/'
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

print("Loading modules to cores...")
asr = dview.execute(import_string)

# without class stratification
print("Running experiment without class stratification...")
command1 = '''
rows = []
errors = []
training_sizes = [5,10,20,30,40,50,60]
for user_id in user_ids:
    for ts in training_sizes:
        print("Getting scores for %s" % user_id)
        start = time.time()
        try:
            user_rows = exp.run_experiment1(user_id, training_size=ts)
        except ValueError as ve:
            errors.append(ve)
            continue
        finish = time.time()
        duration_in_minutes = (finish - start) / 60.
        print("\ttook %s minutes" % (duration_in_minutes))
        rows += user_rows
'''
start = time.time()
asr = dview.execute(command1)
print("finished running processes")
rows = dview.gather('rows')
finish = time.time()
print("Took %.3f hours" % ((finish - start) / 3600))
scores_df = pd.DataFrame(rows)
file_out_name = '%s-%s-%s_%s_exp1_no_stratification.pickle' % (todays_date.year, todays_date.month, todays_date.day, todays_date.hour)
file_out_loc = results_directory + file_out_name
scores_df.to_pickle(file_out_loc)

print("Running experiment with class stratification...")
command2 = '''
rows = []
errors = []
training_sizes = [5,10,20,30,40,50,60]
for user_id in user_ids:
    for ts in training_sizes:
        print("Getting scores for %s" % user_id)
        start = time.time()
        try:
            user_rows = exp.run_experiment1(user_id, training_size=ts, stratified=True)
        except ValueError as ve:
            errors.append(ve)
            continue
        finish = time.time()
        duration_in_minutes = (finish - start) / 60.
        print("\ttook %s minutes" % (duration_in_minutes))
        rows += user_rows
'''
start = time.time()
asr = dview.execute(command2)
print("finished running processes")
rows = dview.gather('rows')
finish = time.time()
print("Took %.3f hours" % ((finish - start) / 3600))
scores_df = pd.DataFrame(rows)
file_out_name = '%s-%s-%s_%s_exp1_with_stratification.pickle' % (todays_date.year, todays_date.month, todays_date.day, todays_date.hour)
file_out_loc = results_directory + file_out_name
scores_df.to_pickle(file_out_loc)
print("Finished experiments")