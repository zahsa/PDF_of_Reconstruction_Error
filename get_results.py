import os
os.environ["CUDA_VISIBLE_DEVICES"]="5"
import numpy as np
import pandas as pd
from tensorflow import keras


import sys
sys.path.insert(0, './preprocessing')
import data_prep
import importlib
import preprocessing.data_prep
importlib.reload(data_prep)
from data_prep import create_data

import model_prep
import preprocessing.model_prep
importlib.reload(model_prep)
from model_prep import AEmodels


importlib.reload(model_prep)



dataset = pd.read_csv('/data/AIS.csv', parse_dates=['time'])



ti = 1199
dataset_size = ti
test_ind = list(range(round(dataset_size * 0.8) + 1,dataset_size))
test_num = 53

# load model

savename = 'model1_sog_all_data_split2_fix_sq300'
model300 = keras.models.load_model('../model/' + 'model_' + savename + '.h5')
savename = 'model1_sog_all_data_split2_fix_sq100'
model100 = keras.models.load_model('../model/' + 'model_' + savename + '.h5')
savename = 'model1_sog_all_data_split2_fix_sq20'
model20 = keras.models.load_model('../model/' + 'model_' + savename + '.h5')


# test_num = len(data.test_ind)

sys.path.insert(0, './analysis')
import analysis.confid_anal
import confid_anal
importlib.reload(confid_anal)
from confid_anal import test_results
# from confid_anal import confidenceLevelThresh
from confid_anal import difference_method
from confid_anal import standardization


time_steps = 100;
test_num = 0
for i, ti in enumerate(test_ind):
    df = dataset[dataset['trajectory'] == ti]
    df_sog = df.loc[:, 'sog']
    if df_sog.shape[0] > 10000 and ~df_sog.isnull().values.any():
        df_sog = pd.DataFrame(df_sog)
        sog_diff = difference_method(df_sog)
        sog_diff_stand = standardization(sog_diff)


        if test_num == 0:
            xx_test = sog_diff_stand

        if test_num > 0:
            xx_test = np.concatenate((xx_test, sog_diff_stand), axis=0)
        test_num += 1


savename = 'model1_sog_all_data_split2_fix_sq300'
model300 = keras.models.load_model('../model/' + 'model_' + savename + '.h5')
savename = 'model1_sog_all_data_split2_fix_sq100'
model100 = keras.models.load_model('../model/' + 'model_' + savename + '.h5')
savename = 'model1_sog_all_data_split2_fix_sq20'
model20 = keras.models.load_model('../model/' + 'model_' + savename + '.h5')


acc = []; sens = []; spec = []; cm = []; prec = []; rec = []; f = [];


test_num = 53

# x_test_G, acc1, sens1, spec1, cm1, prec1, rec1, f1, test_n, y_d, y_pred_d, y_anom_d , y_norm_d = test_results(model20,test_num,test_ind,dataset,pdf_estimator = 'gaussian',time_steps = 20, label_thresh = 0.1)
#


x_test_H_sq300, acc_H_sq300, sens_H_sq300, spec_H_sq300, cm_H_sq300, prec_H_sq300, rec_H_sq300, f_H_sq300 , test_n_H_sq300 , y_d_H_sq300 , y_pred_d_H_sq300 , y_anom_d_H_sq300 , y_norm_d_H_sq300 = test_results(model300, test_num,test_ind,dataset,pdf_estimator = 'histogram',time_steps = 300)


acc_mn_H_sq300 = np.mean(acc_H_sq300,axis=1)
sens_mn_H_sq300 = np.mean(sens_H_sq300,axis=1)
spec_mn_H_sq300 = np.mean(spec_H_sq300,axis=1)
prec_mn_H_sq300 = np.mean(prec_H_sq300,axis=1)
rec_mn_H_sq300 = np.mean(rec_H_sq300,axis=1)
f_mn_H_sq300 = np.mean(f_H_sq300,axis=1)

