import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


# dataset = pd.read_csv('/data/zahs/data/DCAIS_[80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 1017, 1024]_None-mmsi_region_[47.5, 49.3, -125.5, -122.5]_01-04_to_30-06_trips.csv', parse_dates=['time'])



def difference_method(x_input):
    x_transformed = x_input.diff().dropna()
    return x_transformed


def standardization(x_input):
    training_min = x_input.min()
    training_max = x_input.max()
    x_standardized = (x_input - training_min) / (training_max - training_min)
#     print("Number of training samples:", len(x_standardized))
    return x_standardized

# Generated training sequences for use in the model. The length of each sequence is time_steps.
def create_sequences(values, time_steps):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
#         print(i)
    return np.stack(output)

class create_data:
    
    def __init__(self, time_steps = 300, n_traj = 1199):
        
        self.time_steps = time_steps
#         self.dataset = pd.DataFrame()
        self.n_traj = n_traj

        self.train_ind = list(range(round(self.n_traj * 0.8)))
        self.test_ind = list(range(round(self.n_traj * 0.8) + 1,self.n_traj))

    def read_data(self,ds_name):
        self.ds_name = ds_name
        if ds_name == 'Tanker':
            dataset = pd.read_csv('/data/zahs/Projects/data/DCAIS_vessels_Tankers[80, 81, 82, 83, 84, 85, 86, 87, 88, 89]_None-mmsi_01-04_to_30-04_trips.csv', parse_dates=['time'])
        elif ds_name == '4class':
             dataset = pd.read_csv('/data/zahs/data/DCAIS_vessels_Classes[30, 37, 80, 60]_None-mmsi_01-04_to_30-04_trips.csv', parse_dates=['time'])
        elif ds_name == '9class':
             dataset = pd.read_csv('/data/zahs/data_ts/data/preprocessed/DCAIS_vessels_[30_ 32_ 34_ 36_ 37_ 52_ 60_ 71_ 80]_20-04_to_30-04_traj.csv', parse_dates=['time'])
                
        dataset['time'] = dataset['time'].astype('datetime64[ns]')
        df = dataset.sort_values(by=['trajectory', "time"])
        self.dataset = df
        
        # check the shape of each sog trajectory
#         for ti in range(df.size):
#             df = dataset[dataset['trajectory']==ti]
#             df_sog = df.loc[:,'sog']
#             df_sog.dropna()
#             if df_sog.shape[0] == 0:
#                 print(ti, df_sog.shape)
# #             else:
# #                 data_size = ti
# #                 break
        
    def printvals(self):
#         print(self.n_traj)
        print('dataset',self.dataset)
        
#     def create_test_train_ind(self): 
#         train_ind = list(range(round(self.dataset_size * 0.8)))
#         test_ind = list(range(round(self.dataset_size * 0.8) + 1,self.dataset_size))
#         self.train_ind = train_ind
#         self.test_ind = test_ind
        
#         if not os.path.exists(self.train_path):
            
    def create_train_seqs(self):
        ''' create train data with the original trajectories '''
        ''' it returns a matrix with shape  10,000 * n_traj ''' 

        train_num = 0
        for t,ti in enumerate(self.train_ind):
#             print('sample ', t)
            df = self.dataset[self.dataset['trajectory']==ti]
            df_sog = df.loc[:,'sog']
            df_sog = pd.DataFrame(df_sog)
            if df_sog.shape[0] > 10000 and ~df_sog.isnull().values.any():   

                x_train_seq = create_sequences(df_sog.values, time_steps)

                if np.shape(np.argwhere(np.isnan(x_train_seq)))[0] > 0:
                    print('nan-----------------')
                    continue
                if train_num==0:
                    xx_train = x_train_seq
        #         print('first train sample',xx_train.shape)

                if train_num > 0:
                    xx_train = np.concatenate((xx_train,x_train_seq),axis=0)
        #             print('train concatenated',xx_train.shape)
                train_num +=1
#                 print('num ', train_num)

#         print('num of training samples', train_num)
        xx_train.shape
        
#         with open("data/zahs/data/out_files/xx_train_Tanker" + self.ds_name + ".pkl", 'wb') as handle:
#             pickle.dump(xx_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        self.xtrain = xx_train
        self.train_num = train_num
        
        
    def create_train_seg_seqs(self):
        ''' create segmented train data '''

        train_num = 0
        for t,ti in enumerate(self.train_ind):
            print('sample ', t)
            df = self.dataset[self.dataset['trajectory']==ti]
            df_sog = df.loc[:,'sog']
            df_sog = pd.DataFrame(df_sog)
            print('shape', df_sog.shape[0])
            if df_sog.shape[0] > 10000 and ~df_sog.isnull().values.any():   

                x_train_seg_seq = create_sequences(df_sog, self.time_steps)

                if np.shape(np.argwhere(np.isnan(x_train_seg_seq)))[0] > 0:
                    print('nan-----------------')
                    continue
                if train_num == 0:
                    xx_train = x_train_seg_seq


                if train_num > 0:
                    xx_train = np.concatenate((xx_train,x_train_seg_seq),axis=0)

                train_num +=1

                
        print('num of training samples', train_num)
        xx_train.shape

#             with open("data/zahs/data/out_files/xx_train_all_data.pkl", 'wb') as handle:
#                 pickle.dump(xx_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.xtrain = xx_train
        self.train_num = train_num

        
        
    def create_train_seqs(self):
            ''' create train data with the stationary standardized trajectories ''' 

            train_num = 0
            for t,ti in enumerate(self.train_ind):
                print('sample ', t)
                df = self.dataset[self.dataset['trajectory']==ti]
                df_sog = df.loc[:,'sog']
                df_sog = pd.DataFrame(df_sog)
                if df_sog.shape[0] > 10000 and ~df_sog.isnull().values.any():   
            #         print('trajectory shape',df_sog.shape[0])
                    sog_diff = difference_method(df_sog)
                    sog_diff_stand = standardization(sog_diff)
                    x_train_diff_stand_seq = create_sequences(sog_diff_stand.values, self.time_steps)

                    if np.shape(np.argwhere(np.isnan(x_train_diff_stand_seq)))[0] > 0:
                        print('nan-----------------')
                        continue
                    if train_num==0:
                        xx_train = x_train_diff_stand_seq
            #         print('first train sample',xx_train.shape)

                    if train_num > 0:
                        xx_train = np.concatenate((xx_train,x_train_diff_stand_seq),axis=0)
            #             print('train concatenated',xx_train.shape)
                    train_num +=1
#                     print('num ', train_num)

            print('num of training samples', train_num)
            xx_train.shape

#             with open("data/zahs/data/out_files/xx_train_all_data_paper2.pkl", 'wb') as handle:
#                 pickle.dump(xx_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

            self.xtrain = xx_train
            self.train_num = train_num

        
#############TEST#######################################
#         test_num = 0
        
#         for i , ti in enumerate(test_ind):
#             df = dataset[dataset['trajectory']==ti]
#             df_sog = df.loc[:,'sog']
#             if df_sog.shape[0] > 10000 and ~df_sog.isnull().values.any(): 
#                 df_sog = pd.DataFrame(df_sog)
#                 sog_diff = difference_method(df_sog)
#                 sog_diff_stand = standardization(sog_diff)
#                 x_test_diff_stand_seq = create_sequences(sog_diff_stand.values, time_steps)
#                 if np.shape(np.argwhere(np.isnan(x_test_diff_stand_seq)))[0] > 0:
#                     print('nan-----------------')
#                     continue
#                 test_num +=1
             
    def create_train_traj(self):
        train_num = 0
        for t,ti in enumerate(self.train_ind):
#             print('sample ', t)
            df = self.dataset[self.dataset['trajectory']==ti]
            df_sog = df.loc[:,'sog']
            df_sog = pd.DataFrame(df_sog)
            if df_sog.shape[0] > 10000 and ~df_sog.isnull().values.any():   
        #         print('trajectory shape',df_sog.shape[0])
                df_x = df_sog[0:10001]
                if df_x.isnull().values.any() == True:
                    print('nan-----------------')
                    continue
                if train_num==0:
                    xx_train = df_x
                    print('first train sample',xx_train.shape)

                if train_num > 0:
                    xx_train = np.concatenate((xx_train,df_x),axis=1)
                    print('train concatenated',xx_train.shape)
                train_num +=1
                print('num ', train_num)

        print('num of training samples', train_num)
        xx_train.shape
        
#         with open("data/zahs/data/out_files/xx_train_all_data_traj.pkl", 'wb') as handle:
#             pickle.dump(xx_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        self.xtrain = xx_train
        self.train_num = train_num   