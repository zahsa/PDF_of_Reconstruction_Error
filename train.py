import os
os.environ["CUDA_VISIBLE_DEVICES"]="10"
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import ModelCheckpoint
from sklearn.neighbors import KernelDensity
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import pickle


def difference_method(x_input):
    x_transformed = x_input.diff().dropna()
    return x_transformed

def standardization(x_input):
    training_min = x_input.min()
    training_max = x_input.max()
    x_standardized = (x_input - training_min) / (training_max - training_min)
#     print("Number of training samples:", len(x_standardized))
    return x_standardized

# Generated training sequences for use in the model.
def create_sequences(values, time_steps):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
#         print(i)
    return np.stack(output)


def AEmodel1(x_train):
    model = keras.Sequential(
        [
            layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
            layers.Conv1D(
                filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
            ),

            layers.Conv1D(
                filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Conv1DTranspose(
                filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
            ),

            layers.Conv1DTranspose(
                filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    model.summary()
    return (model)



dataset = pd.read_csv('/data/AIS_data.csv', parse_dates=['time'])

dataset['time'] = dataset['time'].astype('datetime64[ns]')
dataset = dataset.sort_values(by=['trajectory', "time"])

# get the real dataset size
dataset_size = dataset.size
for ti in range(dataset_size):
    df = dataset[dataset['trajectory']==ti]
    df_sog = df.loc[:,'sog']
    df_sog.dropna()
    if df_sog.shape[0] > 0:
        # print(ti, df_sog.shape)
    else:
        data_size = ti
        break
dataset_size = ti

y = list(range(df.size))

# split data to test and train
train_ind = list(range(round(dataset_size * 0.8)))
test_ind = list(range(round(dataset_size * 0.8) + 1,dataset_size))

time_steps = 100;  # 300,100,20
# xx_train = np.empty([10, time_steps,1])
train_num = 0
for t, ti in enumerate(train_ind):
    print('sample ', t)
    df = dataset[dataset['trajectory'] == ti]
    df_sog = df.loc[:, 'sog']
    df_sog = pd.DataFrame(df_sog)
    if df_sog.shape[0] > 10000 and ~df_sog.isnull().values.any():
        #         print('trajectory shape',df_sog.shape[0])
        sog_diff = difference_method(df_sog)
        sog_diff_stand = standardization(sog_diff)
        x_train_diff_stand_seq = create_sequences(sog_diff_stand.values, time_steps)

        if np.shape(np.argwhere(np.isnan(x_train_diff_stand_seq)))[0] > 0:
            print('nan-----------------')
            continue
        if train_num == 0:
            xx_train = x_train_diff_stand_seq
        #         print('first train sample',xx_train.shape)

        if train_num > 0:
            xx_train = np.concatenate((xx_train, x_train_diff_stand_seq), axis=0)
        #             print('train concatenated',xx_train.shape)
        train_num += 1
        print('num ', train_num)

print('num of training samples', train_num)

# write training sequences
with open("/data/xx_train_all_data_fix_sq100.pkl", 'wb') as handle:
    pickle.dump(xx_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

savename = 'model1_sog_all_data_split2_fix_sq100'

# main loop for training
checkpoint_path = '../model/' + savename + '.h5'


# x_train = xx_train#[:1000,:,:]
model = AEmodel1(x_train)


# checkpoint = ModelCheckpoint(checkpoint_path, monitor='loss', verbose=1,
#     save_best_only=True, mode='auto', period=1)

with open("/data/xx_train_all_data_fix_sq100.pkl", 'rb') as handle:
    x_train = pickle.load(handle)

history = model.fit(
    x_train,
    x_train,
    epochs=100,
    batch_size=128,
    validation_split=0.1,
#     callbacks=[
#         keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, mode="min")
#     ],

    callbacks   = [
      keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min', min_delta=0.0001),
      ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
]
)
model.save('../model/' + savename + '.h5')



