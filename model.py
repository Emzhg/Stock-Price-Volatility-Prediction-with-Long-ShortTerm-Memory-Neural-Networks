import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math
import tensorflow as tf
from keras.models import Sequential, Input
from keras.layers import Dense
from keras.layers import LSTM, GRU, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
plt.rc('figure', figsize=(20, 5))
#from keras.layers.core import
import keras.backend as K


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = list(dataset[i:(i+look_back),:])
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def custom_loss(y_true, y_pred):
    # extract the "next day's price" of tensor
    y_true_next = y_true[1:]
    y_pred_next = y_pred[1:]

    # extract the "today's price" of tensor
    y_true_tdy = y_true[:-1]
    y_pred_tdy = y_pred[:-1]

    print('Shape of y_pred_back -', y_pred_tdy.get_shape())

    # substract to get up/down movement of the two tensors
    y_true_diff = tf.subtract(y_true_next, y_true_tdy)
    y_pred_diff = tf.subtract(y_pred_next, y_pred_tdy)

    # create a standard tensor with zero value for comparison
    standard = tf.zeros_like(y_pred_diff)

    # compare with the standard; if true, UP; else DOWN
    y_true_move = tf.greater_equal(y_true_diff, standard)
    y_pred_move = tf.greater_equal(y_pred_diff, standard)
    y_true_move = tf.reshape(y_true_move, [-1])
    y_pred_move = tf.reshape(y_pred_move, [-1])

    # find indices where the directions are not the same
    condition = tf.not_equal(y_true_move, y_pred_move)
    indices = tf.where(condition)

    # move one position later
    ones = tf.ones_like(indices)
    indices = tf.add(indices, ones)
    indices = K.cast(indices, dtype='int32')

    # create a tensor to store directional loss and put it into custom loss output
    direction_loss = tf.ones_like(y_pred)
    updates = K.cast(tf.ones_like(indices), dtype='float32')
    alpha = 100

    direction_loss = tf.tensor_scatter_nd_update(direction_loss, indices, alpha * updates)
    custom_loss = K.mean(tf.add(K.square(y_true - y_pred), direction_loss), axis=-1)

    return custom_loss

# load the dataset
def load_data(ticker='bny', region='us'):
    dataset = pd.read_csv(f'datasets/dataset_{ticker}_{region}.csv').set_index('date')
    dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
    dataset.dropna(inplace=True)
    #data = dataset[dataset['garch_vol_10'].notna()]
    return dataset

def construct_variables(dataset):
    target = dataset['garch_vol_10'].values
    y = target.reshape(-1,1)
    f1 = dataset['log_return'].values
    feature1 = f1.reshape(-1,1)
    f2 = dataset['log_volume_chg'].values
    feature2 = f2.reshape(-1,1)
    f3 = dataset['log_trading_range'].values
    feature3 = f3.reshape(-1,1)
    f4 = dataset['vol_10'].values
    feature4 = f4.reshape(-1,1)
    f5 = dataset['vol_30'].values
    feature5 = f5.reshape(-1,1)
    f6 = dataset['garch_vol_10'].values
    feature6 = f6.reshape(-1,1)
    return y, feature1, feature2, feature3, feature4, feature5, feature6

# normalize the dataset
def normalize_dataset(y, feature1, feature2, feature3, feature4, feature5, feature6):
    scaler1 = StandardScaler()
    scaler2 = StandardScaler()
    scaler3 = StandardScaler()
    scaler4 = StandardScaler()
    scaler5 = StandardScaler()
    scaler6 = StandardScaler()
    scaler7 = StandardScaler()

    data = scaler1.fit_transform(y)
    feature1 = scaler2.fit_transform(feature1)
    feature2 = scaler3.fit_transform(feature2)
    feature3 = scaler4.fit_transform(feature3)
    feature4 = scaler5.fit_transform(feature4)
    feature5 = scaler6.fit_transform(feature5)
    feature6 = scaler7.fit_transform(feature6)
    return scaler1, scaler2, scaler3, scaler4, scaler5, scaler6, scaler7, data, feature1, feature2, feature3, feature4, feature5, feature6

# split into train and test sets, 20% test data, 80% training data
def split_dataset(data, feature1, feature2, feature3, feature4, feature5, feature6, ratio = 0.8, look_back = 30):
    train_size = int(len(data) * ratio)
    test_size = len(data) - train_size
    train, test = data[0:train_size,:], data[train_size:len(data),:]
    feature1_train, feature1_test = feature1[0:train_size,:], feature1[train_size:len(feature1),:]
    feature2_train, feature2_test = feature2[0:train_size,:], feature2[train_size:len(feature2),:]
    feature3_train, feature3_test = feature3[0:train_size,:], feature3[train_size:len(feature3),:]
    feature4_train, feature4_test = feature4[0:train_size,:], feature4[train_size:len(feature4),:]
    feature5_train, feature5_test = feature5[0:train_size,:], feature5[train_size:len(feature5),:]
    feature6_train, feature6_test = feature6[0:train_size,:], feature6[train_size:len(feature6),:]
    # reshape into X=t and Y=t+1, timestep 60

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    feature1_trainX, feature1_trainY = create_dataset(feature1_train, look_back)
    feature2_trainX, feature2_trainY = create_dataset(feature2_train, look_back)
    feature3_trainX, feature3_trainY = create_dataset(feature3_train, look_back)
    feature4_trainX, feature4_trainY = create_dataset(feature4_train, look_back)
    feature5_trainX, feature5_trainY = create_dataset(feature5_train, look_back)
    feature6_trainX, feature6_trainY = create_dataset(feature6_train, look_back)

    feature1_testX, feature1_testY = create_dataset(feature1_test, look_back)
    feature2_testX, feature2_testY = create_dataset(feature2_test, look_back)
    feature3_testX, feature3_testY = create_dataset(feature3_test, look_back)
    feature4_testX, feature4_testY = create_dataset(feature4_test, look_back)
    feature5_testX, feature5_testY = create_dataset(feature5_test, look_back)
    feature6_testX, feature6_testY = create_dataset(feature6_test, look_back)
    return train_size, test_size, trainX, trainY, testX, testY, feature1_trainX, feature1_trainY, feature2_trainX, feature2_trainY, feature3_trainX, feature3_trainY, feature4_trainX, feature4_trainY, feature5_trainX, feature5_trainY, feature6_trainX, feature6_trainY, feature1_testX, feature1_testY, feature2_testX, feature2_testY, feature3_testX, feature3_testY, feature4_testX, feature4_testY, feature5_testX, feature5_testY, feature6_testX, feature6_testY

# reshape input to be [samples, time steps, features]
def reshape_input(trainX, testX, feature1_trainX, feature2_trainX, feature3_trainX, feature4_trainX, feature5_trainX, feature6_trainX, feature1_testX, feature2_testX, feature3_testX, feature4_testX, feature5_testX, feature6_testX):
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    feature1_trainX = np.reshape(feature1_trainX, (feature1_trainX.shape[0], 1, feature1_trainX.shape[1]))
    feature1_testX = np.reshape(feature1_testX, (feature1_testX.shape[0], 1, feature1_testX.shape[1]))
    feature2_trainX = np.reshape(feature2_trainX, (feature2_trainX.shape[0], 1, feature2_trainX.shape[1]))
    feature2_testX = np.reshape(feature2_testX, (feature2_testX.shape[0], 1, feature2_testX.shape[1]))
    feature3_trainX = np.reshape(feature3_trainX, (feature3_trainX.shape[0], 1, feature3_trainX.shape[1]))
    feature3_testX = np.reshape(feature3_testX, (feature3_testX.shape[0], 1, feature3_testX.shape[1]))
    feature4_trainX = np.reshape(feature4_trainX, (feature4_trainX.shape[0], 1, feature4_trainX.shape[1]))
    feature4_testX = np.reshape(feature4_testX, (feature4_testX.shape[0], 1, feature4_testX.shape[1]))
    feature5_trainX = np.reshape(feature5_trainX, (feature5_trainX.shape[0], 1, feature5_trainX.shape[1]))
    feature5_testX = np.reshape(feature5_testX, (feature5_testX.shape[0], 1, feature5_testX.shape[1]))
    feature6_trainX = np.reshape(feature6_trainX, (feature6_trainX.shape[0], 1, feature6_trainX.shape[1]))
    feature6_testX = np.reshape(feature6_testX, (feature6_testX.shape[0], 1, feature6_testX.shape[1]))
    return trainX, testX, feature1_trainX, feature2_trainX, feature3_trainX, feature4_trainX, feature5_trainX, feature6_trainX, feature1_testX, feature2_testX, feature3_testX, feature4_testX, feature5_testX, feature6_testX

# concatenate input elements
def concat_input(trainX, testX, feature1_trainX, feature2_trainX, feature3_trainX, feature4_trainX, feature5_trainX, feature6_trainX, feature1_testX, feature2_testX, feature3_testX, feature4_testX, feature5_testX, feature6_testX):
    tr = np.zeros((trainX.shape[0],7,trainX.shape[2]))
    for i in range(len(trainX)):
        tr[i] = np.concatenate([trainX[i], feature1_trainX[i], feature2_trainX[i], feature3_trainX[i], feature4_trainX[i], feature5_trainX[i], feature6_trainX[i]])
    te = np.zeros((testX.shape[0],7,testX.shape[2]))
    for i in range(len(testX)):
        te[i] = np.concatenate([testX[i], feature1_testX[i], feature2_testX[i], feature3_testX[i], feature4_testX[i], feature5_testX[i], feature6_testX[i]])
    return tr, te

# create and fit the baseline model
class BaselineModel():

    def create(look_back = 30):
        baseline = Sequential()
        baseline.add(LSTM(5, input_shape=(7, look_back)))
        baseline.add(Dense(1))
        print(baseline.summary())

"""    def fit(tr, trainY, validation=0.1):
        baseline.compile(loss='mse', optimizer='adam')
        history = baseline.fit(tr, trainY, epochs=2, batch_size=60, verbose=1,validation_split=validation)
        return history

    # Loss History
    def loss_history(history):
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        print(history)
"""


# create and fit the model
class DLModel():

    def create(look_back = 30):
        model = Sequential()
        model.add(LSTM(20,input_shape=(7, look_back), return_sequences=True))
        model.add(Dropout(0.1))
        model.add(Dense(10, input_shape=(7, look_back), activation="sigmoid"))
        model.add(Dropout(0.1))
        model.add(GRU(10, input_shape=(7, look_back)))
        model.add(Dense(1, input_shape=(7, look_back), activation="relu"))
        print(model.summary())
        return model

    def fit(tr, trainY, validation=0.1):
        model.compile(loss='mse', optimizer='adam')
        history = model.fit(tr, trainY, epochs=350, batch_size=60, verbose=1,validation_split=validation)
        return history

    # Loss History
    def loss_history(history):
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        return history

    # make predictions
    def make_predictions(tr, te):
        trainPredict = model.predict(tr)
        testPredict = model.predict(te)
        return trainPredict, testPredict

    # calculate root mean squared error
    def compute_error(trainY, testY, trainPredict, testPredict):
        trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:,0]))
        print('Train Score: %.2f RMSE' % (trainScore))
        testScore = math.sqrt(mean_squared_error(testY, testPredict[:,0]))
        print('Test Score: %.2f RMSE' % (testScore))
        return trainScore, testScore

    # invert predictions
    def invert_predictions(trainY, testY, trainPredict, testPredict):
        trainPredict = scaler1.inverse_transform(trainPredict)
        trainY = scaler1.inverse_transform([trainY])
        testPredict = scaler1.inverse_transform(testPredict)
        testY = scaler1.inverse_transform([testY])
        return trainY, testY, trainPredict, testPredict

    # calculate root mean squared error
    def compute_rmse(trainY, testY, trainPredict, testPredict):
        trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
        print('Train Score: %.2f RMSE' % (trainScore))
        testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
        print('Test Score: %.2f RMSE' % (testScore))
        return trainScore, testScore

    def shift_predictions(data, trainPredict, look_back = 30):
        # shift train predictions for plotting
        trainPredictPlot = np.empty_like(data)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[look_back+1:len(trainPredict)+look_back+1, :] = trainPredict

        # shift test predictions for plotting
        testPredictPlot = np.empty_like(data)
        testPredictPlot[:,:] = np.nan
        testPredictPlot[len(trainPredict)+(look_back*2)+1:len(data)-1,:] = testPredict
        return trainPredictPlot, testPredictPlot

    def plot_predictions(scaler1, data, trainPredictPlot, testPredictPlot, testPredict, test_size, look_back = 30):
        # plot predictions
        plt.plot(scaler1.inverse_transform(data))
        plt.plot(trainPredictPlot)
        print('testVol:')
        testVol = scaler1.inverse_transform(data[test_size+look_back:])
        print(testVol)
        print('testPredictions:')
        print(testPredict)
        # plot the actual vol
        plt.plot(testPredictPlot)
        plt.show()
        plt.plot(scaler1.inverse_transform(data).squeeze()[-200:],label='REAL')
        plt.plot(testPredictPlot.squeeze()[-200:],label='Predict')
        plt.legend()
        return testVol




if __name__ == '__main__':
    dataset = load_data('bny','us')
    y, feature1, feature2, feature3, feature4, feature5, feature6 = construct_variables(dataset)
    scaler1, scaler2, scaler3, scaler4, scaler5, scaler6, scaler7, data, feature1, feature2, feature3, feature4, feature5, feature6 = normalize_dataset(y, feature1, feature2, feature3, feature4, feature5, feature6)
    train_size, test_size, trainX, trainY, testX, testY, feature1_trainX, feature1_trainY, feature2_trainX, feature2_trainY, feature3_trainX, feature3_trainY, feature4_trainX, feature4_trainY, feature5_trainX, feature5_trainY, feature6_trainX, feature6_trainY, feature1_testX, feature1_testY, feature2_testX, feature2_testY, feature3_testX, feature3_testY, feature4_testX, feature4_testY, feature5_testX, feature5_testY, feature6_testX, feature6_testY = split_dataset(data, feature1, feature2, feature3, feature4, feature5, feature6, ratio = 0.8, look_back = 30)
    trainX, testX, feature1_trainX, feature2_trainX, feature3_trainX, feature4_trainX, feature5_trainX, feature6_trainX, feature1_testX, feature2_testX, feature3_testX, feature4_testX, feature5_testX, feature6_testX = reshape_input(trainX, testX, feature1_trainX, feature2_trainX, feature3_trainX, feature4_trainX, feature5_trainX, feature6_trainX, feature1_testX, feature2_testX, feature3_testX, feature4_testX, feature5_testX, feature6_testX)
    tr, te = concat_input(trainX, testX, feature1_trainX, feature2_trainX, feature3_trainX, feature4_trainX, feature5_trainX, feature6_trainX, feature1_testX, feature2_testX, feature3_testX, feature4_testX, feature5_testX, feature6_testX)

    #baseline = BaselineModel.create()
    #baseline_history = BaselineModel.fit(tr, trainY)
    #baseline_loss_history = BaselineModel.loss_history(baseline_history)

    model = DLModel.create()
    history = DLModel.fit(tr, trainY)
    loss_history = DLModel.loss_history(history)
    trainPredict, testPredict = DLModel.make_predictions(tr, te)
    trainScore, testScore = DLModel.compute_error(trainY, testY, trainPredict, testPredict)
    trainY, testY, trainPredict, testPredict = DLModel.invert_predictions(trainY, testY, trainPredict, testPredict)
    trainScore_, testScore_ = DLModel.compute_rmse(trainY, testY, trainPredict, testPredict)
    trainPredictPlot, testPredictPlot = DLModel.shift_predictions(data, trainPredict)
    testVol = DLModel.plot_predictions(scaler1, data, trainPredictPlot, testPredictPlot, testPredict, test_size)

















































