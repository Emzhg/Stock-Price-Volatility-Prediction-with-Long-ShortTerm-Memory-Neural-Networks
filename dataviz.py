import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from arch.__future__ import reindexing
from arch.univariate import GARCH
from arch import arch_model


class Get():
    def data(ticker='vivo', region='us'):
        data = pd.read_csv(f'archive/Stocks/{ticker}.{region}.txt', sep=",", header=None)
        data.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'openint']
        data.drop(index=data.index[0], axis=0, inplace=True)
        data = data.set_index(['date'])
        print(data.head())
        print(data.shape)

        return data

    def features(data):
        features = pd.DataFrame()
        # open
        features['open'] = data['open'].astype(float)
        # high
        features['high'] = data['high'].astype(float)
        # low
        features['low'] = data['low'].astype(float)
        # close
        features['close'] = data['close'].astype(float)
        # volume
        features['volume'] = data['volume'].astype(float)
        # log return
        features['log_return'] = np.log(features['close']).diff()
        # log volume change
        features['log_volume_chg'] = np.log(features['volume']).diff()
        # log trading range
        features['log_trading_range'] = np.log(features['high'] - features['low']).diff()
        # past 10 days vol
        features['vol_10'] = pd.Series(features['log_return']).rolling(window=10).std() * np.sqrt(5)
        # past 30 days vol
        features['vol_30'] = pd.Series(features['log_return']).rolling(window=30).std() * np.sqrt(30)
        # GARCH forward looking 10-ady vol
        features['garch_vol_10'] = 0
        features = features.fillna(0)
        return features

    def plots(data):
        # Visualize the data
        plt.figure(figsize=(40, 20))
        plt.plot(data['close'])
        # plt.plot(data['close'].tail(200), label='0')
        # plt.plot(data_10['close'].tail(200), label='10')
        # plt.plot(data_20['close'].tail(200), label='20')
        # plt.plot(data_30['close'].tail(200), label='30')
        # plt.plot(data_40['close'].tail(200), label='40')
        plt.title('Apple Close Price History')
        plt.xlabel('1984-09-07 to 2017-11-10')
        plt.ylabel('Close Price USD($)')
        plt.legend(loc='upper left')
        plt.show()


class garch():

    def autocorrelation(data):
        plot_acf(data['log_return'])
        plt.show()

    def split(data):
        # split into train/test
        size = 0.8
        y = data.garch_vol_10
        x = data.drop('garch_vol_10', axis=1)
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)
        return xtrain, xtest, ytrain, ytest

    def model(data):
        rolling_predictions = []
        size = len(data.index)
        test_size = int(size * 0.8)
        for i in range(test_size):
            train = data['log_return'][:-(test_size - i)]
            garch = arch_model(train, p=1, q=1, mean='Zero', vol='ARCH')
            garch_fit = garch.fit()
            forecast = garch_fit.forecast(horizon=10)
            rolling_predictions.append(np.sqrt(forecast.variance.values[-1, :][-1]))
        rolling_predictions = pd.Series(rolling_predictions, index=data['log_return'].index[-test_size:])
        data['garch_vol_10'] = rolling_predictions
        return data


if __name__ == '__main__':
    data = Get.data(ticker='vivo', region='us')
    features = Get.features(data)
    # xtrain, xtest, ytrain, ytest = garch.split(features)
    forecast = garch.model(features)
    csv = forecast.to_csv(r'datasets/dataset_vivo_us.csv', header=True)
    plot = Get.plots(data)
