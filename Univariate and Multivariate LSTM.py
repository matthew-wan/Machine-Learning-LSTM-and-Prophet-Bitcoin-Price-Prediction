##################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn import preprocessing

from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import tensorflow as tf
from keras import metrics

cleanDf = pd.read_csv (r'cleanDf.csv')

#Univariate and multivariate LSTM
#Prepare the dataset
#Comment one of the datasets out depending on whether it is univariate or multivariate
mlData2 = cleanDf[['date', 'BTCPrice', 'DJI', 'LitecoinPrice', 'EstimatedTXVolumeUSDBTC']]
mlData2 = cleanDf[['date', 'BTCPrice']]
mlData2.set_index('date', inplace=True)

#Normalise the dataset
x = mlData2.values 
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
mlData2=pd.DataFrame(x_scaled, columns=mlData2.columns , index = mlData2.index)
mlData2

#Create training and test data
trainProportion = 0.8
train_size = int(len(mlData2) * trainProportion)
test_size = len(mlData2) - train_size
train = mlData2.iloc[0:train_size]
test = mlData2.iloc[train_size:len(mlData2)]
print(len(train), len(test))

def createTestTrain(x, y, historyLength=1):
    x1, y1 = [], []
    for i in range(len(x) - historyLength):
        v = x.iloc[i:(i + historyLength)].values
        x1.append(v)
        y1.append(y.iloc[i + historyLength])
    return np.array(x1), np.array(y1)

#Number of previous days to use
historyLength = 60
#Create training and test set (number of values,feature target, number of days)
xtrain, ytrain = createTestTrain(train, train.BTCPrice, historyLength)
xtest, ytest = createTestTrain(test, test.BTCPrice, historyLength)
print(xtrain.shape, ytrain.shape)


#LSTM model and parameters
model = Sequential()
model.add(LSTM(units=130,input_shape=(xtrain.shape[1], xtrain.shape[2])))
model.add(Dropout(rate=0.2))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=[metrics.mean_squared_error, 
                       metrics.mean_absolute_error, 
                       metrics.mean_absolute_percentage_error])
#Store the model fits into different variables
history = model.fit(xtrain, ytrain, epochs=30, batch_size=32,validation_split=0.1, 
    verbose=1, shuffle = False)
history2 = model.fit(xtrain, ytrain, epochs=30, batch_size=32,validation_split=0.1, 
    verbose=1, shuffle = False)

plt.plot(history.history['val_mean_absolute_percentage_error'], label='Multivariate Model')
plt.plot(history2.history['val_mean_absolute_percentage_error'], label='Univariate Model')
plt.ylabel('MAPE')
plt.xlabel('Time Step')
plt.legend(loc='upper right')
plt.show();
#Plot the loss and validation loss for train and test data
plt.plot(history.history['loss'], label='Train Data')
plt.plot(history.history['val_loss'], label='Test Data')
plt.legend(loc='upper right')
plt.show()

#Make the predictions
yprediction = model.predict(xtest)

#Plot the training prices with the actual and predicted price
plt.plot(np.arange(0, len(ytrain)), ytrain, 'g', label="History")
plt.plot(np.arange(len(ytrain), len(ytrain) + len(ytest)), ytest, marker='.', label="Actual")
plt.plot(np.arange(len(ytrain), len(ytrain) + len(ytest)), yprediction, 'r', label="Predicted")
plt.ylabel('BTC Price')
plt.xlabel('Time Step')
plt.legend(loc='upper right')
plt.show();

#Plot the predicted price with the actual price
plt.plot(ytest, marker='.', label="Actual")
plt.plot(yprediction, 'r', label="Predicted")
plt.ylabel('BTC Price')
plt.xlabel('Time Step')
plt.legend(loc='upper right')
plt.show();

