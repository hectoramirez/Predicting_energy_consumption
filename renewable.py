import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as MSE
from pandas import DataFrame, concat
from numpy import concatenate
from math import sqrt
from tqdm import tqdm

# ===================================================
desired_width = 320
pd.set_option('display.width', desired_width)  # Show columns horizontally in console
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 100)  # Show as many columns as I want in console
pd.set_option('display.max_rows', 1000)  # Show as many rows as I want in console
# ===================================================

files_path = "/Users/hramirez/GitHub/Renewable_Energy/files/"

Meteo = pd.read_csv(files_path + "Meteo.csv", parse_dates=[['Date', 'Time']])
datos_solar = pd.read_excel(files_path + "Datos_solar_y_demanda_residencial.xlsx")


# Get to a common timing for both datasets:

def round_to_5min(t):
    delta = datetime.timedelta(minutes=t.minute % 5,
                               seconds=t.second,
                               microseconds=t.microsecond)
    t -= delta
    if delta > datetime.timedelta(0):
        t += datetime.timedelta(minutes=5)
    return t


Meteo['DateRound'] = Meteo["Date_Time"].dt.round("5min")
datos_solar['DateRound'] = datos_solar['Date'].dt.round("5min")

Meteo['Hour'] = Meteo['DateRound'].dt.hour
Meteo['Day'] = Meteo['DateRound'].dt.dayofyear


# Clean Meteo Frame from characters and columns that do not give information in this case: UV, Solar and Wind**

def clean(x):
    try:
        return x.str.replace(r"[a-zA-Z\%\/²]", '')
    except:
        return x


Meteo = Meteo.apply(lambda x: clean(x))
Meteo = Meteo.drop(columns=['UV', 'Solar', 'Wind'])

'''
Meteo.Temperature = pd.to_numeric(Meteo.Temperature)
datos_solar.plot(x='DateRound', y='Demanda (W)', figsize=(15,5))
Meteo.plot(x='DateRound', y='Temperature', figsize=(15,5)) 

plt.figure(figsize=(15,10))
sns.distplot(datos_solar['Demanda (W)'], color='r').set_title('Demanda')
plt.show()
'''

# ===================================================

# Select relevant features for the final Meteo dataframe
# From solar dataframe selection only 'Demanda (W)'

list_features = ['Date_Time', 'Temperature', 'Dew Point', 'Humidity', 'Speed', 'Gust',
                 'Pressure', 'Precip. Rate.', 'Precip. Accum.', 'DateRound', 'Hour',
                 'Day']

Meteo_features = Meteo[list_features]
solar_demanda = datos_solar[['DateRound', 'Demanda (W)']]

# Merge both pandas based on datetime

df_demanda = pd.merge(Meteo_features, solar_demanda, on='DateRound')
# remove date from df
del df_demanda['DateRound']
# print(df_demanda.head())

# Make plots with all the values for all variables

'''
cols = df_demanda.columns.drop(['Date_Time', 'Hour', 'Day'])
df_demanda[cols] = df_demanda[cols].apply(pd.to_numeric)

for i in range(len(cols)):
    df_demanda.plot(x='Date_Time', y=df_demanda[cols].columns[[i]], figsize=(15,5))
'''


# ===================================================

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = [], []
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


values = df_demanda.drop(['Date_Time', 'Hour', 'Day'], axis=1).values
# values = df_demanda[['Temperature', 'Humidity', 'Hour', 'Demanda (W)']].values

n_features = values.shape[1]
n_hours = 3
n_out = 1

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# specify the number of lag hours
# frame as supervised learning
lagged = series_to_supervised(scaled[:, :], n_hours, n_out, True)  # .reset_index(drop=True)

reframed = pd.concat([lagged.iloc[:, :-1], pd.Series(scaled[:, -1], name='Demanda')], axis=1).dropna()
reframed.head()

print(reframed.shape)

# split into train and test sets
values = reframed.values
n_train_hours = int(reframed.shape[0] * (2 / 3))
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]

n_obs = n_hours * n_features

# split into input and outputs
X_train, y_train = train[:, :n_obs], train[:, -1]
X_test, y_test = test[:, :n_obs], test[:, -1]
print(X_train.shape, y_train.shape)

# reshape input to be 3D [samples, timesteps, features]
X_train_in = X_train.reshape((X_train.shape[0], n_hours, n_features))
X_test_in = X_test.reshape((X_test.shape[0], n_hours, n_features))
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(X_train_in.shape[1], X_train_in.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

# Run network
history = model.fit(X_train_in, y_train, epochs=50, batch_size=72,
                    validation_data=(X_test_in, y_test), verbose=2, shuffle=False)

y_pred_out = model.predict(X_test_in)

# ===================================================

# invert scaling for forecast
y_pred = concatenate((X_test[:, -n_features:-1], y_pred_out), axis=1)
# _=pd.DataFrame(inv_yhat)
# _.tail()
y_pred = scaler.inverse_transform(y_pred)[:, -1]

# invert scaling for actual
y_test = y_test.reshape(y_test.shape[0], 1)
y_inv = concatenate((X_test[:, -n_features:-1], y_test), axis=1)
y_inv = scaler.inverse_transform(y_inv)[:, -1]

# ===================================================

# calculate RMSE
rmse = sqrt(MSE(y_inv, y_pred))
print('Test RMSE: %.3f' % rmse)

# plot history
plt.figure(figsize=(15, 10))
plt.plot(y_inv, label='data')
plt.plot(y_pred, label='prediction')
plt.legend()
plt.show()
# plt.savefig('predvsdata2.pdf')

# ===================================================
# ===================================================


def lstm_net(X_train_in, y_train, X_test_in, y_test, X_test):
    # design network
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train_in.shape[1], X_train_in.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')

    # Run network
    history = model.fit(X_train_in, y_train, epochs=50, batch_size=72,
                        validation_data=(X_test_in, y_test), verbose=0, shuffle=False)

    y_pred_out = model.predict(X_test_in)

    # invert scaling for forecast
    y_pred = concatenate((X_test[:, -n_features:-1], y_pred_out), axis=1)
    # _=pd.DataFrame(inv_yhat)
    # _.tail()
    y_pred = scaler.inverse_transform(y_pred)[:, -1]

    # invert scaling for actual
    y_test = y_test.reshape(y_test.shape[0], 1)
    y_inv = concatenate((X_test[:, -n_features:-1], y_test), axis=1)
    y_inv = scaler.inverse_transform(y_inv)[:, -1]

    # calculate RMSE
    from math import sqrt

    rmse = sqrt(MSE(y_inv, y_pred))
    # print('Test RMSE: %.3f' % rmse)
    return rmse


rmse_l = []
for i in tqdm(range(30)):
    tqdm._instances.clear()
    rmse = lstm_net(X_train_in, y_train, X_test_in, y_test, X_test)
    # print(rmse)
    rmse_l.append(rmse)

plt.figure()
pd.DataFrame(rmse_l, columns=['RMSE']).boxplot()

