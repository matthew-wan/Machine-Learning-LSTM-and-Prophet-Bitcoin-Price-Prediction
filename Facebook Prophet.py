#############################################################################################
import pandas as pd
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric


cleanDf = pd.read_csv (r'cleanDf.csv')

#Prepare data for Prophet. Only requires date and BTC price
fbData = cleanDf[['BTCPrice','date']]
fbData = fbData.rename(columns={"BTCPrice": "y", "date": "ds"})
fbData.reset_index(drop=True, inplace=True)

train_size = int(len(fbData) * 0.8)
test_size = len(fbData) - train_size
fbtrain = fbData.iloc[0:train_size]
fbtest = fbData.iloc[train_size:len(fbData)]
print(len(fbtrain), len(fbtest))

#Modelling for Prophet
m = Prophet(interval_width= 0.95, daily_seasonality=False)
model = m.fit(fbData)
future = m.make_future_dataframe(periods=180, freq='D')
forecast = m.predict(future)
forecast.head()


#Plot  the predictions and components
fbPlot = m.plot(forecast)
fbPlot2 = m.plot_components(forecast)

m.plot(forecast)
ax=forecast.plot(x='ds',y='yhat',legend=True,label='predictions',figsize=(12,8))
fbtest.plot(x='ds',y='y',legend=True,label='Test Data',ax=ax,xlim=('2017-05-29','2018-06-05'))


df_cv = cross_validation(m, initial='90 days', period='180 days', horizon = '180 days')
df_cv.head()

df_p = performance_metrics(df_cv)
df_p.head()
df_p.tail()

#Change the metric for different plots
plot_cross_validation_metric(df_cv, metric='mape')
plot_cross_validation_metric(df_cv,'rmse');
