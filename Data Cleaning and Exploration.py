import pandas as pd
import xlrd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt


sns.set(color_codes=True)

#Load in dataset
df = pd.read_excel (r'Bitcoin Dataset.xlsx')

#Calculate NA values
df.isnull().sum().sum()
df.isnull().sum()

#Fill NA with previous value
dfFill = df.fillna(method ='bfill')
dfFill.isnull().sum()

#Check data types
dfFill.dtypes

#Reverse the order of dataset
cleanData = dfFill.reindex(index=dfFill.index[::-1])

cleanData = cleanData.loc[~(cleanData['BTC Price'] == 0)]

#Clean column names
cleanData.columns = cleanData.columns.str.replace(' ', '')
cleanData.columns = cleanData.columns.str.replace('-', '')

cleanDf = cleanData
cleanDf.isnull().sum()

#save to new excel
cleanDf.to_csv("cleanDf.csv")

cryptoDf = cleanDf[['BTCPrice','BTCnetworkhashrate','AverageBTCblocksize','NUAUBTC','NumberTXBTC','DifficultyBTC','TXfeesBTC','EstimatedTXVolumeUSDBTC']].copy()

#Heatmaps
plt.figure(figsize=(20,10))
plt.rc('xtick', labelsize=11)
plt.rc('ytick', labelsize=11)
plt.rcParams.update({'font.size': 20})
c= cryptoDf.corr()
sns.heatmap(c,annot=True)
c

priceDf = cleanDf[['BTCPrice','GoldinUSD','EthereumPrice','LitecoinPrice','BitcoinCashPrice','CardanoPrice','Nasdaqcompositeindex','DJI']].copy()
plt.figure(figsize=(20,10))
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rcParams.update({'font.size': 20})
c= priceDf.corr()
sns.heatmap(c,annot=True)
c

#Line graph
plt.figure(figsize = (18, 10))
plt.plot(cleanDf["date"], cleanDf["BTCPrice"], color='goldenrod', lw=2)
plt.title("Bitcoin Price History", size=25)
plt.xlabel("Time", size=20)
plt.ylabel("BTC Price (USD)", size=20)
plt.show()

#Scatter plot
plt.scatter(priceDf['BTCPrice'], priceDf['LitecoinPrice'], s = 4)
plt.xlabel("Litecoin Price")
plt.ylabel("BTC Price")
plt.show()

cleanDf2 = cleanDf.copy()
cleanDf2.set_index('date', inplace=True)
cleanDf2 = cleanDf[cleanDf.index.date  >= dt.date(2017,6,1)]

#Line graph
plt.plot(cleanDf2["BTCPrice"], color='goldenrod', lw=1, label= "BTC")
plt.plot(cleanDf2["DJI"], color='blue', lw=1, label= "DJI")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()





