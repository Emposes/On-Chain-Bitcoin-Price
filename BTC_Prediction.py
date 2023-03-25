import pandas as pd
import numpy as np
from scipy.stats import zscore
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
from tensorflow.keras.utils import plot_model

# Set seed for reproducibility
np.random.seed(0)

df = pd.read_csv("local_path/Merged_df.csv")
df = df.set_index("timestamp")

scaler = StandardScaler()
df_norm = df.loc[:, ~df.columns.isin(['Price'])]
df_norm = pd.DataFrame(scaler.fit_transform(df.loc[:, ~df.columns.isin(['Price'])]), columns=df_norm.columns)
scaler_y = StandardScaler()
df_train_y = pd.DataFrame(scaler_y.fit_transform(df[["Price"]]), columns=["Price"])

# df_norm = zscore(df.loc[:, df.columns != 'timestamp'])
df_norm = pd.concat([df_norm,df_train_y], axis=1)

# sns.distplot(tuple(df_norm['Price']))
# plt.show()

# Stationarity test
for i in df_norm.columns:
    while(adfuller(df_norm[i].dropna(),autolag='AIC')[1]>0.05):
        df_norm[i] = df_norm[i].diff()
        
df_norm = df_norm.iloc[1: , :]

# Correlation Selection
corr = df_norm.corr(method='pearson')
corr.sort_values(["Price"], ascending = False, inplace = True)

for i_corr in [0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4]:
    
    to_drop = [column for column in corr.columns if abs(corr[column]["Price"]) <= i_corr]
    
    principalDf = df_norm.drop(to_drop, axis=1)
    principalDf = principalDf.loc[:, ~principalDf.columns.isin(['Price'])]

    # corrmat = df_norm.corr()
    # f, ax = plt.subplots(figsize=(30, 30))
    # sns.heatmap(corrmat, vmax = 1, annot = True, fmt = ".2f",square=True);
    # plt.show()
    
    train_size = int(len(principalDf) * 0.8)
    test_size = len(principalDf) - train_size
    data_train, data_test = principalDf[0:train_size], principalDf[train_size:len(principalDf)]
    # df_train_y, df_test_y= df_norm.Price[0:train_size], df_norm.Price[train_size:len(df_norm.Price)]
    
    RMSE =[]
    MSE = []
    MAE = []
    Parameters = []
    for step in [1,3,5,7,10,14]:
        
        hops=step
        X_train = []
        y_train = []
        
        for i in range(hops,train_size):
            X_train.append(data_train[i-hops:i])
            y_train.append(df_norm.Price[i])
            
        X_train, y_train = np.array(X_train), np.array(y_train)
        
        X_test = []
        y_test = []
        for i in range(hops,test_size):
            X_test.append(data_test[i-hops:i])
            y_test.append(df_norm.Price[i+train_size])
        
        X_test, y_test = np.array(X_test), np.array(y_test)
        
        for i_unit in [32,64,128,256,512,1024,2000,2500,3000]:
            for i_drop in [0.1,0.2,0.4,0.6,0.8]:
                for i_learning in [0.01,0.001,0.0001]:
                    model = Sequential()
                    model.add(LSTM(units=i_unit, return_sequences=True, input_shape=(hops,len(principalDf.columns))))
                    model.add(Dropout(i_drop))
                    model.add(LSTM(units=i_unit))
                    model.add(Dropout(i_drop))
                    model.add(Dense(1))
                    model.compile(optimizer="adam", loss="mean_squared_error",learning_rate=i_learning)
                    model.summary()
                    # plot_model(model)
                    model.fit(X_train, y_train, epochs=50, batch_size=512)
                    
                    y_pred = model.predict(X_test)
                    
                    y_pred = scaler_y.inverse_transform(y_pred)
                    y_test = df.Price[train_size+1:,].values
                    
                    # plt.plot(y_test[hops:], label = "actual" , color = "red")
                    # plt.plot(y_pred, label = "predicted", color="blue")
                    # plt.legend()
                    # plt.show()
                    
                    RMSE.append(np.sqrt(mean_squared_error(y_pred,y_test[hops:], squared=False)))
                    MSE.append(np.sqrt(mean_squared_error(y_pred,y_test[hops:])))
                    MAE.append(np.sqrt(mean_absolute_error(y_pred,y_test[hops:])))
                    Parameters.append(str(i_unit)+" & "+str(i_drop)+ " & "+str(i_learning))
                
    error_df = pd.DataFrame([RMSE,MSE,MAE,Parameters]).transpose()
    error_df.columns=["RMSE","MSE","MAE","Parameters"]
    error_df.to_csv('local_path/results_corr_'+str(i_corr)+'.csv')