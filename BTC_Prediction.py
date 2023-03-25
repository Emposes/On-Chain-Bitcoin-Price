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
 
# Read data
df = pd.read_csv("local_path/Merged_df.csv")
df = df.set_index("timestamp")

# Data Preprocessing
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

''' To select the best model, we will use the following parameters:
    1. Time steps
    2. Moving average
    3. Correlation threshold
    4. Layer size
    5. Dropout rate
    6. Learning rate
    
    We will use the following metrics to evaluate the model:
    1. RMSE
    2. MSE
    3. MAE
    
    We will use the following model:
    1. LSTM
    
    We will use the following optimizer:
    1. Adam
    
    We will use the following loss function:
    1. Mean Squared Error
    
    We will use the following activation function:
    1. Linear
    
    We will use the following epochs:
    1. 100

    The loop will run for 6 times, each time with a different time step.

    The loop will run for 30 times, each time with a different moving average.

    The loop will run for 12 times, each time with a different correlation threshold.

    The loop will run for 9 times, each time with a different layer size.

    The loop will run for 5 times, each time with a different dropout rate.

    The loop will run for 3 times, each time with a different learning rate.

    In total, the loop will run for 6*30*12*9*5*3 = 291.600 times.

    The loop will take approximately 2 days to run.

    The aim is to find the best model with the lowest RMSE, MSE, and MAE.

    All the results will be saved in a csv file.
    '''

# The first for loop is for time steps, iterating from 1 to 14, with irregular steps
for step in [1,3,5,7,10,14]:

    # The second for loop is for moving average, iterating from 1 to 30
    for moving_average in range(1,30):
        df_norm_copy = df_norm.copy()
        df_norm_copy["Price"] = df_norm_copy["Price"].rolling(window=moving_average).mean()
        df_norm_copy = df_norm_copy.iloc[moving_average-1: , :]
        df_norm_copy = df_norm_copy.reset_index(drop=True)

        # The third for loop is for correlation threshold, iterating from 0.95 to 0.4, decreasing by 0.05
        for i_corr in [0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4]:
            
            to_drop = [column for column in corr.columns if abs(corr[column]["Price"]) <= i_corr]
            
            principalDf = df_norm_copy.drop(to_drop, axis=1)
            principalDf = principalDf.loc[:, ~principalDf.columns.isin(['Price'])]

            # corrmat = df_norm_copy.corr()
            # f, ax = plt.subplots(figsize=(30, 30))
            # sns.heatmap(corrmat, vmax = 1, annot = True, fmt = ".2f",square=True);
            # plt.show()
            
            train_size = int(len(principalDf) * 0.8)
            test_size = len(principalDf) - train_size
            data_train, data_test = principalDf[0:train_size], principalDf[train_size:len(principalDf)]
            # df_train_y, df_test_y= df_norm_copy.Price[0:train_size], df_norm_copy.Price[train_size:len(df_norm_copy.Price)]
            
            RMSE =[]
            MSE = []
            MAE = []
            Parameters = []  

            hops=step
            X_train = []
            y_train = []
            
            for i in range(hops,train_size):
                X_train.append(data_train[i-hops:i])
                y_train.append(df_norm_copy.Price[i])
                
            X_train, y_train = np.array(X_train), np.array(y_train)
            
            X_test = []
            y_test = []
            for i in range(hops,test_size):
                X_test.append(data_test[i-hops:i])
                y_test.append(df_norm_copy.Price[i+train_size])
            
            X_test, y_test = np.array(X_test), np.array(y_test)
            
            # The fourth for loop is for layer size, iterating from 32 to 3000
            for i_unit in [32,64,128,256,512,1024,2000,2500,3000]:

                # The fifth for loop is for dropout rate, iterating from 0.1 to 0.8
                for i_drop in [0.1,0.2,0.4,0.6,0.8]:

                    # The sixth for loop is for learning rate, iterating from 0.01 to 0.0001, decreasing by an order of magnitude
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
                        model.fit(X_train, y_train, epochs=100, batch_size=512)
                        
                        y_pred = model.predict(X_test)
                        
                        y_pred = scaler_y.inverse_transform(y_pred)
                        y_test = df.Price[train_size+1:,].values
                        
                        # plt.plot(y_test[hops:], label = "actual" , color = "red")
                        # plt.plot(y_pred, label = "predicted", color="blue")
                        # plt.legend()
                        # plt.show()
                        
                        # The performance of the model is measured by RMSE, MSE, and MAE and the parameters are saved in a list
                        RMSE.append(np.sqrt(mean_squared_error(y_pred,y_test[hops:], squared=False)))
                        MSE.append(np.sqrt(mean_squared_error(y_pred,y_test[hops:])))
                        MAE.append(np.sqrt(mean_absolute_error(y_pred,y_test[hops:])))
                        Parameters.append(str(i_unit)+" & "+str(i_drop)+ " & "+str(i_learning))
                    
        # The results are saved in a csv file
        error_df = pd.DataFrame([RMSE,MSE,MAE,Parameters]).transpose()
        error_df.columns=["RMSE","MSE","MAE","Parameters"]
        error_df.to_csv('local_path/results_corr_'+str(i_corr)+'.csv')