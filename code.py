import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('bgt.csv',parse_dates = ['dt'])
df1 = pd.read_csv('a.csv',parse_dates=['year'])

#change dt to datetime series
df['year'] = pd.DatetimeIndex(df['dt']).year
dt = df['dt']
df.drop(['dt'],inplace=True,axis=1)
df1.drop([37],inplace=True)

#changing into int 
df['year'] = df['year'].apply(int)
df1['year'] = df1['year'].apply(int)

#merging two dataframes
df3 = df.merge(df1,on = 'year',how='left')
#setting index of df3
df3.set_index(['year'],inplace=True)
#-------------------------------------------------------------

air = df3[['landaveragetemperature','landmaxtemperature','landmintemperature','co2','ch4','n2o','cfc12','cfc11','15_minor','co2_eq_ppm_total','aggi_1990_1']]
air.set_index(dt,inplace=True)
air = air.resample('AS').mean()
air_dependent = air.iloc[:,0].values
#-------------------------------------------------------------
feat = air[['co2','ch4','n2o','cfc12','cfc11','15_minor','co2_eq_ppm_total','aggi_1990_1']]
feat['year'] = feat.index
feat['year'] = pd.DatetimeIndex(feat['year']).year
#-------------------------------------------------------------
X = feat.iloc[:,8].values
y = feat.iloc[:,:-1].values

fut_date = np.array([2016,2017,2018,2019,2020,2021,2022,2023,2024,2025,2026,2027,2028,2029,2030,2031,2032,2033,2034,2035,2036,2037,2038,2039,2040])
fut_date = fut_date.reshape(-1,1)
"""
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
X_train = X_train.reshape(-1,1)
X_test = X_test.reshape(-1,1)
"""
X = X.reshape(-1,1)
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X,y)
fut_values = reg.predict(fut_date)

#-------------------------------------------------------------
current_inde = feat.iloc[:,:-1].values
current_dep = air.iloc[:,0].values
from sklearn.linear_model import LinearRegression   
reg = LinearRegression()
reg.fit(current_inde,current_dep)
ans = fut_temp = reg.predict(fut_values)












