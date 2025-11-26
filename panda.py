import pandas as pd
import numpy as np
import math as mt
#series = pd.Series([1,2,3,4,5], index=['a','b','c','d','e'])
#print(series.loc['a'])
#dates1 = pd.date_range('20250312',periods=6)
#series = pd.Series([1,2,3,4,5,6])
#series.index = dates1
#print(series) 
#df = pd.read_csv()
#df = pd.DataFrame(np.random.randn(4,3), columns=list('ABC'), index=list('abcd'))
#print(df)

f = "/home/am/Documents/AI/anaconda3/envs/ML/img/results/output_rgbf1.csv"
df = pd.read_csv(f)
#print(df.tail(2))
#print(df[(df.R>250) & (df.B>200)])
#print(df.sort_values('R', axis=0))
#print(df.sort_values(5,axis=1))
#for column in df:
    #df[column] = df[column].apply()

#print(df.drop(df.index[3:65536]))
#print(df.drop(df.columns[2],axis=1))
#print(pd.crosstab(df.R,df.G))