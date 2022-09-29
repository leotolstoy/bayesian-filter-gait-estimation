import numpy as np
import pandas as pd

filenames = ['xsubject_data_ekf_1-6.csv','xsubject_data_ekf_7-10.csv']
df = pd.concat(map(pd.read_csv, filenames), ignore_index=True)
print(df.head())
df.to_csv('xsubject_data_ekf.csv',header=True,index=False)


filenames = ['xsubject_timing_data_ekf_1-6.csv','xsubject_timing_data_ekf_7-10.csv']
df = pd.concat(map(pd.read_csv, filenames), ignore_index=True)
print(df.head())
df.to_csv('xsubject_timing_data_ekf.csv',header=True,index=False)




