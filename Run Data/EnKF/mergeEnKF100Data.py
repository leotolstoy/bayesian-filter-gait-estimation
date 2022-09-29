import numpy as np
import pandas as pd

filenames = ['xsubject_data_enkf_N100_1-10fp.csv','xsubject_data_enkf_N100_10st.csv']
df = pd.concat(map(pd.read_csv, filenames), ignore_index=True)
print(df.head())
df.to_csv('xsubject_data_enkf_N100.csv',header=True,index=False)


filenames = ['xsubject_timing_data_enkf_N100_1-10fp.csv','xsubject_timing_data_enkf_N100_10st.csv']
df = pd.concat(map(pd.read_csv, filenames), ignore_index=True)
print(df.head())
df.to_csv('xsubject_timing_data_enkf_N100.csv',header=True,index=False)




