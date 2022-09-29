import numpy as np
import pandas as pd

filenames = ['xsubject_data_ukf_1-8p.csv','xsubject_data_ukf_8st.csv','xsubject_data_ukf_9-10.csv']
df = pd.concat(map(pd.read_csv, filenames), ignore_index=True)
print(df.head())
df.to_csv('xsubject_data_ukf.csv',header=True,index=False)


filenames = ['xsubject_timing_data_ukf_1-8p.csv','xsubject_timing_data_ukf_8st.csv','xsubject_timing_data_ukf_9-10.csv']
df = pd.concat(map(pd.read_csv, filenames), ignore_index=True)
print(df.head())
df.to_csv('xsubject_timing_data_ukf.csv',header=True,index=False)




