#### 0.   INCLUDES  _______________________________________ #### 
#Load Libraries:# 
import pandas as pd
import os
from pandas_profiling import ProfileReport



#### 1.   PREPARING DATA _______________________________________ #### 
data = pd.read_csv("../data/train_values.csv")
profile = ProfileReport(data, title="Profiling Report")
profile.to_file("Profiling Report.html")


