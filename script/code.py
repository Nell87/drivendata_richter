# 0.   INCLUDES ---
# Load Libraries:# 
import pandas as pd
import time   #  provides many ways of representing time in code
import warnings
import os

import matplotlib.pyplot as plt
import seaborn as sns
from pylab import *

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE

from sklearn.metrics import confusion_matrix
from pretty_confusion_matrix import pp_matrix_from_data
from sklearn.metrics import f1_score   # the score used in the competition

# Reading data _______________________________________________________ #### 
train_values = data = pd.read_csv("https://raw.githubusercontent.com/Nell87/drivendata_richter/main/data/train_values.csv",index_col='building_id')
train_labels = pd.read_csv("https://raw.githubusercontent.com/Nell87/drivendata_richter/main/data/train_labels.csv",index_col='building_id')
test_values = pd.read_csv("https://raw.githubusercontent.com/Nell87/drivendata_richter/main/data/test_values.csv",index_col='building_id')
train_merge = train_values.merge(train_labels, on = 'building_id', how = 'inner',)
print(train_merge.shape)

####    1. ANALYSIS ______________________________________________________ 

# General info
train_merge.info()

##### 1.1.Manual EDA: Target feature or damage_grade ____________________ ####
train_merge['damage_grade'].value_counts(normalize=True) * 100     # High 33.46, Medium 56.90, Low 9.64

# Add labels
damage_map = {1:"Low", 2:"Medium", 3:"High"}
train_merge["damage_grade"] = train_merge["damage_grade"].map(damage_map)
train_labels["damage_grade"] = train_labels["damage_grade"].map(damage_map)

sns.countplot(x="damage_grade", data=train_merge)
plt.title("Damage grade")
plt.ylabel('Frequency')
plt.show()

##### 1.2.Manual EDA: Numerical features ________________________________ ####

# We identify four numerical features:count_families, count_floors_pre_eq, age, 
# area_percentage and height_percentage
warnings.filterwarnings('ignore')
sns.set(rc={"figure.figsize": (12, 8)}); np.random.seed(0)

count_floors_pre_eq = train_merge["count_floors_pre_eq"]
age = train_merge["age"]
area_percentage = train_merge["area_percentage"]
height_percentage = train_merge["height_percentage"]
count_families = train_merge["count_families"]

subplot(2,3,1)
ax = sns.distplot(count_floors_pre_eq)

subplot(2,3,2)
ax = sns.distplot(age)

subplot(2,3,3)
ax = sns.distplot(area_percentage)

subplot(2,3,4)
ax = sns.distplot(height_percentage)

subplot(2,3,5)
ax = sns.distplot(count_families)

plt.show()
###### 1.2.1.Manual EDA: Numerical features - count_families__________________________ ####
# Number of families that live in the building. Having betwen 0 and 3 families 
# living in a building represents ~98% of data, so we'll get rid of buildings 
# with higher number of families.
count_families_data = train_merge[["count_families", "damage_grade"]]
count_families_data["count_families"] = pd.Categorical(count_families_data["count_families"])
count_families_data["count_families"].value_counts(normalize=True)

###### 1.2.2.Manual EDA: Numerical features - count_floors_pre_eq__________________________ ####
# Number of floors in the building before the earthquake. Having 1, 2 or 3 
# floors represent ~97% of data, so we'll get rid of buildings with higher 
# number of them.
count_floors_data = train_merge[["count_floors_pre_eq", "damage_grade"]]
count_floors_data["count_floors_pre_eq"] = pd.Categorical(count_floors_data["count_floors_pre_eq"])
count_floors_data["count_floors_pre_eq"].value_counts(normalize=True)

###### 1.2.3.Manual EDA: Numerical features - age__________________________ ####
# It's the age of the building in years (between 0 and 995 years).
# Due to its distribution, we'll aply the IQR based removal on this feature, 
# achieving a distribution between 0 and 55 years.
def rem_outliers_IQR(data, feature):  
  # Finding the IQR
  percentile25 = data[feature].quantile(0.25)
  percentile75 = data[feature].quantile(0.75)
  iqr = percentile75 - percentile25

  # Finding upper and lower limit
  upper_limit = percentile75 + 1.5 * iqr
  lower_limit = percentile25 - 1.5 * iqr

  # Outliers removal
  data = data[data[feature] < upper_limit]
  data = data[data[feature] > lower_limit]

  return data

test = rem_outliers_IQR(train_merge, 'age')
sns.distplot(test['age'])

###### 1.2.4.Manual EDA: Numerical features - area_percentage__________________________ ####
# It's the normalized area of the building footprint (1-100). Due to its 
# distribution, we'll aply the IQR based removal on this feature, achieving a 
# distribution between 1 and 14.
test = rem_outliers_IQR(train_merge, 'area_percentage')
sns.distplot(test['area_percentage'])


###### 1.2.5.Manual EDA: Numerical features - height_percentage__________________________ ####
# It's the normalized height of the building footprint (2-32). Due to its 
# distribution, we'll aply the IQR based removal on this feature, achieving a 
# distribution between 1 and 8.
test = rem_outliers_IQR(train_merge, 'height_percentage')
sns.distplot(test['height_percentage'])

##### 1.3.Manual EDA: Categorical features ________________________________ ####
###### 1.3.1.Manual EDA:  Categorical features - Location__________________________ ####
# The features geo_level_1_id, geo_level_2_id, geo_level_3_id represent the 
# geographic region in which building exists, from largest (level 1) to most 
# specific sub-region (level 3). Possible values: level 1: 0-30, level 2: 0-1427, 
# level 3: 0-12567.

# For every location feature there is a high number of categorical values, so 
# we'll apply feature engineering on them. We'll replace every value with their 
# conditional probabilities respect to every damage_grade category (check the 
# "feature selection and feature engineering" section)

###### 1.3.2.Manual EDA:  Categorical features - (More than two levels)____ ####
###### 1.3.3.Manual EDA:  Categorical features - (Two levels)______________ ####

####    2. FEATURE ENGINEERING ____________________________________________ #### 
##### 2.1.Feature engineering: Location features __________________________ ####
# For every location feature there is a high number of categorical values, 
# so we'll apply feature engineering on them. We'll replace every value with 
# their conditional probabilities respect to every damage_grade category
# Function to replace a categorical feature with many values, with their conditional probabilities respecto to the predicted feature
def categoricalvalues_condprob(data, index, pred_feature, new_column_name):
  # Create prob table
  probs = data.groupby(index).size().div(len(data))
  probs_group = data.groupby([index, pred_feature]).size().div(len(data)).div(probs, axis=0, level=index).reset_index()
  probs_group.columns= [index, pred_feature, new_column_name]
  probs_group_wide = probs_group.pivot(index=[index], columns = pred_feature,values = new_column_name) #Reshape from long to wide
  probs_group_wide = probs_group_wide.reset_index()
  
 # Rename columns
  unique_values = np.unique(data[pred_feature])
  unique_values = -(len(unique_values))
  for i in range(unique_values,0):
    probs_group_wide.rename(columns={probs_group_wide.columns[i]: index + "_" + str(probs_group_wide.columns[i])}, inplace = True)
    
  # Add column to main dataset
  data_merge = data.merge(probs_group_wide, on=index, how='left')

  # Get rid of the categorical feature
  data_merge = data_merge.drop(index, axis=1)

  # Return dataset
  return data_merge

# Apply the function
train_merge = categoricalvalues_condprob(train_merge, 'geo_level_1_id', 'damage_grade', 'prob_cond_geo_level_1')
train_merge = categoricalvalues_condprob(train_merge, 'geo_level_2_id', 'damage_grade', 'prob_cond_geo_level_2')
train_merge = categoricalvalues_condprob(train_merge, 'geo_level_3_id', 'damage_grade', 'prob_cond_geo_level_3')

# Replace the missing values with 0
cols = ["geo_level_1_id_Low", "geo_level_1_id_Medium", "geo_level_1_id_High",
                         "geo_level_2_id_Low", "geo_level_2_id_Medium", "geo_level_2_id_High",
                         "geo_level_3_id_Low", "geo_level_3_id_Medium", "geo_level_3_id_High"]

train_merge.fillna({"geo_level_1_id_Low":0, "geo_level_1_id_Medium":0, "geo_level_1_id_High":0,
                         "geo_level_2_id_Low":0, "geo_level_2_id_Medium":0, "geo_level_2_id_High":0,
                         "geo_level_3_id_Low":0, "geo_level_3_id_Medium":0, "geo_level_3_id_High":0}, inplace=True)

####    3. FEATURE SELECTION ______________________________________________ #### 
##### 3.1.Feature selection: Random Forest Feature Importance _____________ ####
# Dummify
train_merge_dum = pd.get_dummies(train_merge.drop("damage_grade",1), drop_first = True)

# Oversampling
oversampler=SMOTE(random_state=0)
os_features,os_labels=oversampler.fit_resample(train_merge_dum,train_labels)
train_merge.info()

# Split in train/test
x_train, x_test, y_train, y_test = train_test_split(os_features, os_labels, test_size = 0.2, random_state = 42)

rf_clf_1 = RandomForestClassifier(random_state=314, n_estimators = 100,
                                  max_features = 'auto', max_depth = 10,
                                  min_samples_leaf = 20)

start_time = time.time()
rf_clf_1 = rf_clf_1.fit(x_train, y_train)  
rf_clf_1_time_fit = time.time() - start_time   

# Check the importance
importances = pd.DataFrame(data={
    'Attribute': x_train.columns,
    'Importance': rf_clf_1.feature_importances_
})
importances = importances.sort_values(by='Importance', ascending=False)
importances = importances[1:20]

plt.bar(x=importances['Attribute'], height=importances['Importance'], color='#087E8B')
plt.title('Feature importances', size=20)
plt.xticks(rotation='vertical')
plt.show()

plt.cla()
plt.clf()
plt.close()
##### 3.2.Feature selection: Chi-Squared between Categorical features _____________ ####

##### 3.3.Feature selection: Correlation between numerical features _____________ ####
corr_matrix=train_merge[["count_floors_pre_eq","age", "area_percentage", "height_percentage", "count_families",
                         "geo_level_1_id_1", "geo_level_1_id_2", "geo_level_1_id_2",
                         "geo_level_2_id_1", "geo_level_2_id_2", "geo_level_2_id_3",
                         "geo_level_3_id_1", "geo_level_3_id_2", "geo_level_3_id_3"]].corr()
plt.figure(figsize=(12, 12))                       
sns.heatmap(corr_matrix, annot=True)
plt.show()

####    4. MODELING ______________________________________________ #### 
##### 4.1.Baseline Random Forest _____________ ####
###### 4.1.1.Preprocessing: Dummify and split____ ####

# Dummify
train_values_dum = pd.get_dummies(train_values, drop_first = True)

# Split in train/test
x_train, x_test, y_train, y_test = train_test_split(train_values_dum, train_labels, test_size = 0.2, random_state = 42)

###### 4.1.2.Modeling____________________________________________ ####
# Create the RF object
# rf_clf = RandomForestClassifier(random_state=314) 

# Best parameters (auto deprecated???) # 1141 seg (19 m)
#param_grid = { 
#     'n_estimators': [100, 200],
#    'max_features': ['none', 'auto', 'sqrt', 'log2'],
#    'max_depth' : [1,10],
#    'min_samples_leaf' : [10,20]
#}

#start_time = time.time()
#rf_clf_GS = GridSearchCV(rf_clf, param_grid, cv=5)
#rf_clf_GS.fit(x_train, y_train)
#dt_time_fit = time.time() - start_time
#rf_clf_GS.best_params_

# Train using the best parameters  # 12 seg
rf_clf_1 = RandomForestClassifier(random_state=314, n_estimators = 100,
                                  max_features = 'auto', max_depth = 10,
                                  min_samples_leaf = 20)
                                  
                                  
start_time = time.time()
rf_clf_1 = rf_clf_1.fit(x_train, y_train)  
rf_clf_1_time_fit = time.time() - start_time   

###### 4.1.3.Predicting _________________________________________________ ####

#Predictions to check # 0.631
pred_rf_clf_1 = rf_clf_1.predict(x_test)
f1_score(y_test,pred_rf_clf_1, average='micro')

# confusion matrix
confusion_matrix(y_test,pred_rf_clf_1)
pp_matrix_from_data(y_test, pred_rf_clf_1)

# Predictions to send # 0.6312 on competition (1469 / 5974)
test = pd.get_dummies(test, drop_first = True)
pred_rf_clf_1_final = rf_clf_1.predict(test)

my_submission = pd.read_csv("https://raw.githubusercontent.com/Nell87/drivendata_richter/main/data/submission_format.csv",
                            index_col='building_id')
                            
my_submission = pd.DataFrame(data=pred_rf_clf_1_final,
                             columns=my_submission.columns,
                             index=my_submission.index)

my_submission.head()
# my_submission.to_csv('../data/submission_rf_clf_1.csv')

##### 4.2.Baseline Random Forest + preprocessing _____________ ####
# Create train_merge
train_values = data = pd.read_csv("https://raw.githubusercontent.com/Nell87/drivendata_richter/main/data/train_values.csv",index_col='building_id')
train_labels = pd.read_csv("https://raw.githubusercontent.com/Nell87/drivendata_richter/main/data/train_labels.csv",index_col='building_id')
train_merge = train_values.merge(train_labels, on = 'building_id', how = 'inner',)

# Add labels
damage_map = {1:"Low", 2:"Medium", 3:"High"}
train_merge["damage_grade"] = train_merge["damage_grade"].map(damage_map)

###### 4.2.1.Preprocessing: Dummify, ourliers, oversampling and split____ ####
# Feature engineering
# Function to replace a categorical feature with many values, with their conditional probabilities respecto to the predicted feature
def categoricalvalues_condprob(data, index, pred_feature, new_column_name):
  # Create prob table
  probs = data.groupby(index).size().div(len(data))
  probs_group = data.groupby([index, pred_feature]).size().div(len(data)).div(probs, axis=0, level=index).reset_index()
  probs_group.columns= [index, pred_feature, new_column_name]
  probs_group_wide = probs_group.pivot(index=[index], columns = pred_feature,values = new_column_name) #Reshape from long to wide
  probs_group_wide = probs_group_wide.reset_index()
  
 # Rename columns
  unique_values = np.unique(data[pred_feature])
  unique_values = -(len(unique_values))
  for i in range(unique_values,0):
    probs_group_wide.rename(columns={probs_group_wide.columns[i]: index + "_" + str(probs_group_wide.columns[i])}, inplace = True)
    
  # Add column to main dataset
  data_merge = data.merge(probs_group_wide, on=index, how='left')

  # Get rid of the categorical feature
  data_merge = data_merge.drop(index, axis=1)

  # Return dataset
  return data_merge
# Apply the function
train_merge = categoricalvalues_condprob(train_merge, 'geo_level_1_id', 'damage_grade', 'prob_cond_geo_level_1')
train_merge = categoricalvalues_condprob(train_merge, 'geo_level_2_id', 'damage_grade', 'prob_cond_geo_level_2')
train_merge = categoricalvalues_condprob(train_merge, 'geo_level_3_id', 'damage_grade', 'prob_cond_geo_level_3')

# Replace the missing values with 0
cols = ["geo_level_1_id_Low", "geo_level_1_id_Medium", "geo_level_1_id_High",
                         "geo_level_2_id_Low", "geo_level_2_id_Medium", "geo_level_2_id_High",
                         "geo_level_3_id_Low", "geo_level_3_id_Medium", "geo_level_3_id_High"]

train_merge.fillna({"geo_level_1_id_Low":0, "geo_level_1_id_Medium":0, "geo_level_1_id_High":0,
                         "geo_level_2_id_Low":0, "geo_level_2_id_Medium":0, "geo_level_2_id_High":0,
                         "geo_level_3_id_Low":0, "geo_level_3_id_Medium":0, "geo_level_3_id_High":0}, inplace=True)



# Remove outliers
train_merge_ready = train_merge[train_merge['count_floors_pre_eq'] <= 3]
train_merge_ready = train_merge[train_merge['count_families'] > 3]
train_merge_ready = rem_outliers_IQR(train_merge, 'age')
train_merge_ready = rem_outliers_IQR(train_merge, 'area_percentage')
train_merge_ready = rem_outliers_IQR(train_merge, 'height_percentage')

# Dummify
train_merge_dum = pd.get_dummies(train_merge.drop("damage_grade",1), drop_first = True)

# Oversampling
oversampler=SMOTE(random_state=0)
os_features,os_labels=oversampler.fit_resample(train_merge_dum,train_labels)
train_merge.info()

# Split in train/test
x_train, x_test, y_train, y_test = train_test_split(os_features, os_labels, test_size = 0.2, random_state = 42)

###### 4.2.2.Modeling____________________________________________ ####
# Create the RF object
# rf_clf = RandomForestClassifier(random_state=314) 

# Best parameters (auto deprecated???) # 1141 seg (19 m)
#param_grid = { 
#     'n_estimators': [100, 200],
#    'max_features': ['none', 'auto', 'sqrt', 'log2'],
#    'max_depth' : [1,10],
#    'min_samples_leaf' : [10,20]
#}

#start_time = time.time()
#rf_clf_GS = GridSearchCV(rf_clf, param_grid, cv=5)
#rf_clf_GS.fit(x_train, y_train)
#dt_time_fit = time.time() - start_time
#rf_clf_GS.best_params_

# Train using the best parameters  # 12 seg
rf_clf_2 = RandomForestClassifier(random_state=314)
                                  
                                  
start_time = time.time()
rf_clf_2 = rf_clf_2.fit(x_train, y_train)  
rf_clf_2_time_fit = time.time() - start_time   

###### 4.2.3.Predicting _________________________________________________ ####

#Predictions to check # 0.794
pred_rf_clf_2 = rf_clf_2.predict(x_test)
f1_score(y_test,pred_rf_clf_2, average='micro')

# confusion matrix
confusion_matrix(y_test,pred_rf_clf_2)
pp_matrix_from_data(y_test, pred_rf_clf_2)

# Predictions to send # 0.6312 on competition (1469 / 5974)

# Prepare the test
# Create train_merge
train_values = data = pd.read_csv("https://raw.githubusercontent.com/Nell87/drivendata_richter/main/data/train_values.csv",index_col='building_id')
train_labels = pd.read_csv("https://raw.githubusercontent.com/Nell87/drivendata_richter/main/data/train_labels.csv",index_col='building_id')
train_merge = train_values.merge(train_labels, on = 'building_id', how = 'inner',)

# Add labels
damage_map = {1:"Low", 2:"Medium", 3:"High"}
train_merge["damage_grade"] = train_merge["damage_grade"].map(damage_map)
test_values = pd.read_csv("https://raw.githubusercontent.com/Nell87/drivendata_richter/main/data/test_values.csv",index_col='building_id')

# Apply the function to replace the categorical feature (without remove the original)
def categoricalvalues_condprob(data, index, pred_feature, new_column_name):
  # Create prob table
  probs = data.groupby(index).size().div(len(data))
  probs_group = data.groupby([index, pred_feature]).size().div(len(data)).div(probs, axis=0, level=index).reset_index()
  probs_group.columns= [index, pred_feature, new_column_name]
  probs_group_wide = probs_group.pivot(index=[index], columns = pred_feature,values = new_column_name) #Reshape from long to wide
  probs_group_wide = probs_group_wide.reset_index()
  
 # Rename columns
  unique_values = np.unique(data[pred_feature])
  unique_values = -(len(unique_values))
  for i in range(unique_values,0):
    probs_group_wide.rename(columns={probs_group_wide.columns[i]: index + "_" + str(probs_group_wide.columns[i])}, inplace = True)
    
  # Add column to main dataset
  data_merge = data.merge(probs_group_wide, on=index, how='left')

  # Get rid of the categorical feature
  #data_merge = data_merge.drop(index, axis=1)

  # Return dataset
  return data_merge

# Apply the function
train_merge1 = categoricalvalues_condprob(train_merge, 'geo_level_1_id', 'damage_grade', 'prob_cond_geo_level_1')
train_merge1_prob = train_merge1[["geo_level_1_id", "geo_level_1_id_High", "geo_level_1_id_Low", "geo_level_1_id_Medium"]]
train_merge1_prob = train_merge1_prob.drop_duplicates()      

train_merge2 = categoricalvalues_condprob(train_merge, 'geo_level_2_id', 'damage_grade', 'prob_cond_geo_level_2')
train_merge2_prob = train_merge2[["geo_level_2_id", "geo_level_2_id_High", "geo_level_2_id_Low", "geo_level_2_id_Medium"]]
train_merge2_prob = train_merge2_prob.drop_duplicates()      


train_merge3 = categoricalvalues_condprob(train_merge, 'geo_level_3_id', 'damage_grade', 'prob_cond_geo_level_3')
train_merge3_prob = train_merge3[["geo_level_3_id", "geo_level_3_id_High", "geo_level_3_id_Low", "geo_level_3_id_Medium"]]
train_merge3_prob = train_merge3_prob.drop_duplicates()      


# Add new columns to test dataset
test_prb = test_values.merge(train_merge1_prob,on="geo_level_1_id",  how='left')
test_prb = test_prb.merge(train_merge2_prob,on="geo_level_2_id",  how='left')
test_prb = test_prb.merge(train_merge3_prob,on="geo_level_3_id",  how='left')


# Replace the missing values with 0
test_prb.fillna({"geo_level_1_id_Low":0, "geo_level_1_id_Medium":0, "geo_level_1_id_High":0,
                         "geo_level_2_id_Low":0, "geo_level_2_id_Medium":0, "geo_level_2_id_High":0,
                         "geo_level_3_id_Low":0, "geo_level_3_id_Medium":0, "geo_level_3_id_High":0}, inplace=True)
                         
test_prb_clean = test_prb.drop(["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"], axis=1)
                         
test_prb_clean = pd.get_dummies(test_prb_clean, drop_first = True)

# Predicing
pred_rf_clf_2_final = rf_clf_2.predict(test_prb_clean)

my_submission2 = pd.read_csv("https://raw.githubusercontent.com/Nell87/drivendata_richter/main/data/submission_format.csv",
                            index_col='building_id')
                            
my_submission2 = pd.DataFrame(data=pred_rf_clf_2_final,
                             columns=my_submission2.columns,
                             index=my_submission2.index)

my_submission2.head()
my_submission2.to_csv('submission_rf_clf_3.csv')
