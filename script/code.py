#### 0.   INCLUDES  _______________________________________ #### 
#Load Libraries:# 
import pandas as pd
import time   #  provides many ways of representing time in code

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import f1_score   # the score used in the competition

#### 1.  READING TRAIN AND TEST DATA _______________________________________ #### 
train_values= data = pd.read_csv("https://raw.githubusercontent.com/Nell87/drivendata_richter/main/data/train_values.csv",index_col='building_id')
train_labels = pd.read_csv("https://raw.githubusercontent.com/Nell87/drivendata_richter/main/data/train_labels.csv",index_col='building_id')
# train = train.merge(train_labels, on='building_id')
test = pd.read_csv("https://raw.githubusercontent.com/Nell87/drivendata_richter/main/data/test_values.csv",index_col='building_id')

#### 2.  PREPROCESSING DATA     _______________________________________ #### 
train_values = pd.get_dummies(train_values, drop_first = True)
x_train, x_test, y_train, y_test = train_test_split(train_values, train_labels, test_size = 0.2, random_state = 42)


#### 3.  MODELING     _______________________________________ #### 
rf_clf = RandomForestClassifier(random_state=314) 

# Best parameters (auto deprecated???) # 1141 seg (19 m)
param_grid = { 
    'n_estimators': [100, 200],
    'max_features': ['none', 'auto', 'sqrt', 'log2'],
    'max_depth' : [1,10],
    'min_samples_leaf' : [10,20]
}

start_time = time.time()
rf_clf_GS = GridSearchCV(rf_clf, param_grid, cv=5)
rf_clf_GS.fit(x_train, y_train)
dt_time_fit = time.time() - start_time
rf_clf_GS.best_params_



# Train using the best parameters  # 12 seg
rf_clf_1 = RandomForestClassifier(random_state=314, n_estimators = 100,
                                  max_features = 'auto', max_depth = 10,
                                  min_samples_leaf = 20)
                                  
                                  
start_time = time.time()
rf_clf_1 = rf_clf_1.fit(x_train, y_train)  
rf_clf_1_time_fit = time.time() - start_time                                  


#### 4.  PREDICTIONS     _______________________________________ #### 

#Predictions to check # 0.631
pred_rf_clf_1 = rf_clf_1.predict(x_test)
f1_score(y_test,pred_rf_clf_1, average='micro')

# Predictions to send # 0.6312 on  (1469 / 5974)
test = pd.get_dummies(test, drop_first = True)
pred_rf_clf_1_final = rf_clf_1.predict(test)

my_submission = pd.read_csv("https://raw.githubusercontent.com/Nell87/drivendata_richter/main/data/submission_format.csv",
                            index_col='building_id')
                            
my_submission = pd.DataFrame(data=pred_rf_clf_1_final,
                             columns=my_submission.columns,
                             index=my_submission.index)

my_submission.head()
my_submission.to_csv('../data/submission_rf_clf_1.csv')




#### ____   EXTRA IDEAS  _______________________________________ #### 

#Decision Tree
dt = DecisionTreeClassifier(max_features = None,
                            max_depth = 45,
                            min_samples_split = 3,
                            min_samples_leaf = 30,
                            random_state=42)
start_time = time.time()
model = dt.fit(X_train, y_train)
dt_time_fit = time.time() - start_time

#Predictions - Decision Tree
start_time = time.time()
model.predict(X_test)
dt_time_pred = time.time() - start_time

print("Decision Tree")
print("Fit Time: {} seconds".format(dt_time_fit))
print("Prediction Time: {} seconds".format(dt_time_pred))
print("Training Score: {}".format(dt.score(X_train, y_train)))
print("Test Score: {}".format(dt.score(X_test, y_test)))
print("----------------------------------------")



# Numerical columns
num_col = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id', 'age', 'area_percentage','count_floors_pre_eq' ,'height_percentage']

# Define standard scaler
scaler = StandardScaler()

# Transform data 
train = train[num_col]
train[num_col] = scaler.fit_transform(train[num_col])

# EXTRA IDEAS
# We add the label 1 (train) and 0 (test) and merge both datasets
train["check"]=1
test["check"]=0
merged_data = pd.concat([train,test], axis=0)


#### 1.   PREPARING DATA _______________________________________ #### 
data = pd.read_csv("../data/train_values.csv")
profile = ProfileReport(merged_data, title="Profiling Report")
profile.to_file("Profiling Report.html")


