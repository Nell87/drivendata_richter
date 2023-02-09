# RICHTER COMPETITION
DrivenData is a website that hosts data science competitions focused on having a real-world impact. Using the CRISP-DM methodology, I worked on the "Richter's Predictor" competition. The goal is to predict the level of damage to buildings caused by the 2015 Gorkha earthquake in Nepal. 

## 1. Business Understanding
Earthquakes may not happen frequently, however their consequences can be catastrophic. So based on this data, the goal is to identify which buildings are prone to be damaged, in case another earquakes happens. The objective was to develop a model to predict the kind of damage for each building. 

## 2. Data Understanding
The data was collected through surveys by Kathmandu Living Labs and the Central Bureau of Statistics, which works under the National Planning Commission Secretariat of Nepal. This survey is one of the largest post-disaster datasets ever collected, containing valuable information on earthquake impacts, household conditions, and socio-economic-demographic statistics.

The train dataset has 260,601 rows and the test dataset has 86,868 rows. Both have 38 features to predict the level of damage to every building (the damage_grade feature). During the first EDA we reach some conclusions:
- There are not missing values or duplicates.
- We deal with imbalanced data (almost 60% of the buildings suffer a medium amount of damage, followed by high with around 30% and low with around 10%)
- All numerical features indicate the existence of outliers
- For every location feature there is a high number of categorical values, so we'll apply feature engineering on them
- Some of the features are highly correlated

## 3. Data Preparation
- We handle imbalanced data using SMOTE.
- We remove outliers using business understanding and IQR 
- We apply feature engineering on location features. We replace every value with their conditional probabilities respect to every damage_grade category.

## 4. Modeling
- We dummy categorical features. 
- Split into train and test
- Train a RF baseline model 
- Train the same RF baseline model with all the preprocessing we mentioned

## 5. Evaluation
We are predicting the level of damage from 1 to 3. The level of damage is an ordinal variable meaning that ordering is important. This can be viewed as a classification or an ordinal regression problem. To measure the performance of our algorithms, we'll use the F1 score which balances the precision and recall of a classifier. 

We achieved a micro-averaged F1 score = 0.7248 (Current Rank 969/6031)

## Next steps
- Comprehensive analysis of every feature's impact
- Drop out highly correlated or not relevant features
- Create new features from existing features
- Test more machine learning models
