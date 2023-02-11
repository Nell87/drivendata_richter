# Load Libraries:
import numpy as np  # If we import numpy in code.py, this function doesn't work 

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

  # Return dataset
  return data_merge

