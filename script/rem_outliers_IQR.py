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
