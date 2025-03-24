from ucimlrepo import fetch_ucirepo 
  
cdc_diabetes_health_indicators = fetch_ucirepo(id=891) 
  
data_train = cdc_diabetes_health_indicators.data.features.values
data_target = cdc_diabetes_health_indicators.data.targets.values.ravel()
