from ucimlrepo import fetch_ucirepo

def read_data(database_id):
    
    cdc_diabetes_health_indicators = fetch_ucirepo(id=database_id)
    x = cdc_diabetes_health_indicators.data.features
    y = cdc_diabetes_health_indicators.data.targets.squeeze()
    
    print(x, y)
    
    return  x, y
