# Main executing script for the program
import pandas as pd

from bikeshare.configs.config import CFGLog
from bikeshare.model.bikeshare_model import BikeshareXGBoost
from bikeshare.executor.inferrer import Inferrer


def run():
    # Load the configuration file
    config = CFGLog
    print("----------------------------------")
    print("Configuration file loaded")
    print("----------------------------------")
    
    # Gradient Boosting model
    xgb_model = BikeshareXGBoost(config)
    xgb_model.load_data()
    xgb_model.build()
    xgb_model.train()
    xgb_model.evaluate()
    xgb_model.export_model()
    
    # new data:
    # data_dict = {
    #     'hour': 0,
    #     'temp': 5.0,
    #     'humidity': 60,
    #     'wind_speed': 2.0,
    #     'visibility': 2000,
    #     'solar_rad': 0.0,
    #     'rainfall': 0.0,
    #     'snowfall': 0.0,
    #     'seasons': 'Winter',
    #     'holiday': 'No Holiday',
    #     'day': 'Friday',
    #     'month': 'January'
    # }
    
    # data_df = pd.DataFrame([data_dict])
    
    # # Inferrer
    # inferrer = Inferrer()
    # print("\nDecision Tree Prediction: ", inferrer.dt_infer(data_df))
    # print("\nDecision Tree Feature Importance: ", inferrer.dt_feature_importance())
    
    # print("\nRandom Forest Prediction: ", inferrer.rf_infer(data_df))
    # print("\nRandom Forest Feature Importance: ", inferrer.rf_feature_importance())
    
    # print("\nXGBoost Prediction: ", inferrer.xgb_infer(data_df))
    # print("\nXGBoost Feature Importance: ", inferrer.xgb_feature_importance())

    # col_names = inferrer.get_col_names_after_transform()
    # print("\nDecision Tree Column Names: ", col_names)
    
    
    
    # print(len(dt_names))
    # print(len(rf_names))
    # print(len(xgb_names))
    
    print("----------------------------------")
    

if __name__ == "__main__":
    run()


