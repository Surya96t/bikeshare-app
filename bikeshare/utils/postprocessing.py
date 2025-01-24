import datetime
import os
import pickle 
import numpy as np
import json

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class ModelSaving(object):
    
    @staticmethod
    def get_current_timestamp():
        now = datetime.datetime.now()
        return now.strftime("%Y-%m-%d_%H-%M-%S")
    
    @staticmethod 
    def save_model_with_timestamp(col_transformer, model, model_name, output_config):
        filename = model_name + "_" + ModelSaving.get_current_timestamp() + ".pkl"
        filepath = os.path.join(output_config, filename)
        with open(filepath, 'wb') as outputfile:
            pickle.dump((col_transformer, model), outputfile)
        
        return print("Saved column transformer and model to: ", filepath)
    
    @staticmethod
    def save_model_metrics(y_true, y_pred, model_name, output_config):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        metrics = {
            "RMSE": rmse,
            "MAE": mae,
            "MSE": mse,
            "R2_Score": r2
        }
        metrics = {model_name: metrics}
        filename = model_name + "_metrics" + ".json"
        filepath = os.path.join(output_config, filename)
        with open(filepath, 'w') as outputfile:
            json.dump(metrics, outputfile, indent=4)
        
        return print("Saved model metrics to: ", filepath)