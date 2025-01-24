from bikeshare.utils.config import Config
from bikeshare.configs.config import CFGLog
import os 
import pickle

class Inferrer:
    def __init__(self):
        self.config = Config.from_json(CFGLog)
        # self.dt_saved_path = os.path.join(self.config.output.output_path, self.config.output.dt_path, self.config.output.dt_model)
        # with open(self.dt_saved_path, "rb") as f:
        #     self.dt_col_transformer, self.dt_model = pickle.load(f)
        
        # self.rf_saved_path = os.path.join(self.config.output.output_path, self.config.output.rf_path, self.config.output.rf_model)
        # with open(self.rf_saved_path, "rb") as f:
        #     self.rf_col_transformer, self.rf_model = pickle.load(f)
        
        self.xgb_saved_path = os.path.join(self.config.output.output_path, self.config.output.xgb_model)
        with open(self.xgb_saved_path, "rb") as f:
            self.xgb_col_transformer, self.xgb_model = pickle.load(f)
    
    # def dt_preprocess(self, new_data):
    #     return self.dt_col_transformer.transform(new_data)
    
    # def rf_preprocess(self, new_data):
    #     return self.rf_col_transformer.transform(new_data)
    
    def xgb_preprocess(self, new_data):
        return self.xgb_col_transformer.transform(new_data)
    
    # def get_col_names_after_transform(self):
    #     """ Get column names after transformation """
    #     dt_num_names = self.dt_col_transformer.named_transformers_['num'].get_feature_names_out()
    #     dt_cat_names = self.dt_col_transformer.named_transformers_['cat'].get_feature_names_out()
    #     dt_cols = list(dt_num_names) + list(dt_cat_names)

    #     return dt_cols
    
    
    ############# Decision Tree #############
    # def dt_infer(self, new_data):
    #     """ Infer data using decision tree model """
    #     transformed_data = self.dt_preprocess(new_data)
    #     dt_prediction = self.dt_model.predict(transformed_data)
    #     print(f'Model in use: {self.dt_saved_path}')
    #     return dt_prediction
    
    # def dt_feature_importance(self):
    #     """ Return feature importance for decision tree model """
    #     return self.dt_model.feature_importances_
    
    
    ############# Random Forest #############
    # def rf_infer(self, new_data):
    #     """ Infer data using random forest model """
    #     transformed_data = self.rf_preprocess(new_data)
    #     rf_prediction = self.rf_model.predict(transformed_data)
    #     print(f'Model in use: {self.rf_saved_path}')
    #     return rf_prediction
    
    # def rf_feature_importance(self):
    #     """ Return feature importance for random forest model """
    #     return self.rf_model.feature_importances_
    
    
    ############# XGBoost #############
    def xgb_infer(self, new_data):
        """ Infer data using xgboost model """
        transformed_data = self.xgb_preprocess(new_data)
        xgb_prediction = self.xgb_model.predict(transformed_data)
        print(f'Model in use: {self.xgb_saved_path}')
        return xgb_prediction
    
    def xgb_feature_importance(self):
        """ Return feature importance for xgboost model """
        return self.xgb_model.feature_importances_
    
    