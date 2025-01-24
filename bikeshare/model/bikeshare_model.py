from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xg

from datetime import datetime
from .base_model import BaseModel
from bikeshare.dataloader.dataloader import DataLoader
from bikeshare.executor.trainer import ModelTrainer
from bikeshare.utils.postprocessing import ModelSaving


class BikeshareXGBoost(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self._name = "XGBoost"
        
    def load_data(self):
        """ Load data """
        self.dataset = DataLoader().load_data(self.config.data)
        self.X, self.y, self.X_train, self.X_test, self.y_train, self.y_test, \
            self.col_transformer = DataLoader().preprocess_data(self.config.data, dataset = self.dataset)
            
            
    def build(self):
        """ Build the model """
        self.model = xg.XGBRegressor()
        print("\nXGBoost model built")
        
        
    def train(self):
        """ Complies and trains the model with the configured hyperparameters """
        print("Setting the XGBoost training parameters")
        self.model = xg.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=self.config.gradient_boosting.n_estimators,
            max_depth=self.config.gradient_boosting.max_depth,
            subsample=self.config.gradient_boosting.subsample,
            learning_rate=self.config.gradient_boosting.learning_rate,
            random_state=self.config.data.random_state
        )
        print("XGBoost training is started")
        start_time = datetime.now()
        trainer = ModelTrainer(
            self.model,
            X_train=self.X_train,
            y_train=self.y_train
        )
        trainer.train()
        end_time = datetime.now()  
        training_time = (end_time - start_time).total_seconds()
        print(f"XGBoost training is completed. Time taken: {"{:.2f}".format(training_time)} seconds")
        
    def evaluate(self):
        """ XGBoost predicts the results for the test data"""
        self.y_test_pred = self.model.predict(self.X_test)
        output_config = self.config.output.output_path + self.config.output.xgb_path
        ModelSaving().save_model_metrics(self.y_test, self.y_test_pred, self._name, output_config)
        print("XGBoost model evaluation on test test completed, check model attributes for results")
        
    def evaluate_new_data(self, new_data):
        """ Predicts the results for the new data """
        new_data = self.col_transformer.transform(new_data)
        return self.model.predict(new_data)
        
    def export_model(self):
        """ Saves the model """
        output_config = self.config.output.output_path + self.config.output.xgb_path
        ModelSaving().save_model_with_timestamp(self.col_transformer, self.model, self._name, output_config)

# class BikeshareDecisionTree(BaseModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self._name = "DecisionTree"
        
#     def load_data(self):
#         """ Load data """
#         self.dataset = DataLoader().load_data(self.config.data)
#         self.X, self.y, self.X_train, self.X_test, self.y_train, self.y_test, \
#             self.col_transformer = DataLoader().preprocess_data(self.config.data, dataset = self.dataset)
            
            
#     def build(self):
#         """ Build the model """
#         self.model = DecisionTreeRegressor()
#         print("\nDecision Tree model built")
        
        
#     def train(self):
#         """ Complies and trains the model with the configured hyperparameters """
#         print("Setting the Decision Tree training parameters")
#         self.model = DecisionTreeRegressor(
#             criterion=self.config.decision_tree.criterion,
#             max_depth=self.config.decision_tree.max_depth,
#             max_features=self.config.decision_tree.max_features,
#             max_leaf_nodes=self.config.decision_tree.max_leaf_nodes,
#             min_samples_leaf=self.config.decision_tree.min_samples_leaf,
#             min_samples_split=self.config.decision_tree.min_samples_split,
#             splitter=self.config.decision_tree.splitter,
#             random_state=self.config.data.random_state
#         )
#         print("Decision Tree training is started")
#         start_time = datetime.now()
#         trainer = ModelTrainer(
#             self.model,
#             X_train=self.X_train,
#             y_train=self.y_train
#         )
#         trainer.train()
#         end_time = datetime.now()  
#         training_time = (end_time - start_time).total_seconds()
#         print(f"Decision Tree training is completed. Time taken: {"{:.2f}".format(training_time)} seconds")
        
#     def evaluate(self):
#         """ Decision Tree predicts the results for the test data"""
#         self.y_test_pred = self.model.predict(self.X_test)
#         output_config = self.config.output.output_path  + self.config.output.dt_path
#         ModelSaving().save_model_metrics(self.y_test, self.y_test_pred, self._name, output_config)
#         print("Decision Tree model evaluation on test test completed, check model folder for results")
        
#     def evaluate_new_data(self, new_data):
#         """ Predicts the results for the new data """
#         new_data = self.col_transformer.transform(new_data)
#         return self.model.predict(new_data)
        
#     def export_model(self):
#         """ Saves the model """
#         output_config = self.config.output.output_path  + self.config.output.dt_path
#         ModelSaving().save_model_with_timestamp(self.col_transformer, self.model, self._name, output_config)
        
        
# class BikeshareRandomForest(BaseModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self._name = "RandomForest"
        
#     def load_data(self):
#         """ Load data """
#         self.dataset = DataLoader().load_data(self.config.data)
#         self.X, self.y, self.X_train, self.X_test, self.y_train, self.y_test, \
#             self.col_transformer = DataLoader().preprocess_data(self.config.data, dataset = self.dataset)
            
            
#     def build(self):
#         """ Build the model """
#         self.model = RandomForestRegressor()
#         print("\nRandom Forest model built")
        
        
#     def train(self):
#         """ Complies and trains the model with the configured hyperparameters """
#         print("Setting the Random Forest training parameters")
#         self.model = RandomForestRegressor(
#             n_estimators=self.config.random_forest.n_estimators,
#             max_depth=self.config.random_forest.max_depth,
#             max_features=self.config.random_forest.max_features,
#             min_samples_leaf=self.config.random_forest.min_samples_leaf,
#             min_samples_split=self.config.random_forest.min_samples_split,
#             random_state=self.config.data.random_state
#         )
#         print("Random Forest training is started")
#         start_time = datetime.now()
#         trainer = ModelTrainer(
#             self.model,
#             X_train=self.X_train,
#             y_train=self.y_train
#         )
#         trainer.train()
#         end_time = datetime.now()  
#         training_time = (end_time - start_time).total_seconds()
#         print(f"Random Forest training is completed. Time taken: {"{:.2f}".format(training_time)} seconds")
        
#     def evaluate(self):
#         """ Random Forest predicts the results for the test data"""
#         self.y_test_pred = self.model.predict(self.X_test)
#         output_config = self.config.output.output_path + self.config.output.rf_path
#         ModelSaving().save_model_metrics(self.y_test, self.y_test_pred, self._name, output_config)
#         print("Random Forest model evaluation on test test completed, check model attributes for results")
        
#     def evaluate_new_data(self, new_data):
#         """ Predicts the results for the new data """
#         new_data = self.col_transformer.transform(new_data)
#         return self.model.predict(new_data)
        
#     def export_model(self):
#         """ Saves the model """
#         output_config = self.config.output.output_path + self.config.output.rf_path
#         ModelSaving().save_model_with_timestamp(self.col_transformer, self.model, self._name, output_config)
        
