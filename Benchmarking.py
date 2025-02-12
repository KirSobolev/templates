import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn import metrics, svm
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.metrics import classification_report

class Benchmarking:
    def __init__(self, df: pd.DataFrame, target_col: str, mode: str, test_size=0.2):
        self.df = df
        self.target_column = target_col
        self.mode = mode
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = None
        self.metrics_df = None
        self.test_size = test_size
        if self.mode == "regression":
            self.models = [LinearRegression(), 
                           svm.SVR(), 
                           RandomForestRegressor(), 
                           XGBRegressor(), 
                           LGBMRegressor()]
            self.metrics_df  = pd.DataFrame({'Metric': ['Mean Squared Error', 
                                                        'Root Mean Squared Error', 
                                                        'Mean Absolute Error', 
                                                        'R-squared', 
                                                        'Time consumed']})
        elif self.mode == "classification":
            self.models = [LogisticRegression(), 
                           svm.SVC(), 
                           RandomForestClassifier(), 
                           XGBClassifier(), 
                           LGBMClassifier()]
            self.metrics_df = pd.DataFrame({'Metric': ['Total accuracy', 
                                                      'Macro precision', 
                                                      'Macro recall', 
                                                      'Macro F1',
                                                      'Time Consumed']})
        else:
            raise ValueError("Wrong mode! Should be 'regression' or 'classification'")

    def split_data(self):
        """Splits the data into X/y and X/y train/test"""
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, 
                                                                                test_size=self.test_size, 
                                                                                random_state=888)
        
    def train(self):
        """Train all models in self.models"""
        for model in self.models:
            start_time = time.time()
            # Training the model
            print(f"Training {model.__class__.__name__}...")
            model.fit(self.X_train, self.y_train)
            end_time = time.time()
            execution_time = end_time - start_time
            predictions = model.predict(self.X_test)

            # Regression metrics
            if self.mode == 'regression':
                mae = round(metrics.mean_absolute_error(self.y_test, predictions), 2)
                mse = round(metrics.mean_squared_error(self.y_test, predictions), 2)
                rmse = round(np.sqrt(metrics.mean_squared_error(self.y_test, predictions)), 2)
                r2 = round(metrics.r2_score(self.y_test, predictions), 2)
                self.metrics_df[model.__class__.__name__] = [mse, rmse, mae, r2, execution_time]

            # Classification metrics
            elif self.mode == 'classification':
                report = classification_report(self.y_test, predictions, output_dict=True)
                acc = round(report['accuracy'], 2)
                macro_precision = round(report['macro avg']['precision'], 2)
                macro_recall = round(report['macro avg']['recall'], 2)
                macro_f1 = round(report['macro avg']['f1-score'], 2)
                self.metrics_df[model.__class__.__name__] = [acc, macro_precision, macro_recall, macro_f1, execution_time]

    def print_metrics(self):
        """Prints dataframe with metrics"""
        print()
        print()
        print(self.metrics_df.to_markdown())

    def benchmark(self):
        """Creates benchamrking pipeline"""
        self.split_data()
        self.train()
        self.print_metrics()

