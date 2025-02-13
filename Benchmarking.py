import pandas as pd
import numpy as np
import catboost as cb

import time
import logging

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import metrics, svm
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class BaseBenchmark:
    def __init__(self, df: pd.DataFrame, target_col: str, categorical_variables: list=None, models: dict=None, test_size=0.2, scaler=MinMaxScaler):
        self.df = df
        self.target_col = target_col
        self.test_size = test_size
        self.scaler_X = scaler()
        self.X_train, self.X_test, self.y_train, self.y_test, self.X_train_scaled, self.X_test_scaled = self.split_data()
        self.categorical_variables = categorical_variables
        
        # Allow user to pass custom models, otherwise use defaults
        self.models = models if models is not None else self.get_default_models()
        self.results = pd.DataFrame()

    def split_data(self):
        """Splits the dataset into training and testing sets and applies scaling of choice"""
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        # Perform train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=888)

        # Apply scaling if enabled
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)

        return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled

    def get_default_models(self):
        """To be implemented by subclasses."""
        raise NotImplementedError

    def train_and_evaluate(self):
        """Trains models and evaluates them."""
        metrics_data = []
        for name, model in self.models.items():
            try:
                predictions = None
                start_time = time.time()
                logging.info(f"Training {name}...")

                # Handle KNN n neighbors
                if name == "KNNRegressor" or name == "KNNClassifier":
                    max_iterations = 70
                    error = []
                    # Calculating error for K values between 1 and 40
                    for i in range(1, max_iterations):
                        # try with current k-value, train the model and make a test prediction
                        knn = model(n_neighbors=i, metric="minkowski")
                        knn.fit(self.X_train, self.y_train)
                        pred_i = knn.predict(self.X_test)
                        # save the error value for this k-value
                        if name == "KNNRegressor":
                            error.append(np.sqrt(metrics.mean_squared_error(self.y_test, pred_i)))
                        elif name == "KNNClassifier":
                            error.append(np.mean(pred_i != self.y_test))
                    k_value = np.argmin(error) + 1
                    model = model(n_neighbors=k_value, metric="minkowski")
                    model.fit(self.X_train_scaled, self.y_train)
                    
                elif name == "LGBMRegressor":
                    model = model(objective="regression", verbose=0)
                    model.fit(self.X_train_scaled, self.y_train)

                elif name == "CatBoostRegressor" or name == "CatBoostClassifier":
                    # CatBoost doesn't work well with scaling, because it messes up cat_features
                    if self.categorical_variables:
                        model = model(verbose=0)
                        model.fit(self.X_train, self.y_train, cat_features=self.categorical_variables)
                        predictions = model.predict(self.X_test)
                    else:
                        logging.info("Categorical variable list was not provided. Skipping CatBoost model...")
                        continue
                else:
                    model = model()
                    model.fit(self.X_train_scaled, self.y_train)
                    
                execution_time = (time.time() - start_time) * 1000  # Convert to ms

                if predictions is None:
                    predictions = model.predict(self.X_test_scaled)

                metrics_row = self.evaluate_model(name, predictions, execution_time)
                metrics_data.append(metrics_row)

            except Exception as e:
                logging.error(f"{name}: {e}")
                continue

        self.results = pd.DataFrame(metrics_data)

    def evaluate_model(self, name, predictions, execution_time):
        """To be implemented by subclasses."""
        raise NotImplementedError

    def benchmark(self):
        """Runs the benchmarking process."""
        self.train_and_evaluate()
        print("\nBenchmarking Results:\n", self.results.to_markdown())

class RegressionBenchmark(BaseBenchmark):
    def get_default_models(self):
        """Returns default regression models."""
        return {
            "LinearRegression": LinearRegression,
            "SVR": svm.SVR,
            "RandomForestRegressor": RandomForestRegressor,
            "XGBRegressor": XGBRegressor,
            "LGBMRegressor": LGBMRegressor,
            "KNNRegressor": KNeighborsRegressor,
            "CatBoostRegressor": CatBoostRegressor,
        }

    def evaluate_model(self, name, predictions, execution_time):
        """Computes regression metrics."""
        mse = metrics.mean_squared_error(self.y_test, predictions)
        rmse = np.sqrt(mse)
        mae = metrics.mean_absolute_error(self.y_test, predictions)
        r2 = metrics.r2_score(self.y_test, predictions)

        return {
            "Model": name,
            "MSE": round(mse, 4),
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "R2": round(r2, 4),
            "Time (ms)": round(execution_time, 2),
        }

class ClassificationBenchmark(BaseBenchmark):
    def get_default_models(self):
        """Returns default classification models."""
        return {
            "LogisticRegression": LogisticRegression,
            "SVC": svm.SVC,
            "RandomForestClassifier": RandomForestClassifier,
            "XGBClassifier": XGBClassifier,
            "LGBMClassifier": LGBMClassifier,
            "KNNClassifier": KNeighborsClassifier,
            "CatBoostClassifier": CatBoostClassifier,
        }

    def evaluate_model(self, name, predictions, execution_time):
        """Computes classification metrics."""
        report = classification_report(self.y_test, predictions, output_dict=True)

        return {
            "Model": name,
            "Accuracy": round(report["accuracy"], 4),
            "Precision": round(report["macro avg"]["precision"], 4),
            "Recall": round(report["macro avg"]["recall"], 4),
            "F1-score": round(report["macro avg"]["f1-score"], 4),
            "Time (ms)": round(execution_time, 2),
        }


# TODO: Add ANN and optimization feature.

if __name__=="__main__":
    print("Module is supposed to be imported")