import pandas as pd
import numpy as np
import time
import logging

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import metrics, svm
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.metrics import classification_report

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class BaseBenchmark:
    def __init__(self, df: pd.DataFrame, target_col: str, test_size=0.2):
        self.df = df
        self.target_col = target_col
        self.test_size = test_size
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()
        self.models = self.get_models()
        self.results = pd.DataFrame()

    def split_data(self):
        """Splits the dataset into training and testing sets."""
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        return train_test_split(X, y, test_size=self.test_size, random_state=888)

    def get_models(self):
        """To be implemented by subclasses."""
        raise NotImplementedError

    def train_and_evaluate(self):
        """Trains models and evaluates them."""
        metrics_data = []

        for name, model in self.models.items():
            try:
                start_time = time.time()
                logging.info(f"Training {name}...")
                model.fit(self.X_train, self.y_train)
                execution_time = (time.time() - start_time) * 1000  # Convert to ms

                predictions = model.predict(self.X_test)
                metrics_row = self.evaluate_model(name, predictions, execution_time)
                metrics_data.append(metrics_row)

            except Exception as e:
                logging.error(f"Error training {name}: {e}")
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
    def get_models(self):
        """Returns regression models."""
        return {
            "LinearRegression": LinearRegression(),
            "SVR": svm.SVR(),
            "RandomForestRegressor": RandomForestRegressor(),
            "XGBRegressor": XGBRegressor(),
            "LGBMRegressor": LGBMRegressor(),
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
    def get_models(self):
        """Returns classification models."""
        return {
            "LogisticRegression": LogisticRegression(),
            "SVC": svm.SVC(),
            "RandomForestClassifier": RandomForestClassifier(),
            "XGBClassifier": XGBClassifier(),
            "LGBMClassifier": LGBMClassifier(),
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
