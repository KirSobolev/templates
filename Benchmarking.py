import pandas as pd
import numpy as np
import optuna
import joblib

import time
import logging

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
# Models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import metrics, svm
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from catboost import CatBoostClassifier, CatBoostRegressor


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class BaseBenchmark:
    """
    Parent class for classification and regression benchmarking. Split the data, trains and evaluates the models.

    Attributes:
    -----------
    df : pd.DataFrame
        Dataset
    target_col : str
        Target variable
    categorical_variables : list
        List of categorical variables. Needed for CatBoost algorithm. Defaults to None, which will skip CatBoost
    models : dict
        Dictionary with model names and models (not callable). By default will run through all supported models.
    test_size : float
        Test data size in train / test data split
    scaler : sklearn scaler 
        Defaults to MinMaxScaler. Should be not callable. 

    Methods:
    --------
    split_data():
        Scales the X data. Does not scale provided categorical variables. Returns tuple with X-test/train and y-test/train data.
    get_default_models() -> dict:
        Returns a list of default supported models.
    train_and_evaluate():
        Cycles through model dict, trains the model evaluates models by calling evaluate_model() and saves the results.
    evaluate_model(evaluate_model(self, name: str, predictions, execution_time) -> dict):
        Evaluates model and returns a dictionary with metrics
    optimize(self, trial, models: dict) -> dict
        Function that optimizes one model at a time. Returns a dictionary with model's best hyperparameters
    benchamrking(optimization_direction):
        Runs train_and_evaluate() and optimization().
    optimization(self, direction: str)
        Runs through the model list, optimizes them and saves the metrics to self.results
     mean_scores(self, model, X_train, y_train)
        Method used by optuna to evaluate the model while optimizing
    get_hyperparameters(self, trial, model_name : str) -> dict
        Returns dictionary with model's supported hyperparameters.
    print_results(self)
        Prints out self.results

    """
    def __init__(self, df: pd.DataFrame, 
                 target_col: str,
                 categorical_variables: list=None, 
                 models: dict=None, 
                 test_size=0.2, 
                 scaler=MinMaxScaler):
        """
        Initializes class. 
        
        Attributes:
        -----------
        df : pd.DataFrame
            Dataset
        target_col : str
            Target variable
        categorical_variables : list
            List of categorical variables. Needed for CatBoost algorithm. Defaults to None, which will skip CatBoost
        models : dict
            Dictionary with model names and models (not callable). By default will run through all supported models.
        test_size : float
            Test data size in train / test data split
        scaler : sklearn scaler 
            Defaults to MinMaxScaler. Should be not callable. """
        
        self.df = df
        self.target_col = target_col
        self.test_size = test_size
        self.scaler_X = scaler()
        self.categorical_variables = categorical_variables
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()
        
        
        # Allow user to pass custom models, otherwise use defaults
        self.models = models if models is not None else self.get_default_models()
        self.results = pd.DataFrame()

    def split_data(self) -> tuple:
        """Splits the dataset into training and testing sets and applies scaling of choice to all other columns except one presented in categorical_variables"""
        # X is everything else but target variable
        X = self.df.drop(columns=[self.target_col])
        # y is only target variable. Don't usually need to be scaled
        y = self.df[self.target_col]
        def apply_scailing():
            """Applies scaling to X data"""
            # Parse for numerical columns only
            numerical_cols = X.select_dtypes(include=['number']).columns
            if self.categorical_variables is None:
                # Apply to every numerical column if categorical_variables is not provided
                cols_to_scale = [col for col in numerical_cols]
            else:
                # Leave categorical_variables as they are
                cols_to_scale = [col for col in numerical_cols if col not in self.categorical_variables]
            # Applying Scaler to selected columns
            X[cols_to_scale] = self.scaler_X.fit_transform(X[cols_to_scale])

        apply_scailing()
        # Perform train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=888)
        return X_train, X_test, y_train, y_test

    def get_default_models(self) -> dict:
        """To be implemented by subclasses. Should provide default dict with all supported models"""
        raise NotImplementedError

    def optimize(self, trial, models: dict) -> dict:
        """To be implemented by subclasses. Should provide dict with best hyperparameters for selected model
        Attributes
        ----------
        """
        raise NotImplementedError
    
    def train_and_evaluate(self) -> None:
        """Trains models one by one and evaluates them."""
        # Initialize fit parameters dict so code works for every model
        fit_params = {}
        for name, model in self.models.items():
            try:
                # Measure execution time
                start_time = time.time()
                logging.info(f"Training {name}...")

                # KNN algorithm need N-neighbors number.
                if name == "KNNRegressor" or name == "KNNClassifier":
                    def calculate_k_value(max_iter=70):
                        """Calculates best K value"""
                        error = []
                        # Calculating error for K values between 1 and 40
                        for i in range(1, max_iter):
                            # try with current k-value, train the model and make a test prediction
                            knn = model(n_neighbors=i, metric="minkowski")
                            knn.fit(self.X_train, self.y_train)
                            pred_i = knn.predict(self.X_test)
                            # save the error value for this k-value
                            if name == "KNNRegressor":
                                error.append(np.sqrt(metrics.mean_squared_error(self.y_test, pred_i)))
                            elif name == "KNNClassifier":
                                error.append(np.mean(pred_i != self.y_test))
                        # K value
                        k_value = np.argmin(error) + 1
                        return k_value
                    # Actual KNN model training
                    model = model(n_neighbors=calculate_k_value(), metric="minkowski")
                    
                # Turn off verbose.
                elif name == "LGBMRegressor" or name== "LGBMClassifier":
                    model = model(verbose=0)
                
                # CatBoost wants list of categorical variables
                elif name == "CatBoostRegressor" or name == "CatBoostClassifier":
                    model = model(verbose=0)
                    fit_params = {"cat_features" : self.categorical_variables}
                else:
                    model = model()

                model.fit(self.X_train, self.y_train, **fit_params)
                # Mesure time to complete training
                execution_time = (time.time() - start_time) * 1000  # Convert to ms
                # Save the model
                model_path = f"trained_models/{name}"
                joblib.dump(model, model_path)
                # Make predictions
                predictions = model.predict(self.X_test)
                
                # Get and save the metrics to the list
                metrics_row = self.evaluate_model(name, predictions, execution_time)
                # Reformat to pandas dataframe
                metrics_row = pd.DataFrame(metrics_row)
                metrics_row["Model"] = name
                # Append new row with metrics to the resulting dataframe
                self.results = pd.concat([self.results, metrics_row], ignore_index=True)    

            except Exception as e:
                logging.error(f"{name}: {e}")
                continue

    def optimization(self, direction: str):
        """Runs complete optimization pipeline
           Attributes
           ----------
           direction: str
                Supported are only maximize for classification problems and minimize for regression
        """
        def optimize_model(model_name):
            """Optimizes single model and appends metrics into self.results"""
            logging.info(f"Starting optimizing {model_name}...")
            
            def objective(trial):
                # Get possible parameters for model
                params = self.get_hyperparameters(trial=trial, model_name=model_name)
                # Unpack model and parameters
                model = self.models[model_name](**params)
                # Evaluate model
                score = self.mean_scores(model, self.X_train, self.y_train)
                return score  # Optuna will utilize this

            # Create optuna study with random name on local storage
            # Direction minimize for regression problems and maximize for calssification
            study = optuna.create_study(direction=direction) 
            # Turn off optuna verbosity
            optuna.logging.set_verbosity(optuna.logging.ERROR)
            # Time mesurement
            start_time = time.time()
            # Find best parameters
            study.optimize(objective, n_trials=30)
            best_params = study.best_params
            # Turn off verbose for catboost last training as well
            if model_name=="CatBoostRegressor" or model_name=="CatBoostClassifier":
                best_params["verbose"] = 0
            logging.info(f"Best params for {model_name}: {best_params}")
            # Train final model with best parameters
            final_model = self.models[model_name](**best_params)
            final_model.fit(self.X_train, self.y_train)  # Train on full training data
            execution_time = (time.time() - start_time) * 1000

            # Make predictions
            predictions = final_model.predict(self.X_test)
                
            # Get and save the metrics to the list
            metrics_row = self.evaluate_model(model_name, predictions, execution_time=execution_time)
            metrics_row = pd.DataFrame(metrics_row)
            model_name = "Optimized_" + model_name
            # Save the model
            model_path = f"trained_models/{model_name}"
            joblib.dump(final_model, model_path)
            # Update name in metrics row
            metrics_row["Model"] = model_name
            # Concatenate with the rest of metrics
            self.results = pd.concat([self.results, metrics_row], ignore_index=True)

        # Run for each model in models dict.  
        for model_name in self.models.keys():
            optimize_model(model_name)
    
    def evaluate_model(self, name: str, predictions, execution_time: int) -> dict:
        """
        To be implemented by subclasses.
        Gets the needed metrics based on if problem is classification or regression.
        
        Attibutes: 
        -----------
        name : str 
            Name of the model
        predictions
            Predictions of the trained model
        execution_time
            How long does it take to train the model
        """
        raise NotImplementedError
    
    def mean_scores(self, model, X_train, y_train):
        """Utilizes cros_val_score and returns mean value of scores.
        Attributes
        ----------
        model
            Actual model with parameters
        X_train
            Train X data
        y_train
            Train y data"""
        raise NotImplementedError
    
    def get_hyperparameters(self, trial, model_name: str):
        """Returns a dictionary with supported hyperparameters for the model in question.
        Attributes
        -----------
        trial
            Optuna required parameter
        model_name: str
            Name of the model. Should be in the list of suppoorted models"""
        raise NotImplementedError

    def benchmark(self, optimization_direction=None):
        """Runs the benchmarking process. First runs normal benchmarking, then tries to optimize same models in case if optimization direction provided.
        Attributes
        ----------
        optimization_direction
            Should be minimize for regression problems or maximize for classification. In case if not provided or invalid skips iptimization phase."""
        self.train_and_evaluate()
        if optimization_direction in ["maximize", "minimize"]:
            self.optimization(direction=optimization_direction)
        else:
            print("Optimization direction is not set or invalid. Skipping optimization phase...")
        print("Benchmarking completed.")

    def print_results(self):
        """"Prints out self.results"""
        print(self.results.to_markdown())


class RegressionBenchmark(BaseBenchmark):
    def get_default_models(self) -> dict:
        """Returns default regression models of such algorithms:
        Linear Regression, SVR, Random Forest, XGB, LGBM, KNN, CatBoost"""
        return {
            "LinearRegression": LinearRegression,
            "SVR": svm.SVR,
            "RandomForestRegressor": RandomForestRegressor,
            "XGBRegressor": XGBRegressor,
            "LGBMRegressor": LGBMRegressor,
            "KNNRegressor": KNeighborsRegressor,
            "CatBoostRegressor": CatBoostRegressor,
        }
    
    def get_hyperparameters(self, trial, model_name : str):
            """
            Gets hyperparameters of the model by model name in Optuna required format. Supported model names:
            LinearRegression, SVR, RandomForestRegressor, XGBRegressor, LGBMRegressor, KNNRegressor, CatBoostRegressor
            
            Attributes:
            ----------
            trial
                Optuna required
            model_name: str
                Name of the model. Should be in the list of suppoorted models
            """
            if model_name == "LinearRegression":
                return {}  # No hyperparameters to tune for basic LinearRegression

            elif model_name == "SVR":
                return {
                    "C": trial.suggest_float("C", 1e-2, 1e2, log=True),
                    "epsilon": trial.suggest_float("epsilon", 1e-4, 1e-1, log=True),
                    "kernel": trial.suggest_categorical("kernel", ["linear", "rbf", "sigmoid"]),
                }

            elif model_name == "RandomForestRegressor":
                return {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 20),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                }

            elif model_name == "XGBRegressor":
                return {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 15),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                }

            elif model_name == "LGBMRegressor":
                return {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "num_leaves": trial.suggest_int("num_leaves", 20, 100),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "verbose": -1
                }

            elif model_name == "KNNRegressor":
                return {
                    "n_neighbors": trial.suggest_int("n_neighbors", 1, 20),
                    "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                    "p": trial.suggest_int("p", 1, 2),
                }

            elif model_name == "CatBoostRegressor":
                return {
                    "iterations": trial.suggest_int("iterations", 50, 500),
                    "depth": trial.suggest_int("depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10, log=True),
                    "cat_features": self.categorical_variables,
                    "verbose": False
                }
        
    def mean_scores(self, model, X_train, y_train):
        """Utilizes cros_val_score and returns mean value of scores. Uses negative mean squared error.

        Attributes
        ----------
        model
            Actual model with parameters
        X_train
            Train X data
        y_train
            Train y data"""
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_squared_error")
        return np.mean(scores)  # Optuna minimizes, so MSE is negated
    
    def evaluate_model(self, name, predictions, execution_time = 0):
        """Computes regression metrics."""
        mse = metrics.mean_squared_error(self.y_test, predictions)
        rmse = np.sqrt(mse)
        mae = metrics.mean_absolute_error(self.y_test, predictions)
        r2 = metrics.r2_score(self.y_test, predictions)
        return {
            "Model": [name],
            "MSE": [round(mse, 4)],
            "RMSE": [round(rmse, 4)],
            "MAE": [round(mae, 4)],
            "R2": [round(r2, 4)],
            "Time (ms)": [round(execution_time, 2)],
        }


class ClassificationBenchmark(BaseBenchmark):
    def get_default_models(self):
        """Returns default classification models of such algorithms:
        Logistic Regression, SVC, Random Forest, XGB, LGBM, KNN, CatBoost"""

        return {
            "LogisticRegression": LogisticRegression,
            "SVC": svm.SVC,
            "RandomForestClassifier": RandomForestClassifier,
            "XGBClassifier": XGBClassifier,
            "LGBMClassifier": LGBMClassifier,
            "KNNClassifier": KNeighborsClassifier,
            "CatBoostClassifier": CatBoostClassifier,
        }

    def mean_scores(self, model, X_train, y_train):
        """Utilizes cros_val_score and returns mean value of scores. Uses accuracy.
        Attributes
        ----------
        model
            Actual model with parameters
        X_train
            Train X data
        y_train
            Train y data"""
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        return np.mean(scores)  # Higher accuracy is better
    
    def get_hyperparameters(self, trial, model_name : str):
        """
            Gets hyperparameters of the model by model name in Optuna required format. Supported model names:
            LogisticRegression, SVC, RandomForestClassifier, XGBClassifier, LGBMClassifier, KNNClassifier, CatBoostClassifier
            
            Attributes:
            ----------
            trial
                Optuna required
            model_name: str
                Name of the model. Should be in the list of suppoorted models
            """
        if model_name == "LogisticRegression":
            return {
                "C": trial.suggest_float("C", 1e-3, 10, log=True),
                "solver": trial.suggest_categorical("solver", ["liblinear", "lbfgs"]),
            }

        elif model_name == "SVC":
            return {
                "C": trial.suggest_float("C", 1e-2, 1e1, log=True),
                "kernel": trial.suggest_categorical("kernel", ["linear", "rbf", "sigmoid"]),
            }

        elif model_name == "RandomForestClassifier":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            }

        elif model_name == "XGBClassifier":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            }

        elif model_name == "LGBMClassifier":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "num_leaves": trial.suggest_int("num_leaves", 20, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "verbose": -1
            }

        elif model_name == "KNNClassifier":
            return {
                "n_neighbors": trial.suggest_int("n_neighbors", 1, 20),
                "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                "p": trial.suggest_int("p", 1, 2),
            }

        elif model_name == "CatBoostClassifier":
            return {
                "iterations": trial.suggest_int("iterations", 50, 500),
                "depth": trial.suggest_int("depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10, log=True),
                "cat_features": self.categorical_variables,
                "silent": True
            }
    
    def evaluate_model(self, name, predictions, execution_time=0):
        """Computes classification metrics."""
        report = classification_report(self.y_test, predictions, output_dict=True)

        return {
            "Model": [name],
            "Accuracy": [round(report["accuracy"], 4)],
            "Precision": [round(report["macro avg"]["precision"], 4)],
            "Recall": [round(report["macro avg"]["recall"], 4)],
            "F1-score": [round(report["macro avg"]["f1-score"], 4)],
            "Time (ms)": [round(execution_time, 2)],
        }

# TODO: PROVIDE BETTER PARAMETER GRIDS, add AdaBoost

if __name__=="__main__":
    print("Module is supposed to be imported")