import logging
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import metrics, svm
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import StackingClassifier, StackingRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Stacking:
    def __init__(self, models: list, problem: str, meta_model):
        """ Initializes class.
        Attributes
        ----------
        models: list
            List of models in format [ ('model_name', model_callable(params)) ]
        problem: str
            Kind of problem: regression or classification
        meta_model
            Model judge. Callable"""
        self.models = models
        self.meta_model = meta_model
        self.classification = False
        self.regression = False
        if problem.lower() == "regression":
            self.regression = True
        elif problem.lower() == "classification":
            self.classification = True
        else:
            logging.error("Problem must be either regression or classification")
            raise ValueError
        self.stacking_model = None
        
    def create_stacking_model(self):
        if self.regression:
            stacking_estimator = StackingRegressor
        else:
            stacking_estimator = StackingClassifier

        stacking_model = stacking_estimator(estimators=self.models, 
                                            final_estimator = self.meta_model,
                                            cv=5)
        return stacking_model
        
    def fit_stacking_model(self, X_train, y_train):
        self.stacking_model = self.create_stacking_model()
        self.stacking_model.fit(X_train, y_train)

    def evaluate_stacking_model(self, X_test, y_test):
        if self.classification:
            accuracy = self.stacking_model.score(X_test, y_test)
            print(f"Accuracy :{round(accuracy, 4)}")
        elif self.regression:
            predictions = self.stacking_model.predict(X_test)
            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, predictions)
            print()
            print(f"MAE: {mae}")
            print(f"MSE: {mse}")
            print(f"RMSE: {rmse}")
            print(f"R2: {r2}")
        

if __name__=="__main__":
    print("Module is supposed to be imported")