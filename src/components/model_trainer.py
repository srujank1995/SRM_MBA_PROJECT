import os
import sys

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.utils import evaluate_models
from src.utils import save_object


class ModelTrainerConfig:

    trained_model_file_path=os.path.join("artifacts","model.pkl")


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,X_train,X_test,y_train,y_test):

        try:

            models={
            "RandomForest":RandomForestRegressor(),
            "LinearRegression":LinearRegression(),
            "CatBoost":CatBoostRegressor(verbose=False)
            }

            model_report=evaluate_models(X_train,y_train,X_test,y_test,models)

            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]

            best_model.fit(X_train,y_train)

            save_object(
            file_path=self.model_trainer_config.trained_model_file_path,
            obj=best_model
            )

            return best_model_score

        except Exception as e:
            raise CustomException(e,sys)