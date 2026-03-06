import sys
import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.utils import save_object


class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")


class DataTransformation:

    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):

        categorical_columns=["Country"]

        preprocessor=ColumnTransformer(
        [
        ("OneHotEncoder",OneHotEncoder(),categorical_columns)
        ])

        return preprocessor


    def initiate_data_transformation(self,train_path,test_path):

        try:

            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            target_column="Quantity"

            X_train=train_df.drop(columns=[target_column])
            y_train=train_df[target_column]

            X_test=test_df.drop(columns=[target_column])
            y_test=test_df[target_column]

            preprocessing_obj=self.get_data_transformer_object()

            X_train_arr=preprocessing_obj.fit_transform(X_train)
            X_test_arr=preprocessing_obj.transform(X_test)

            save_object(
            file_path=self.data_transformation_config.preprocessor_obj_file_path,
            obj=preprocessing_obj
            )

            return (
            X_train_arr,
            X_test_arr,
            y_train,
            y_test
            )

        except Exception as e:
            raise CustomException(e,sys)