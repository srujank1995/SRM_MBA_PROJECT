import pandas as pd
from src.utils import load_object


class PredictPipeline:

    def predict(self,features):

        model=load_object("artifacts/model.pkl")

        preprocessor=load_object("artifacts/preprocessor.pkl")

        data_scaled=preprocessor.transform(features)

        preds=model.predict(data_scaled)

        return preds