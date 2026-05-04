"""Prediction script for demand forecasting."""
import sys
import argparse
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline
from src.logger import logging
from src.exception import CustomException


def main():
    parser = argparse.ArgumentParser(description="Generate demand forecasts")
    parser.add_argument("--country", type=str, default="United Kingdom",
                        help="Country for prediction")
    parser.add_argument("--output", type=str, default="predictions.csv",
                        help="Output CSV file path")
    args = parser.parse_args()

    try:
        logging.info(f"Generating prediction for country: {args.country}")
        data = pd.DataFrame({"Country": [args.country]})
        pipeline = PredictPipeline()
        result = pipeline.predict(data)
        print(f"Predicted Quantity for {args.country}: {result[0]:.2f}")

        pd.DataFrame({
            "Country": [args.country],
            "PredictedQuantity": [round(result[0], 2)],
        }).to_csv(args.output, index=False)
        print(f"Saved predictions to {args.output}")

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()
