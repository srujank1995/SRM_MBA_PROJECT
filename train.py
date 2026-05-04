"""Training script for AI-Based Product Demand Forecasting System."""
import sys
import argparse
from src.pipeline.train_pipeline import TrainPipeline
from src.logger import logging
from src.exception import CustomException


def main():
    parser = argparse.ArgumentParser(description="Train demand forecasting models")
    parser.add_argument("--mode", type=str, default="legacy",
                        choices=["legacy", "advanced"],
                        help="Training mode: 'legacy' (CatBoost/RF/LR) or 'advanced' (all models)")
    parser.add_argument("--data", type=str, default="notebook/data/data.csv",
                        help="Path to training data CSV")
    args = parser.parse_args()

    try:
        pipeline = TrainPipeline(data_path=args.data)

        if args.mode == "advanced":
            logging.info("Starting advanced training pipeline")
            results = pipeline.run_advanced_pipeline()
            print("\n=== Model Training Complete ===")
            print(f"Best Model: {results['best_model']}")
            print("\nModel Comparison:")
            print(results["comparison"])
        else:
            logging.info("Starting legacy training pipeline")
            score = pipeline.run_pipeline()
            print(f"\nModel training completed. Best R² score: {score:.4f}")

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()