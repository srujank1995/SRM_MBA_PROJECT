from src.pipeline.train_pipeline import TrainPipeline

if __name__ == "__main__":

    obj = TrainPipeline()
    obj.run_pipeline()

    print("Model training completed")