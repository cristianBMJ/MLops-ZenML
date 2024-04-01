from pipelines.training_pipeline import train_pipeline
from zenml.client import Client


if __name__ == "__main__":
    # run the pipeline .
    print( Client().active_stack.experiment_tracker.get_tracking_uri() )
    train_pipeline( data_path="/workspaces/MLops-ZenML/data/olist_customers_dataset.csv")

# mlflow ui --backend-store-uri "file:/home/codespace/.config/zenml/local_stores/29847686-a44a-4851-b931-78b37a6596ab/mlruns"