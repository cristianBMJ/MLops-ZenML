from pipelines.training_pipeline import train_pipeline

if __name__ == "__main__":
    # run the pipeline
    train_pipeline( data_path="/workspaces/MLops-ZenML/data/olist_customers_dataset.csv")

