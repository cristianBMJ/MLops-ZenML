from zenml import pipeline

from steps.clean_data import clean_df
from steps.ingest_data import ingest_df
from steps.model_train import train_model
from steps.evaluation import evaluate_model



@pipeline(enable_cache=True) # caching to improve performance,reduce latency,and  avoiding repetitive computations or data retrievals.
def train_pipeline(data_path: str):
    df = ingest_df(data_path)
    clean_df(df)
    train_model(df)
    evaluate_model(df)