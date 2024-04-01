from rich import print
from  zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from  zenml.integrations.mlflow.model_deployers import MLFlowModelDeployer
from  zenml.integrations.mlflow.services import MLFlowDeploymentService

import click
from pipelines.deployment_pipelines import  continous_deployment_pipeline  #inference_pipeline



DEPLOY="deploy"
PREDICT="predict"
DEPLOY_AND_PREDICT="deploy_and_predict"

@click.command()
@click.option(
 "--config",
 "--c",
 type=click.Choice( [DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
 default=DEPLOY_AND_PREDICT,
 help="Optionally you can choose to only run the deployment"
 "pipeline to train and deploy model (`deploy`), or  to"
 "only run a prediction against the deployed model"
 "(`predict`). By default both will be run (`deploy_and_predict`)",
)

@click.option(
 "--min-accuracy",
 default=0.92,
 help="Minimum accuracy required to deploy the model"
)

def run_deployment( config: str, min_accuracy: float):
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    deploy = config == DEPLOY or config ==DEPLOY_AND_PREDICT
    predict = config == PREDICT or config ==DEPLOY_AND_PREDICT

    if deploy:
      continous_deployment_pipeline(
         data_path ="/workspaces/MLops-ZenML/data/olist_customers_dataset.csv",       
         min_accuracy=min_accuracy,
         workers = 3,
         timeout = 60)
    if predict:
      inference_pipeline()
    
    print( 
      "You can run: \n"
      f"[italic green] mlflow ui --backend-store-uri '{get_tracking_uri()}"
      "[italic green] \n ...to inspect  your experiment runs within mlflow"
      "UI. \n You can find runs tracked within the"
      "mlflow_example_pipeline experiment. There you'l also be able to "
      "compare two or more runs. \n\n"
    )



    # if existing_services:
     #  service = cast(MLFlowDeploymentService, existing_services[0])
      
if __name__ == "__main__":
  run_deployment()

