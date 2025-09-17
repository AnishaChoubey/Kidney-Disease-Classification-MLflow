# Kidney-Disease-Classification-MLflow

## Workflows

1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the dvc.yaml
10. app.py

# How to run?
### STEPS:

### STEP 01- Create a virtual environment after opening the repository
```bash
python -v venv  Kidney python=3.11.3 -y
```
### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```

##### cmd
- mlflow ui

### dagshub
[dagshub](https://dagshub.com/)

MLFLOW_TRACKING_URI=(https://dagshub.com/AnishaChoubey/Kidney-Disease-Classification-MLflow.mlflow)
MLFLOW_TRACKING_USERNAME=AnishaChoubey
MLFLOW_TRACKING_PASSWORD=2254846308235d831836322ec845966deac15585
python script.py

Run this to export as env variables:

```bash

export MLFLOW_TRACKING_URI=https://dagshub.com/AnishaChoubey/Kidney-Disease-Classification-MLflow.mlflow
export MLFLOW_TRACKING_USERNAME=AnishaChoubey 

export MLFLOW_TRACKING_PASSWORD=2254846308235d831836322ec845966deac15585

```
### DVC cmd

1. dvc init
2. dvc repro
3. dvc dag


## About MLflow & DVC

MLflow

 - Its Production Grade
 - Trace all of your expriements
 - Logging & taging your model


DVC 

 - Its very lite weight for POC only
 - lite weight expriements tracker
 - It can perform Orchestration (Creating Pipelines)




projectlink: https://huggingface.co/spaces/AnishaChoubey/kidney-disease-classification
