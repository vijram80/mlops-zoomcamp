if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter
import mlflow
import joblib

@data_exporter
def export_data(data, *args, **kwargs):
    # Specify your data exporting logic here
    EXPERIMENT_NAME = "week3"
    mlflow.set_tracking_uri("http://mlflow:5000")
    #mlflow.create_experiment(EXPERIMENT_NAME, artifact_location="artifacts-local")
    mlflow.set_experiment(EXPERIMENT_NAME)

    dv = data[0]
    lr = data[1]
    rmse = data[2]

    
    with mlflow.start_run():
        mlflow.set_tag("developer", "vijay")
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(lr, "linear regression")
        
        vectorizer_path = "dict_vectorizer.pkl"
        joblib.dump(dv, vectorizer_path)
        mlflow.log_artifact(vectorizer_path, "vectorizer")


