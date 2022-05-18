from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from memoized_property import memoized_property
import mlflow
from  mlflow.tracking import MlflowClient
import joblib
from sklearn.model_selection import train_test_split
from google.cloud import storage

STORAGE_LOCATION = 'models/taxifaremodel/model.joblib'
BUCKET_NAME = "wagon-data-847-cameronpesant"

class Trainer(object):

    MLFLOW_URI = "https://mlflow.lewagon.ai/"
    EXPERIMENT_NAME = "[CA] [Montreal] [scameronp] taxifare + 1.4"

    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.experiment_name = Trainer.EXPERIMENT_NAME
        self.MLFLOW_URI = Trainer.MLFLOW_URI


    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        pipeline = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])
        self.pipeline = pipeline

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse

    def predict(self, X_test):
        """predict y_pred"""
        y_pred = self.pipeline.predict(X_test)
        return y_pred


    def save_model(self):
        """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
        HINTS : use joblib library and google-cloud-storage"""

        # saving the trained model to disk is mandatory to then beeing able to upload it to storage
        # Implement here
        joblib.dump(self.pipeline, 'model.joblib')
        print("saved model.joblib locally")

        # Implement here
        self.upload_model_to_gcp()
        print(f"uploaded model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}")


    def upload_model_to_gcp(self):
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(STORAGE_LOCATION)
        blob.upload_from_filename('model.joblib')


    # MLFlow methods
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(self.MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


if __name__ == "__main__":
    # Get and clean data
    N = 10000
    df = get_data(nrows=N)
    df = clean_data(df)
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # Train and save model, locally and
    trainer = Trainer(X=X_train, y=y_train)
    # trainer.set_experiment_name('xp2')
    trainer.run()
    rmse = trainer.evaluate(X_test, y_test)
    print(f"rmse: {rmse}")
    trainer.save_model()
    trainer.mlflow_log_param("model", "LinearRegression")
    trainer.mlflow_log_metric("rmse", rmse)
    experiment_id = trainer.mlflow_experiment_id
    print(f"experiment URL: https://mlflow.lewagon.ai/#/experiments/{experiment_id}")
