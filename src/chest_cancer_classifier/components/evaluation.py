from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
import tensorflow as tf
from chest_cancer_classifier.config.configuration import EvaluationConfig
from chest_cancer_classifier.utils.common import read_yaml,create_dir,save_json



class Evaluation:
    def __init__(self,config : EvaluationConfig):
        self.config = config
        
    def val_gen(self):
         
        datagenrator_kwargs = dict(
            rescale = 1./255,
            validation_split = 0.30
        )   
        
        dataflow_kwargs = dict(
            target_size=self.config.image_size_params[:-1],
            batch_size=self.config.batch_size_params,
            interpolation="bilinear"
        )
        
        valid_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenrator_kwargs
        )
        
        self.valid_gen = valid_data_gen.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )
        
    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(r"C:\Users\aditi\OneDrive\Desktop\Chest-Cancer-ML-OPS\artifacts\training\model.h5")
    
    
    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self.val_gen()
        self.score = self.model.evaluate(self.valid_gen)
        self.save_score()
        
    def save_score(self):
        scores = {"loss" : self.score[0] , "accuracy" : self.score[1]}
        save_json(path=Path("scores.json"),data = scores)
        
    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_url)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss" : self.score[0] , "accuracy" : self.score[1]}
            )
            
            if tracking_url_type_store != "file":
                mlflow.keras.log_model(self.model,"model",registered_model_name="VGG16Model")
            else:
                mlflow.keras.log_model(self.model,"model")
        
        
        