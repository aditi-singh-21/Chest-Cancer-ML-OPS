from chest_cancer_classifier.constants import *
from chest_cancer_classifier.utils.common import read_yaml,create_dir
from chest_cancer_classifier.entity.config_entity import DataIngestionConfig
from chest_cancer_classifier.entity.config_entity import PrepareBaseModelConfig
from pathlib import Path

class ConfigManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        param_filepath = PARAMS_FILE_PATH):
        print(type(config_filepath))
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(param_filepath)
        
        create_dir([self.config.artifacts_root])
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        
        create_dir([config.root_dir])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir = config.root_dir,
            source_url = config.source_url,
            local_data_file = config.local_data_file,
            unzip_dir = config.unzip_dir
        )
        
        return data_ingestion_config
    
    def prepare_base_model(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        create_dir([config.root_dir])
        
        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir= Path(config.root_dir),
            base_model_path = Path(config.base_model_path),
            updated_base_model_path = Path(config.updated_base_model_path),
            image_size_params = self.params.IMAGE_SIZE,
            learning_rate_params=self.params.LEARNING_RATE,
            include_top_params=self.params.INCLUDE_TOP,
            weights_params=self.params.WEIGHTS,
            classes_params=self.params.CLASSES 
        )
        return prepare_base_model_config