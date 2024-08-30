from chest_cancer_classifier.constants import *
from chest_cancer_classifier.utils.common import read_yaml,create_dir
from chest_cancer_classifier.entity.config_entity import DataIngestionConfig

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