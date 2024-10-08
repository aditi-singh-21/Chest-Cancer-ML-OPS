from chest_cancer_classifier.config.configuration import ConfigManager
from chest_cancer_classifier.components.data_ingestion import DataIngestion
from chest_cancer_classifier import logger

STAGE_NAME = "Data Ingestion Stage"

class DataIngestionStage:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config = data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()
        

if __name__ == '__main__':
    try:
        logger.info(f">>>>>>>>>>>>STAGE {STAGE_NAME} started <<<<<<<<<<<")
        obj = DataIngestionStage()
        obj.main()
        logger.info(f">>>>>>>>>>STAGE {STAGE_NAME} completed <<<<<<<<<<")
    except Exception as e:
        raise e