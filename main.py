from chest_cancer_classifier import logger
from chest_cancer_classifier.pipeline.stage_01_data_ingestion import DataIngestionStage

STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>>>>>>>>>>STAGE {STAGE_NAME} started <<<<<<<<<<<")
    obj = DataIngestionStage()
    obj.main()
    logger.info(f">>>>>>>>>>STAGE {STAGE_NAME} completed <<<<<<<<<<")
except Exception as e:
    raise e


