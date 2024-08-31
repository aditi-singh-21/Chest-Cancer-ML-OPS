from chest_cancer_classifier.config.configuration import ConfigManager
from chest_cancer_classifier.pipeline.stage_02_base_model import PrepareBaseModelStage
from chest_cancer_classifier import logger

STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>>>>>>>>>>STAGE {STAGE_NAME} started <<<<<<<<<<<")
    obj = PrepareBaseModelStage()
    obj.main()
    logger.info(f">>>>>>>>>>STAGE {STAGE_NAME} completed <<<<<<<<<<")
except Exception as e:
    raise e


