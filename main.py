from chest_cancer_classifier.config.configuration import ConfigManager
from chest_cancer_classifier.pipeline.stage_01_data_ingestion import DataIngestionStage
from chest_cancer_classifier.pipeline.stage_02_base_model import PrepareBaseModelStage
from chest_cancer_classifier.pipeline.stage_03_training import TrainingStage
from chest_cancer_classifier.pipeline.stage_04_evaluation import EvaluationStage
from chest_cancer_classifier import logger

STAGE_NAME = "Data Ingestion Stage"
try:
        logger.info(f">>>>>>>>>>>>STAGE {STAGE_NAME} started <<<<<<<<<<<")
        obj = DataIngestionStage()
        obj.main()
        logger.info(f">>>>>>>>>>STAGE {STAGE_NAME} completed <<<<<<<<<<")
except Exception as e:
    raise e

STAGE_NAME = "BASE MODEL Stage"
try:
        logger.info(f">>>>>>>>>>>>STAGE {STAGE_NAME} started <<<<<<<<<<<")
        obj = PrepareBaseModelStage()
        obj.main()
        logger.info(f">>>>>>>>>>STAGE {STAGE_NAME} completed <<<<<<<<<<")
except Exception as e:
    raise e

STAGE_NAME = "Model Training Stage"
try:
        logger.info(f">>>>>>>>>>>>STAGE {STAGE_NAME} started <<<<<<<<<<<")
        obj = TrainingStage()
        obj.main()
        logger.info(f">>>>>>>>>>STAGE {STAGE_NAME} completed <<<<<<<<<<")
except Exception as e:
    raise e


STAGE_NAME = "Evaluation Stage"

try:
    logger.info(f">>>>>>>>>>>>STAGE {STAGE_NAME} started <<<<<<<<<<<")
    obj = EvaluationStage()
    obj.main()
    logger.info(f">>>>>>>>>>STAGE {STAGE_NAME} completed <<<<<<<<<<")
except Exception as e:
    raise e


