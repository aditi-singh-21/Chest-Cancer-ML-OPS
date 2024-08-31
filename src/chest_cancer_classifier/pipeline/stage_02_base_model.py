from chest_cancer_classifier.config.configuration import ConfigManager
from chest_cancer_classifier.components.prepare_base_model import PrepareBaseModel
from chest_cancer_classifier import logger

STAGE_NAME = "BASE MODEL Stage"

class PrepareBaseModelStage():
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigManager()
        prepare_base_model_config = config.prepare_base_model()
        prepare_base_model = PrepareBaseModel(config = prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()
        

if __name__ == '__main__':
    try:
        logger.info(f">>>>>>>>>>>>STAGE {STAGE_NAME} started <<<<<<<<<<<")
        obj = PrepareBaseModelStage()
        obj.main()
        logger.info(f">>>>>>>>>>STAGE {STAGE_NAME} completed <<<<<<<<<<")
    except Exception as e:
        raise e