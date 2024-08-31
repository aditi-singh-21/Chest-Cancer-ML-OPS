from chest_cancer_classifier.config.configuration import ConfigManager
from chest_cancer_classifier.components.training import Training
from chest_cancer_classifier import logger

STAGE_NAME = "Model Training Stage"

class TrainingStage():
    def __init__(self):
        pass
    
    def main(self):
        config= ConfigManager()
        training_config = config.train_model_config()
        training =Training(config = training_config)
        training.get_base_model()
        training.train_valid_gen()
        training.train()
        

if __name__ == '__main__':
    try:
        logger.info(f">>>>>>>>>>>>STAGE {STAGE_NAME} started <<<<<<<<<<<")
        obj = TrainingStage()
        obj.main()
        logger.info(f">>>>>>>>>>STAGE {STAGE_NAME} completed <<<<<<<<<<")
    except Exception as e:
        raise e