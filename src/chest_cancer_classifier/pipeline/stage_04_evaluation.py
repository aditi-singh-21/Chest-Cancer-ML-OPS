from chest_cancer_classifier.config.configuration import ConfigManager
from chest_cancer_classifier.components.evaluation import Evaluation
from chest_cancer_classifier import logger

STAGE_NAME = "Evaluation Stage"

class EvaluationStage():
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigManager()
        eval_config = config.get_evaluation_config()
        eval=Evaluation(eval_config)
        eval.evaluation()
        eval.log_into_mlflow()
        
        

if __name__ == '__main__':
    try:
        logger.info(f">>>>>>>>>>>>STAGE {STAGE_NAME} started <<<<<<<<<<<")
        obj = EvaluationStage()
        obj.main()
        logger.info(f">>>>>>>>>>STAGE {STAGE_NAME} completed <<<<<<<<<<")
    except Exception as e:
        raise e