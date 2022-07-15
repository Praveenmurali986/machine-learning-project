import shutil
from housing.logger import logging
from housing.exception import HousingException
import os,sys
from housing.entity.artifact_entitiy import ModelEvaluationArtifact,ModelPusherArtifact
from housing.entity.config_entity import ModelPusherConfig
from housing.constant import *






class ModelPusher:
    def __init__(
        self,
        model_evaluation_artifact:ModelEvaluationArtifact,
        model_pusher_config:ModelPusherConfig
    ) -> None:
        try:
            logging.info(f'{"="*20} model pusher log started. {"="*20}')
            self.model_evaluation_artifact=model_evaluation_artifact
            self.model_pusher_config=model_pusher_config
        except Exception as e:
            raise HousingException(e,sys) from e

    def export_model(self):
        try:
            evaluated_model_file_path=self.model_evaluation_artifact.evaluated_model_path
            export_dir=self.model_pusher_config.export_dir_path
            model_file_name=os.path.basename(evaluated_model_file_path)
            export_model_file_path=os.path.join(
                export_dir,
                model_file_name
            )
            logging.info(f'exporting model file: {export_model_file_path}')
            os.makedirs(export_dir,exist_ok=True)


            shutil.copy(src=evaluated_model_file_path,dst=export_model_file_path)
            logging.info(f'trained model:{evaluated_model_file_path} copied to export dir: [{export_model_file_path}]')

            model_pusher_artifact=ModelPusherArtifact(
                export_model_file_path=export_model_file_path,
                is_model_pusher=True
            )
            logging.info(f'model pusher artifact : {ModelPusherArtifact}')
            return model_pusher_artifact

        except Exception as e:
            raise HousingException(e,sys) from e

    def __del__(self):
        logging.info(f'{"="*20} model pusher log completed.{"="*20}')