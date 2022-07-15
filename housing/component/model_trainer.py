from housing.entity.model_factory import ModelFactory
from housing.logger import logging
from housing.exception import HousingException
import os,sys
from housing.entity.artifact_entitiy import DataTransformationArtifact,ModelTrainerArtifact
from housing.entity.config_entity import *
from housing.util.util import load_data,load_obj
from housing.config.configuration import Configuration










class ModelTrainer:
    def __init__(self,data_transformation_artifact:DataTransformationArtifact,model_trainer_config:ModelTrainerConfig) -> None:
        try:
            logging.info(f'{"="*30}Model trainer log started. {"="*30}')
            self.data_transformation_artifact=data_transformation_artifact
            self.model_trainer_config=model_trainer_config
        except Exception as e:
            raise HousingException(e,sys) from e 

    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            logging.info('loading transformed trainig dataset')
            transformed_train_file_path=self.data_transformation_artifact.transformed_train_file_path
            train_array=load_data(file_path=transformed_train_file_path)

            logging.info('loading transformed testing dataset')
            transformed_test_file_path=self.data_transformation_artifact.transformed_test_file_path
            test_array=load_data(file_path=transformed_test_file_path)

            logging.info('splitting train and test input and target feature')
            x_train,y_train,x_test,y_test=train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1]

            logging.info('extracting model config file path')
            model_config_file_path = self.model_trainer_config.model_config_file_path

            logging.info(f'initiating model factory class using above model config file:{model_config_file_path}')
            model_factory=ModelFactory(model_config_path=model_config_file_path)

            base_accuracy=self.model_trainer_config.base_accuracy
            logging.info(f'excpected accuracy:{base_accuracy}')

            logging.info(f'initiating operation model selection')
            best_model=model_factory.get_best_model(X=x_train,y=y_train,base_accuracy=base_accuracy)

            model_trainer_artifact=  ModelTrainerArtifact(is_trained=True,message="Model Trained successfully",
            trained_model_file_path=trained_model_file_path,
            train_rmse=metric_info.train_rmse,
            test_rmse=metric_info.test_rmse,
            train_accuracy=metric_info.train_accuracy,
            test_accuracy=metric_info.test_accuracy,
            model_accuracy=metric_info.model_accuracy
            
            )

            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact

            
        except Exception as e:
            raise HousingException(e,sys) from e
        
