from collections import namedtuple
from typing import NamedTuple
from unicodedata import name



DataIngestionConfig=namedtuple('DataIngestionConfig',["dataset_download_url","tgz_download_dir","raw_data_dir","ingested_train_dir","ingested_test_dir"])



DataValidationConfig=namedtuple('DataValidationConfig',["schema_file_path",'report_file_path','report_page_file_path'])


DataTransformationConfig=namedtuple('DataTransformationConfig',["add_bedroom_per_room","transformed_train_dir","transformed_test_dir","preprocessed_object_file_path"])
#preprocessed_object_file_path is the file path of pickle file of the transformed data

ModelTrainerConfig=namedtuple('ModelTrainerConfig',["trained_model_file_path","base_accuracy"])
#base_accuracy is the minimum expected accuracy


ModelEvaluationConfig=namedtuple('ModelEvaluationConfig',["model_evaluation_file_path","time_stamp"])
#model_evaluation_file_path is the path where the previously created models are stored to compare with new model for validation (will also evaluated using test data)


ModelPusherConfig=namedtuple('ModelPusherConfig',["export_dir_path"])


TrainingPipelineConfig=namedtuple('TrainingPipelineConfig',["artifact_dir"])