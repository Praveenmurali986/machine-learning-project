from housing.entity.config_entity import DataValidationConfig
from housing.entity.artifact_entitiy import DataIngestionArtifact
from housing.entity.artifact_entitiy import DataValidationArtifact
from housing.logger import logging
from housing.exception import HousingException
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab
import json
import os,sys
import pandas as pd




class DataValidation:
    def __init__(self,data_validation_config:DataValidationConfig,data_ingestion_artifact:DataIngestionArtifact):
        try:
            self.data_validation_config=data_validation_config
            self.data_ingestion_artifact=data_ingestion_artifact
        except Exception as e:
            raise HousingException(e,sys) from e

    def get_train_and_test_dataframe(self):
        try:
            train_df=pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df=pd.read_csv(self.data_ingestion_artifact.test_file_path)

            return train_df,test_df
        except Exception as e:
            raise HousingException(e,sys) from e


    def is_train_test_file_exists(self)->bool:
        try:
            logging.info('checking if training and tesing file is available')
            is_train_file_exist=False
            is_test_file_exist=False

            train_file_path=self.data_ingestion_artifact.train_file_path
            test_file_path=self.data_ingestion_artifact.test_file_path

            is_train_file_exist=os.path.exists(train_file_path)
            is_test_file_exist=os.path.exists(test_file_path)
            is_available= is_train_file_exist and is_test_file_exist 

            logging.info(f'is train and test file available? : {is_available}')

            if not is_available:
                
                message=f'training file :{train_file_path} or testing file :{test_file_path}' \
                    'is not available'
                raise Exception(message)

            return is_available
        except Exception as e:
            raise HousingException(e,sys) from e

        
    def validate_dataset_schema(self)->bool:
        try:
            #no of col
            #values of ocean proximity
            #col names
            validation_status=True
            return validation_status
        except Exception as e:
            raise HousingException(e,sys) from e


    def get_and_save_data_drift_report(self):
        try:
            profile=Profile(sections=[DataDriftProfileSection()])

            train_df,test_df=self.get_train_and_test_dataframe()

            profile.calculate(train_df,test_df)
            report = json.loads(profile.json())

            report_file_path=self.data_validation_config.report_file_path
            report_file_dir=os.path.dirname(report_file_path)

            os.makedirs(report_file_dir,exist_ok=True)

            with open(report_file_path,'w') as report_file:
                json.dump(report,report_file,indent=6)

            return report 

        except Exception as e:
            raise HousingException(e,sys) from e


    def save_data_drift_report_page(self):
        try:
            train_df,test_df=self.get_train_and_test_dataframe()

            dashboard=Dashboard(tabs=[DataDriftTab()])
            dashboard.calculate(train_df,test_df)

            report_page_file_path=self.data_validation_config.report_page_file_path
            report_page_file_dir=os.path.dirname(report_page_file_path)
            os.makedirs(report_page_file_dir,exist_ok=True)

            dashboard.save(report_page_file_path)

        except Exception as e:
            raise HousingException(e,sys) from e


    def is_data_drift_found(self):
        try:
            report =self.get_and_save_data_drift_report()
            self.save_data_drift_report_page()
            return True

        except Exception as e:
            raise HousingException(e,sys) from e


    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            self.is_train_test_file_exists()
            self.validate_dataset_schema()
            self.is_data_drift_found()
            data_validation_artifact=DataValidationArtifact(
                schema_file_path=self.data_validation_config.schema_file_path,
                report_file_path=self.data_validation_config.report_file_path,
                report_page_file_path=self.data_validation_config.report_page_file_path,
                is_validated=True,
                message='data validated succesfully.'
            )
            logging.info(f'data validation artifact :{data_validation_artifact}')

            return data_validation_artifact

            

        except Exception as e:
            raise HousingException(e,sys) from e

    def __del__(self):
        logging.info(f"{'>>'*30}Data Valdaition log completed.{'<<'*30} \n\n")
