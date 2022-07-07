

from sklearn import preprocessing
from housing.logger import logging
from housing.exception import HousingException
from housing.entity.config_entity import DataTransformationConfig
from housing.entity.artifact_entitiy import DataIngestionArtifact, DataTransformationArtifact
from housing.entity.artifact_entitiy import DataValidationArtifact
import os,sys
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from housing.constant import *
import pandas as pd
from housing.util.util import read_yaml_file,save_numpy_array_data,save_object,load_data





class FeatureGenerator(BaseEstimator,TransformerMixin):
    def __init__(
        self,
        add_bedroom_per_rooms=True,
        total_bedrooms_ix=4,
        total_rooms_ix=3,
        households_ix=6,
        population_ix=5,
        columns=None
    ) -> None:
        
        try:
            self.columns=columns
            if self.columns is not None:
                self.total_bedrooms_ix=self.columns.index(COLUMN_TOTAL_BEDROOMS)
                self.total_rooms_ix=self.columns.index(COLUMN_TOTAL_ROOMS)
                self.households_ix=self.columns.index(COLUMN_HOUSEHOLDS)
                self.population_ix=self.columns.index(COLUMN_POPULATION)

            self.add_bedrooms_per_room=add_bedroom_per_rooms
            self.total_bedrooms_ix=total_bedrooms_ix
            self.total_rooms_ix=total_rooms_ix
            self.households_ix=households_ix
            self.population_ix=population_ix
    
        except Exception as e:
            raise HousingException(e,sys) from e

    def fit(self,X,y=None):
        return self

    def transform(self,X,y=None):
        try:
            room_per_household=X[:,self.total_rooms_ix]/X[:,self.households_ix]

            population_per_household=X[:,self.population_ix]/X[:,self.households_ix]

            if self.add_bedrooms_per_room:
                bed_room_per_rooms=X[:,self.total_bedrooms_ix]/X[:,self.total_rooms_ix]

                generated_feature=np.c_[
                    X,
                    room_per_household,
                    population_per_household,
                    bed_room_per_rooms
                ]
            else:
                generated_feature=np.c_[
                    X,
                    room_per_household,
                    population_per_household
                ]
            
            return generated_feature

        except Exception as e:
            raise HousingException(e,sys) from e





class DataTransformation:
    def __init__(
        self,
        data_transformation_config:DataTransformationConfig,
        data_ingestion_artifact:DataIngestionArtifact,
        data_validation_artifact:DataValidationArtifact
    ) -> None:
        try:
            logging.info(f'{"="*20}data transformation log started {"="*20}')
            self.data_transformation_config=data_transformation_config
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_validation_artifact=data_validation_artifact
        except Exception as e:
            raise HousingException(e,sys) from e

    

    def get_data_transformer_object(self)->ColumnTransformer:
        try:
            schema_file_path=self.data_validation_artifact.schema_file_path
            dataset_schema=read_yaml_file(file_path=schema_file_path)

            numerical_columns=dataset_schema[NUMERICAL_COLUMN_KEY]
            catagorical_columns=dataset_schema[CATEGORICAL_COLUMN_KEY]

            num_pipeline=Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('feature_generator',FeatureGenerator(
                    add_bedroom_per_rooms=self.data_transformation_config.add_bedroom_per_room,
                    columns=numerical_columns
                )),
                ('scaler',StandardScaler())
            ])

            cat_pipeline=Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder',OneHotEncoder()),
                ('scaler',StandardScaler(with_mean=False))
            ])

            logging.info(f'numerical columns : {numerical_columns}')
            logging.info(f'catagorical columns : {catagorical_columns}')

            preprocessing=ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_columns),
                ('cat_pipeline',cat_pipeline,catagorical_columns)
            ])
            return preprocessing


        except Exception as e:
            raise HousingException(e,sys) from e
    
    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:

            logging.info(f'data preprocessing object')
            preprocessing_obj=self.get_data_transformer_object()

            logging.info(f'obtaining training and test file path.')

            train_file_path=self.data_ingestion_artifact.train_file_path
            test_file_path=self.data_ingestion_artifact.test_file_path
            schema_file_path=self.data_validation_artifact.schema_file_path

            logging.info('loading test and train data as pd.dataframe')

            train_df=load_data(file_path=train_file_path,schema_file_path=schema_file_path)
            test_df=load_data(file_path=test_file_path,schema_file_path=schema_file_path)

            schema=read_yaml_file(schema_file_path)

            target_column_name=schema[TARGET_COLUMN_KEY]

            logging.info('spliting input and target feature from trainig and testing dataframe')

            input_feature_train_df=train_df.drop(column=[target_column_name],axis=1)
            target_feature_train_df=train_df[[target_column_name]]

            input_feature_test_df=test_df.drop(column=[target_column_name],axis=1)
            target_feature_test_df=test_df[[target_column_name]]

            logging.info('applying preprocessing object on train and test dataframe')

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_(input_feature_train_arr,np.array(target_feature_train_df))
            test_arr=np.c_(input_feature_test_arr,np.array(target_feature_test_df))

            transformed_train_dir=self.data_transformation_config.transformed_train_dir
            transformed_test_dir=self.data_transformation_config.transformed_test_dir


            train_file_name=os.path.basename(train_file_path).replace('.csv','.npz')
            test_file_name=os.path.basename(test_file_path).replace('.csv','.npz')

            transformed_test_file_path=os.path.join(transformed_test_dir,test_file_name)
            transformed_train_file_path=os.path.join(transformed_train_dir,train_file_name)

            logging.info('saving transformed training and testing array')

            save_numpy_array_data(file_path=transformed_train_file_path,array=train_arr)
            save_numpy_array_data(file_path=transformed_test_file_path,array=test_arr)

            preprocessing_obj_file_path=self.data_transformation_config.preprocessed_object_file_path

            logging.info('saving preprocessing object')

            save_object(file_path=preprocessing_obj_file_path,obj=preprocessing_obj)

            data_transformation_artifact=DataTransformationArtifact(
                is_transformed=True,
                message='data transformation completed',
                transformed_test_file_path=transformed_test_file_path,
                transformed_train_file_path=transformed_train_file_path,
                preprocessed_object_file_path=preprocessing_obj_file_path

            )
            
            logging.info(f'data transformation artifact : {DataTransformationArtifact}')

            return data_transformation_artifact

        except Exception as e:
            raise HousingException(e,sys) from e
    