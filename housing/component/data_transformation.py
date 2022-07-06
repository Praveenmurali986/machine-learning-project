from logging import exception
from multiprocessing import Pipe

from sklearn import preprocessing
from housing.logger import logging
from housing.exception import HousingException
from housing.entity.config_entity import DataTransformationConfig
from housing.entity.artifact_entitiy import DataIngestionArtifact
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
from housing.util.util import read_yaml_file





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
            self.data_transformation_config=data_transformation_config
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_validation_artifact=data_validation_artifact
        except Exception as e:
            raise HousingException(e,sys) from e

    @staticmethod
    def load_data(file_path:str,schema_file_path:str)->pd.DataFrame:
        try:
            dataset_schema=read_yaml_file(schema_file_path)

            schema=dataset_schema[DATASET_SCHEMA_COLUMNS_KEY]

            dataframe=pd.read_csv(file_path)

            error_message=""

            for column in dataframe.columns:
                if column in list(schema.keys()):
                    dataframe[column].astype(schema[column])
                else:
                    error_message=f"{error_message} \ncolumn : {column} is not in schema"
            if len(error_message)>0:
                raise Exception(error_message)
            return dataframe

        except Exception as e:
            raise HousingException(e,sys) from e

    def get_data_transformer_object(self)->ColumnTransformer:
        try:
            schema_file_path=self.data_validation_artifact.schema_file_path
            dataset_schema=read_yaml_file(schema_file_path)

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


            