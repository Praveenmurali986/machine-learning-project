import os,sys
import yaml
from housing.exception import HousingException
import numpy as np
import dill
import pandas as pd
from housing.constant import *




def read_yaml_file(file_path)->dict:
    '''
    Read yaml file and return as dict 
    '''
    try:
        with open(file_path,'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise HousingException(e,sys) from e


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


def save_numpy_array_data(file_path:str,array:np.array):
    try:
        dir_name=os.path.dirname(file_path)
        os.makedirs(dir_name,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            np.save(file_obj,array)
    except Exception as e:
            raise HousingException(e,sys) from e


def save_object(file_path:str,obj):
    try:
        path_dir=os.path.dirname(file_path)
        os.makedirs(path_dir,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise HousingException(e,sys) from e

def load_obj(file_path:str):
    try:
        with open(file_path,'rb') as file_obj:
            dill.load(file_obj)
    except Exception as e:
        raise HousingException(e,sys) from e


def write_yaml_file(file_path,data:dict=None):

    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,'w') as yaml_file:
            if data is not None:
                yaml.dump(data,yaml_file)

    except Exception as e:
        raise HousingException(e,sys) from e

def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise HousingException(e, sys) from e
