import os,sys
import yaml
from housing.exception import HousingException
import numpy as np


def read_yaml_file(file_path)->dict:
    '''
    Read yaml file and return as dict 
    '''
    try:
        with open(file_path,'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
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