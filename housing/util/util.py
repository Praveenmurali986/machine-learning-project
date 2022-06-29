import os,sys
import yaml
from housing.exception import HousingException


def read_yaml_file(file_path)->dict:
    '''
    Read yaml file and return as dict 
    '''
    try:
        with open(file_path,'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise HousingException(e,sys) from e