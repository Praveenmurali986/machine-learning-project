from collections import namedtuple
from email import message
from operator import mod
import sys
from typing import List
from matplotlib.pyplot import grid
import numpy as np
import yaml
from housing.exception import HousingException
from housing.logger import logging
import importlib








GRID_SEARCH_KEY='grid_search'
CLASS_KEY='class'
MODULE_KEY='module'
PARAMS_KEY='params'
MODEL_SELECTION_KEY='model_selection'
SEARCH_PARAM_GRID_KEY='search_param_grid'

InitializedModelDetail = namedtuple('InitializedModelDetail',['model_serial_number','model','param_grid_search','model_name'])

GridSearchBestModel = namedtuple('GridSearchBestModel',['model_serial_number',
                                                        'model',
                                                        'best_model',
                                                        'best_parameters',
                                                        'best_score'
    ])
                                                    

BestModel = namedtuple('BestModel',['model_serial_number',
                                    'model',
                                    'best_model',
                                    'best_parameters',
                                    'best_score',])

MetricInfoArtifact = namedtuple('MetricInfoArtifact',['model_name',
                                                    'model_object',
                                                    'train_rmse',
                                                    'test_rmse',
                                                    'train_accuracy',
                                                    'test_accuracy',
                                                    'model_accuracy',
                                                    'index_number'
                                                        ])




def evaluate_regression_model(model_list: list, X_train:np.ndarray, y_train:np.ndarray, X_test:np.ndarray, y_test:np.ndarray, base_accuracy:float=0.6) -> MetricInfoArtifact:
    pass


class ModelFactory:
    def __init__(self,model_config_path:str=None) -> None:
        try:
            self.config:dict=ModelFactory.read_params(model_config_path)
            self.grid_search_cv_module:str=self.config[GRID_SEARCH_KEY][MODULE_KEY]
            self.grid_search_class_name:str=self.config[GRID_SEARCH_KEY][CLASS_KEY]
            self.grid_search_property_data:dict=dict(self.config[GRID_SEARCH_KEY][PARAMS_KEY])
            self.models_initialization_config:dict=dict(self.config[MODEL_SELECTION_KEY])

            self.initialized_model_list=None
            self.grid_searched_best_model_list=None




        except Exception as e:
            raise HousingException(e,sys) from e
    
    @staticmethod
    def read_params(config_path:str)->dict:
        '''
        this function will read the yaml file of given file path.and return as dict
        '''
        try:
            with open(config_path,'rb') as yaml_file:
                config=yaml.safe_load(yaml_file)
            return config

        except Exception as e:
            raise HousingException(e,sys) from e

    @staticmethod
    def class_for_name(module_name:str,class_name:str):
        '''
        imports class_name form module_name.
        '''
        try:
            module=importlib.import_module(module_name)
            logging.info(f'exicuting command: from {module} import {class_name}')
            class_ref=getattr(module,class_name)

            return class_ref

        except Exception as e:
            raise HousingException(e,sys) from e

    @staticmethod
    def update_property_of_class(instance_ref:object,property_data:dict):
        '''
        check if property_data is dict type. if true update the instace_ref with data in property_data
        '''         
        try:
            if not isinstance(property_data,dict):
                raise Exception('property_data need to be dictionary')
            print(property_data)
            for key , value in property_data.items():
                logging.info(f'executing:$ {str(instance_ref)}.{key}={value}')
                setattr(instance_ref,key,value)
            return instance_ref

        except Exception as e:
            raise HousingException(e,sys) from e


    def get_initialized_model_list(self)->List[InitializedModelDetail]:
        '''
        this function will return list of model details.
        return List[ModelDetail]
        '''
        try:
            initialized_model_list=[]
            for model_serial_number in self.models_initialization_config.keys():

                model_initialization_config=self.models_initialization_config[model_serial_number]
                model_obj_ref=ModelFactory.class_for_name(
                    module_name=model_initialization_config[MODULE_KEY],
                    class_name=model_initialization_config[CLASS_KEY]
                )
                model=model_obj_ref()

                if PARAMS_KEY in model_initialization_config:
                    model_object_property=dict(model_initialization_config[PARAMS_KEY])
                    model=ModelFactory.update_property_of_class(instance_ref=model,
                                                                property_data=model_object_property
                                                                )                                    
                param_grid_search=model_initialization_config[SEARCH_PARAM_GRID_KEY]
                model_name=f'{model_initialization_config[MODULE_KEY]}.{model_initialization_config[CLASS_KEY]}'

                model_initialization_config= InitializedModelDetail(model_serial_number=model_serial_number,
                                                                    model=model,
                                                                    param_grid_search=param_grid_search,
                                                                    model_name=model_name
                                                                    )
                initialized_model_list.append(model_initialization_config)
            
            self.initialized_model_list=initialized_model_list
            return self.initialized_model_list
      
        except Exception as e:
            raise HousingException(e,sys) from e


    def execute_grid_search_operation(self,
                                        initialized_model:InitializedModelDetail,
                                        input_feature,
                                        output_feature
                                        )->GridSearchBestModel:
        '''
        execute_grid_search_operation(): function will perform parameter search operation and 
        it will return you the best optimistic model with best parameter:
        estimator: Model object
        param_grid: dictionary of parameter to perform search operation
        input_feature: your all input features
        output_feature: target/dependent features
        ======================================================================================
        return: function will return GridSearchOperation object

        '''
        try:

            grid_search_cv_ref=ModelFactory.class_for_name(module_name=self.grid_search_cv_module,
                                                            class_name=self.grid_search_class_name
                                                            )
            grid_search_cv=grid_search_cv_ref(estimator=initialized_model.model,
                                              param_grid=initialized_model.param_grid_search
                                              )   
            grid_search_cv=ModelFactory.update_property_of_class(instance_ref=grid_search_cv,
                                                                 property_data=self.grid_search_property_data
                                                                 )
            message=f'{"="*20} training {type(initialized_model.model).__name__} started.{"="*20}'
            logging.info(message)
            grid_search_cv.fit(input_feature,output_feature)
            message=f'{"="*20} training {type(initialized_model.model).__name__} completed.{"="*20}'
            logging.info(message)
            grid_searched_best_model=GridSearchBestModel(model_serial_number=initialized_model.model_serial_number,
                                                        model=initialized_model.model,
                                                        best_model=grid_search_cv.best_estimator_,
                                                        best_parameters=grid_search_cv.best_params_,
                                                        best_score=grid_search_cv.best_score_
                                                        )
            return grid_searched_best_model

        except Exception as e:
            raise HousingException(e,sys) from e


    
    
    
