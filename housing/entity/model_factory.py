from collections import namedtuple
import sys
from typing import List
import numpy as np
import yaml
from housing.exception import HousingException
from housing.logger import logging
import importlib
from sklearn.metrics import r2_score,mean_squared_error








GRID_SEARCH_KEY='grid_search'
CLASS_KEY='class'
MODULE_KEY='module'
PARAMS_KEY='params'
MODEL_SELECTION_KEY='model_selection'
SEARCH_PARAM_GRID_KEY='search_param_grid'

InitializedModelDetail = namedtuple('InitializedModelDetail',['model_serial_number',
                                    'model',
                                    'param_grid_search',
                                    'model_name'
                                    ])

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
                                    'best_score'
                                     ])

MetricInfoArtifact = namedtuple('MetricInfoArtifact',['model_name',
                                                    'model_object',
                                                    'train_rmse',
                                                    'test_rmse',
                                                    'train_accuracy',
                                                    'test_accuracy',
                                                    'model_accuracy',
                                                    'index_number'
                                                     ])


def evaluate_classification_model(model_list: list, 
                                  X_train:np.ndarray, 
                                  y_train:np.ndarray, 
                                  X_test:np.ndarray, 
                                  y_test:np.ndarray, 
                                  base_accuracy:float=0.6
                                  ) -> MetricInfoArtifact:
    pass

def evaluate_regression_model(model_list: list, 
                              X_train:np.ndarray, 
                              y_train:np.ndarray, 
                              X_test:np.ndarray, 
                              y_test:np.ndarray, 
                              base_accuracy:float=0.6
                              ) -> MetricInfoArtifact:
    """
    Description:
    This function compare multiple regression model return best model
    Params:
    model_list: List of model
    X_train: Training dataset input feature
    y_train: Training dataset target feature
    X_test: Testing dataset input feature
    y_test: Testing dataset input feature
    return
    It retured a named tuple
    
    MetricInfoArtifact = namedtuple("MetricInfo",
                                ["model_name", "model_object", "train_rmse", "test_rmse", "train_accuracy",
                                 "test_accuracy", "model_accuracy", "index_number"])
    """
    try:
        
        index_number= 0
        metic_info_artifact= None
        for model in model_list:
            model_name=str(model)
            logging.info(f'{"="*20} started evaluating model: [{type(model).__name__}] {"="*20}')

            y_train_pred=model.predict(X_train)
            y_test_pred=model.predict(X_test)

            train_acc= r2_score(y_train,y_train_pred)
            test_acc= r2_score(y_test,y_test_pred)

            train_rmse=np.sqrt(mean_squared_error(y_train,y_train_pred))
            test_rmse=np.sqrt(mean_squared_error(y_test,y_test_pred))

            model_accuracy=(2*(test_acc*train_acc))/(train_acc+test_acc)
            diff_test_train_acc=abs(train_acc-test_acc)

            logging.info(f'{"="*20} score {"="*20}')
            logging.info(f'train score /t/t test score /t/t average score')
            logging.info(f'{train_acc}/t/t {test_acc}/t/t {model_accuracy}')

            logging.info(f'{"="*20} loss {"="*20}')
            logging.info(f'diff test train acuracy :[{diff_test_train_acc}]')
            logging.info(f'train root mean squared error : [{train_rmse}')
            logging.info(f'test root mean squared error : [{test_rmse}')

            if model_accuracy >= base_accuracy and diff_test_train_acc < 0.05:
                base_accuracy=model_accuracy
                metric_info_artifact=MetricInfoArtifact(model_name=model_name,
                                                        model_object=model,
                                                        train_accuracy=train_acc,
                                                        test_accuracy=test_acc,
                                                        train_rmse=train_rmse,
                                                        test_rmse=test_rmse,
                                                        index_number=index_number,
                                                        model_accuracy=model_accuracy
                                                        )
                logging.info(f'acceptable model found {metric_info_artifact}.')
            index_number += 1
            if metric_info_artifact is None:
                logging.info(f'no model found with higher accuracy than base accuracy')
            return metric_info_artifact

    except Exception as e:
        raise HousingException(e,sys) from e


def get_sample_model_config_yaml_file(export_dir: str):
    try:
        model_config = {
            GRID_SEARCH_KEY: {
                MODULE_KEY: "sklearn.model_selection",
                CLASS_KEY: "GridSearchCV",
                PARAMS_KEY: {
                    "cv": 3,
                    "verbose": 1
                }

            },
            MODEL_SELECTION_KEY: {
                "module_0": {
                    MODULE_KEY: "module_of_model",
                    CLASS_KEY: "ModelClassName",
                    PARAMS_KEY:
                        {"param_name1": "value1",
                         "param_name2": "value2",
                         },
                    SEARCH_PARAM_GRID_KEY: {
                        "param_name": ['param_value_1', 'param_value_2']
                    }

                },
            }
        }
        os.makedirs(export_dir, exist_ok=True)
        export_file_path = os.path.join(export_dir, "model.yaml")
        with open(export_file_path, 'w') as file:
            yaml.dump(model_config, file)
        return export_file_path
    except Exception as e:
        raise HousingException(e, sys)





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

    def initiate_best_grid_search_for_initialized_model(self,
                                                        initialized_model:InitializedModelDetail,
                                                        input_feature,
                                                        output_feature
                                                        ):
        try:
            return self.execute_grid_search_operation(initialized_model=initialized_model,
                                                      input_feature=input_feature,
                                                      output_feature=output_feature)
        except Exception as e:
            raise HousingException(e,sys) from e

    def initiate_best_grid_search_for_initialized_models(self,
                                                         initialized_model_list:List[InitializedModelDetail],
                                                         input_feature,
                                                         output_feature
                                                         ):
        try:
            self.grid_searched_best_model_list=[]
            for initialized_model_list in initialized_model_list:
                grid_searched_best_model=self.initiate_best_grid_search_for_initialized_model(initialized_model=initialized_model_list,
                                                                                              input_feature=input_feature,
                                                                                              output_feature=output_feature
                                                                                              )
                self.grid_searched_best_model_list.append(grid_searched_best_model)
            return self.grid_searched_best_model_list
            
        except Exception as e:
            raise HousingException(e,sys) from e

    @staticmethod
    def get_model_detail(model_details:List[InitializedModelDetail],
                         model_serial_number:str
                         )->InitializedModelDetail:
        try:
            for model_data in model_details:
                if model_data.model_serial_number==model_serial_number:
                    return model_data
            
        except Exception as e:
            raise HousingException(e,sys) from e


    @staticmethod
    def get_best_model_from_grid_searched_best_model_list(grid_searched_best_model_list:List[GridSearchBestModel],
                                                          base_accuracy=0.6
                                                          )->BestModel:
        try:
            best_model=None
            for grid_searched_best_model in grid_searched_best_model_list:

                if base_accuracy<grid_searched_best_model.best_score:

                    logging.info(f'acceptable model found {grid_searched_best_model}')
                    base_accuracy=grid_searched_best_model.best_score

                    best_model=grid_searched_best_model

                if not best_model:
                    
                    raise Exception(f'none of model has base acuracy : {base_accuracy}')
                logging.info(f'best model : {best_model}')
                return best_model

        except Exception as e:
            raise HousingException(e,sys) from e

    def get_best_model(self,
                    X,
                    y,
                    base_accuracy=0.6
                    )->BestModel:
        try:
            logging.info(f'started initializing model form config file')
            initialized_model_list = self.get_initialized_model_list()
            logging.info(f'initialized model list : {initialized_model_list}')
            grid_searched_model_list=self.initiate_best_grid_search_for_initialized_models(initialized_model_list=initialized_model_list,
                                                                                           input_feature=X,
                                                                                           output_feature=y
                                                                                           )
            return ModelFactory.get_best_model_from_grid_searched_best_model_list(grid_searched_best_model_list=grid_searched_model_list,
                                                                                  base_accuracy=base_accuracy)
        except Exception as e:
            raise HousingException(e,sys) from e
        