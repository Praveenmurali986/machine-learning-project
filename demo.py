from housing.pipeline.pipeline import Pipeline
from housing.logger import logging
import os,sys
from housing.constant import *
from housing.config.configuration import Configuration
from housing.component.data_transformation import DataTransformation


def main():
    try:
        pipeline=Pipeline(config=Configuration)
        pipeline.get_experiments_status()
        # data=Configuration().get_data_transformation_config()
        # print(data)
        # schema_file_path=r'C:\Users\PRAVEEN\,FSDS\ml\ml_project\project_1\sample-end-to-end-project\config\schema.yaml'
        # file_path=r'C:\Users\PRAVEEN\,FSDS\ml\ml_project\project_1\sample-end-to-end-project\housing\artifact\data_ingestion\2022-07-05-18-58-44\raw_data\housing.csv'
        # df=DataTransformation.load_data(file_path=file_path,schema_file_path=schema_file_path)
        # print(df.columns)
        # print(df.dtypes)
    except Exception as e:
        logging.error(f'{e}')
        print(e)

if __name__=='__main__':
    main()

