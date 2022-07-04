from housing.pipeline.pipeline import Pipeline
from housing.logger import logging
import os,sys
from housing.constant import *
from housing.config.configuration import Configuration


def main():
    try:
        pipeline=Pipeline()
        pipeline.run_pipeline()
        # data=Configuration().get_data_validation_config()
        # print(data)
    except Exception as e:
        logging.error(f'{e}')

if __name__=='__main__':
    main()

