from lib2to3.pgen2.token import NAME
from unicodedata import name
from pytz import VERSION
from setuptools import setup
from setuptools import setup,find_packages
from tables import Description
from typing import List
import numpy as np





NAME='housing_predictor'
VERSION='0.0.3'
AUTHOR='praveen'
DESCRIPTION='First ml project'
PACKAGES=['housing']
REQUIREMENT_FILE_NAME='requirements.txt'


def get_requirements_list()->List[str]:
    '''
    Description: returns requrements libraries writen in requirements.txt file

    returns: list of name of requirements

    '''
    with open(REQUIREMENT_FILE_NAME) as requirement_file:
        return requirement_file.readlines().remove('-e .')


setup(
name=NAME,
version=VERSION,
author=AUTHOR,
description=DESCRIPTION,
packages=find_packages(),
install_requires=get_requirements_list()

)


