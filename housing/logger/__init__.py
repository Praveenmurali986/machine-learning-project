from distutils.log import INFO
import logging
from datetime import datetime
import os



LOG_Dir='housing_log'

CURRENT_TIME_STAMP=f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

LOG_FILE_NAME=f"log_{CURRENT_TIME_STAMP}.log"

os.makedirs(LOG_Dir,exist_ok=True)

LOG_FILE_PATH=os.path.join(LOG_Dir,LOG_FILE_NAME)

logging.basicConfig(filename=LOG_FILE_PATH,
filemode='w',
format='[%(asctime)s] %(name)s-%(levelname)s-%(message)s',
level=logging.INFO
)



