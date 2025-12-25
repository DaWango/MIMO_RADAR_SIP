from dataclasses import dataclass
from enum import Enum
from functools import wraps
import logging
from logging.handlers import RotatingFileHandler
import os
import time
import numpy as np


RESET = "\33[0m"
COLORS = {
    "DEBUG": "\033[37m",      # White
    "INFO": "\033[32m",       # Green
    "WARNING": "\033[33m",    # Yellow
    "ERROR": "\033[31m",      # Red
    "CRITICAL": "\033[1;31m", # Bold Red
}

###------------------###
### ENUM definitions ###
###------------------###
class PlotType(Enum):
    RANGE = 1
    RANGE_DOPPLER = 2
    CA_CFAR = 3

class ScaleDB(Enum):
    AMPLITUDE = 1
    POWER = 2
    NONE = 3


###-------------------------###
### Dataclasses definitions ###
###-------------------------###
@dataclass()
class PlotConfig():
    plot_type : PlotType = None
    plot3d : bool = False
    scale_db : ScaleDB = ScaleDB.AMPLITUDE
    for_rx : int = 0




###--------------------------###
### helper class definitions ###
###--------------------------###
class ColorFormatter(logging.Formatter):
    def format(self,record):
        color = COLORS.get(record.levelname,RESET)
        record.levelname = f"{color}{record.levelname}{RESET}"
        return super().format(record=record)

###------------------------------### 
### helper functions definitions ###
###------------------------------###

def setup_logging(level=logging.INFO,file_output = True,console_output = True):
    """
    Configure logging with colored log levels in console
    and plain log levels in file.
    """

    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)

    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)
    logger.handlers = []
    
    if file_output:
        logfile_path = os.path.join(os.path.dirname(__file__),"..","log","mimo_log.txt")
        log_dir = os.path.dirname(logfile_path)
        os.makedirs(log_dir, exist_ok=True)

        file_handler = RotatingFileHandler(logfile_path,mode="w",maxBytes=100*1024,backupCount=3,encoding="UTF-8")
        file_format = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
        file_handler.setFormatter(logging.Formatter(file_format)) 
        logger.addHandler(file_handler)
    
    if console_output:
        console_handler = logging.StreamHandler()
        console_format = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
        console_handler.setFormatter(ColorFormatter(console_format))
        console_handler.setLevel(level=level)
        logger.addHandler(console_handler)


def is_perfect_square(n):
    r = np.sqrt(n)
    r_int = np.rint(r).astype(int)
    if r_int * r_int == n:
        return True, r_int
    else:
        return False, r_int


def measure_time(func):
    @wraps(func)
    def wrapper(self,*args, **kwargs):
        start = time.perf_counter()
        result = func(self,*args, **kwargs)
        end = time.perf_counter()
        cls = self.__class__.__name__
        logging.getLogger().info(f"{cls}.{func.__name__} runs for {(end - start)*1000:.3f} ms")
        return result
    return wrapper






    