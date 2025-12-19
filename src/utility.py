import logging
import os


from logging.handlers import RotatingFileHandler

RESET = "\33[0m"
COLORS = {
    "DEBUG": "\033[37m",      # White
    "INFO": "\033[32m",       # Green
    "WARNING": "\033[33m",    # Yellow
    "ERROR": "\033[31m",      # Red
    "CRITICAL": "\033[1;31m", # Bold Red
}


class ColorFormatter(logging.Formatter):
    def format(self,record):
        color = COLORS.get(record.levelname,RESET)
        record.levelname = f"{color}{record.levelname}{RESET}"
        return super().format(record=record)

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


    