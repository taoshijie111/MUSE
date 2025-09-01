import os
import logging
import random
import numpy as np
from pathlib import Path
from datetime import datetime

from src.tools.path import ensure_dir

def add_time_to_log_dir(name, prefix=None):
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M')
    log_dir = f'{prefix}_{name}_{current_time}'
    return log_dir


class Logger:
    def __init__(self, name, prefix='', state='info', root='.'):
        self.name = name
        self.prefix = prefix
        self.state = state
        log_name = f'{prefix}_{name}'

        # unique path name
        LOGROOT = Path(root) / add_time_to_log_dir(name, prefix)
        ensure_dir(LOGROOT)
        self.log_path = LOGROOT
        # 1. log file
        log_file_path = LOGROOT / f'main.log'
        file_fmt = "%(asctime)-15s %(levelname)s  %(message)s"
        file_date_fmt = "%a %d %b %Y %H:%M:%S"
        file_for_matter = logging.Formatter(file_fmt, file_date_fmt)

        # logger level
        if state.lower() == 'info':
            level = logging.INFO
        elif state.lower() == 'debug':
            level = logging.DEBUG
        elif state.lower() == 'error':
            level = logging.ERROR
        elif state.lower() == 'warning':
            level = logging.WARNING
        elif state.lower() == 'critical':
            level = logging.CRITICAL
        else:
            level = logging.INFO

        # logger save
        fh = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
        fh.setLevel(level)
        fh.setFormatter(file_for_matter)
        sh = logging.StreamHandler()
        sh.setLevel(level)
        root_logger = logging.getLogger()
        for h in root_logger.handlers:
            root_logger.removeHandler(h)
        logging.basicConfig(level=level, handlers=[sh, fh])
        self.logger = logging.getLogger(log_name)
        # self.logger = get_logger(log_name, log_level="INFO")
    
    def info(self, msg):
        self.logger.info(msg)
    
    def info(self, s):
        self.logger.info(s)

    def warning(self, s):
        self.logger.warning(s)