import logging
import os
import pickle
import random
import shutil
from easydict import EasyDict

def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger

def log_config_to_file(cfg, pre='cfg', logger=None):
    for key, val in cfg.items():
        if isinstance(cfg[key], EasyDict):
            logger.info('----------- %s -----------' % (key))
            log_config_to_file(cfg[key], pre=pre + '.' + key, logger=logger)
            continue
        logger.info('%s.%s: %s' % (pre, key, val))


def summary_model(model,level=2):
    message = ""
    if level < 1:
        return message
    for name1, module1 in model.named_children():
        message += "[1] {}\n".format(name1)
        if level > 1:
            for name2, module2 in module1.named_children():
                message += "- [2] {}\n".format(name2)
                if level > 2:
                    for name3, module3 in module2.named_children():
                        message += " +++ [3] {}\n".format(name3)
                        if level > 3:
                            for name4, module4 in module3.named_children():
                                message += " +++++ [4] {}\n".format(name4)
    return message