"""
some of the code is taken from
https://github.com/LunaBlack/KGAT-pytorch/blob/master/utils/log_helper.py
"""
import os
import logging
import csv
from collections import OrderedDict
import yaml
import json

# parser===========================================================
def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def create_save_checkpoint(name, dir_path):
    
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    checkpoint = os.path.join(dir_path, name + '.pt')
    return checkpoint

def create_log_id(name, dir_path):
    
    log_count = 0
    file_path = os.path.join(dir_path, name + '-log{:d}.log'.format(log_count))
    while os.path.exists(file_path):
        log_count += 1
        file_path = os.path.join(dir_path, name + '-log{:d}.log'.format(log_count))
        
    return name + '-log{:d}'.format(log_count)

def logging_dict(dicts, indent = ''):
    cur_indent = '' + indent
    for k, v in dicts.items():
        if isinstance(v, dict):
            logging.info(cur_indent +'|-' +k)
            logging_dict(v, indent = cur_indent +'    ')
        else:
            logging.info(cur_indent + '|-{0} : {1}'.format(k, v))
    
    
    
def logging_config(folder=None, name=None,
                   level=logging.DEBUG,
                   console_level=logging.DEBUG,
                   no_console=True):

    if not os.path.exists(folder):
        os.makedirs(folder)
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
    logging.root.handlers = []
    logpath = os.path.join(folder, name + ".log")
    print("All logs will be saved to %s" %logpath)

    logging.root.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logfile = logging.FileHandler(logpath)
    logfile.setLevel(level)
    logfile.setFormatter(formatter)
    logging.root.addHandler(logfile)

    if not no_console:
        logconsole = logging.StreamHandler()
        logconsole.setLevel(console_level)
        logconsole.setFormatter(formatter)
        logging.root.addHandler(logconsole)
        
    return folder