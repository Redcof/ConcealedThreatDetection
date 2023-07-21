import platform
import sys

import torch
import yaml


def update_args(cfg):
    # TODO consume command line arguments to modify the cfg
    return cfg


def read(yaml_path, modify_with_args=True, load_env=True):
    def decorated(f):
        def wrapper(*args, **kwargs):
            # read config file
            with open(yaml_path) as fp:
                cfg = yaml.full_load(fp)
                from easydict import EasyDict
                cfg = EasyDict(cfg)
            if modify_with_args:
                cfg = update_args(cfg)
            if load_env:
                from dotenv import load_dotenv
                load_dotenv('.env')  # take environment variables from .env.
            cfg.dataset.classnames = classnames(cfg.dataset.class_file)
            cfg.dataset.num_classes = len(cfg.dataset.classnames)
            cfg.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            cfg.device_name = torch.cuda.get_device_name()
            cfg.os = platform.uname()
            cfg.python = sys.version
            cfg.exp_name = "%s_%s_%s_%s" % (cfg.name, cfg.dataset.name, cfg.model.name, cfg.exp_name_postfix)
            result = f(cfg, *args, **kwargs)
            return result
        
        return wrapper
    
    return decorated


def classnames(path):
    with open(path) as fp:
        return list(map(lambda x: x.strip(), filter(lambda x: ".ignore" not in x, fp.readlines())))
