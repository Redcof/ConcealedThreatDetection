import torch
import yaml


def modify_with_args(cfg):
    # TODO consume command line arguments to modify the cfg
    return cfg


def read(yaml_path):
    def decorated(f):
        def wrapper(*args, **kwargs):
            with open(yaml_path) as fp:
                cfg = yaml.full_load(fp)
                from easydict import EasyDict
                cfg = EasyDict(cfg)
            cfg = modify_with_args(cfg)
            cfg.dataset.classnames = classnames(cfg.dataset.class_file)
            cfg.dataset.num_classes = len(cfg.dataset.classnames)
            cfg.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            result = f(cfg, *args, **kwargs)
            return result
        
        return wrapper
    
    return decorated


def classnames(path):
    with open(path) as fp:
        return list(map(lambda x: x.strip(), filter(lambda x: ".ignore" not in x, fp.readlines())))
