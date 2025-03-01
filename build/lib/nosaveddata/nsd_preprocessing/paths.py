import os
import yaml

from types import SimpleNamespace

def fname_of(path):
    return path.split(os.sep)[-1].split('.')[0]

def fexists(path):
    return os.path.exists(path)

def path_with_idx(path,idx):
    return path.split('.')[0] + '_' + str(idx) + '.' + path.split('.')[1]

def folder_name_of(path):
    return path.split(os.sep)[-2]

def path_of(path):
    return '/'.join(path.split(os.sep)[:-1])

def full_path_of(path):
    return '/'.join(os.path.abspath(path).split(os.sep)[:-1])

def mkdir(path):
    os.makedirs(path,exist_ok=True)



def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d


def read_yaml(path):
    with open(path, "r") as file:
        yaml_dict = yaml.safe_load(file)
    return dict_to_namespace(yaml_dict)