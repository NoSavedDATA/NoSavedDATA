import os

def fname_of(path):
    return path.split(os.sep)[-1].split('.')[0]

def folder_name_of(path):
    return path.split(os.sep)[-2]

def path_of(path):
    return '/'.join(path.split(os.sep)[:-1])

def full_path_of(path):
    return '/'.join(os.path.abspath(path).split(os.sep)[:-1])

def mkdir(path):
    os.makedirs(path,exist_ok=True)