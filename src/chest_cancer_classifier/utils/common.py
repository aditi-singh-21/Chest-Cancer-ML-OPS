import os
from box.exceptions import BoxValueError
import yaml
from chest_cancer_classifier import logger
import json
import joblib #For saving and loading Python objects as binary files.
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64

@ensure_annotations
def read_yaml(path_to_yaml : Path) -> ConfigBox:
    try:
        with open (path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file {path_to_yaml} successfully loaded")
            return ConfigBox(content)
    except BoxValueError:
            raise ValueError(f"YAML file {path_to_yaml} is empty")
    except Exception as e:
        raise e

@ensure_annotations
def create_dir(path_to_dir:list,verbose = True):
    for path in path_to_dir:
        os.makedirs(path,exist_ok=True)
        if verbose:
            logger.info(f"Created Directory at : {path}")
            
@ensure_annotations
def save_json(path :Path, data:dict):
    with open(path ,"w") as f:
        json.dump(data,f,indent=4)
    logger.info(f"json file saved at: {path}")
    
@ensure_annotations
def load_json(path : Path) -> ConfigBox:
    with open(path) as f:
        content = json.load(f)
    
    logger.info(f"json file loaded successfully from : {path}")
    return ConfigBox(content)

@ensure_annotations
def save_bin(data:Any , path:Path):
    joblib.dump(value=data,filename=path)  #Uses joblib.dump() to serialize the object and save it as a binary file.
    logger.info(f"binary file saved at :{path}")
    
@ensure_annotations
def load_bin(path:Path) -> Any:
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data

@ensure_annotations
def get_size(path:Path) -> str:
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~{size_in_kb} KB"

def decodeImage(imgstring,file_name):
    imgdata = base64.b64decode(imgstring)
    with open(file_name,'wb') as f:
        f.write(imgdata)
        f.close()
   
def encodeImage(croppedimagepath):
    with open(croppedimagepath,'rb')as f:
        return base64.b64encode(f.read()) 