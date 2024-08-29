import os
from box.exceptions import BoxValueError
import yaml
from chest_cancer_classifier import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64

@ensure_annotations
def read_yaml(path_to_yaml : Path) -> ConfigBox:
    try:
        print("none")
            
    except:
        print("error")
            