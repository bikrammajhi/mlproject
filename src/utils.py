import os
import sys
import dill

import numpy as np
import pandas as pd

from src.exception import CustromException

def save_object(file_path: str, obj: object):
    
    try:
        dir_path = os.path.dirname(file_path)
        
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        raise CustromException(e, sys)