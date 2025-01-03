import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import CustromException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformermationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformermationConfig()
        
    def get_data_transformer_obeject(self):
        
        ''' This function is  responsible for data transformation. '''
        	
        logging.info("Entered the data transformation method")
        
        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course',
            ]
            num_pipeline = Pipeline(
                steps= [
                    ('imputer', SimpleImputer(strategy='median')), # impute the missing values with median
                    ('std_scaler', StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')), # impute the missing values with most frequent
                    ('one_hot_encoder', OneHotEncoder(sparse_output=False))
                ]
            )
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('numerical_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )
            logging.info("Data transformation completed")
            return preprocessor
        
        
        except Exception as e:
            raise CustromException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        preprocessor = self.get_data_transformer_obeject()
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and Test Data successfully read")
            logging.info("Obtaining precprocessing object")
            preprocessing_obj = self.get_data_transformer_obeject()
            
            target_columns = 'math_score'
            numerical_columns = ['writing_score', 'reading_score']
            
            input_feature_train_df = train_df.drop(columns=[target_columns], axis=1)
            target_feature_train_df = train_df[target_columns]
            
            input_feature_test_df = test_df.drop(columns=[target_columns], axis=1)
            target_feature_test_df = test_df[target_columns]
            
            logging.info(f" Applying preprocessing object on train and test dataframes")
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logging.info(f"Saved preprocessing object.")
            
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustromException(e, sys)