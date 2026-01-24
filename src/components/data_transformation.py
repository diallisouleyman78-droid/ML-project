import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer # for applying different transformations to different columns create pipelines
from sklearn.impute import SimpleImputer # for handling missing values
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl') # path to save the preprocessor object

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig() # create an instance of DataTransformationConfig

    def get_data_transformer_object(self):   

        '''
        Responsible for data transformation
        1. Numerical columns: imputation (median), scaling (StandardScaler)
        2. Categorical columns: imputation (most frequent), encoding (OneHotEncoder), scaling (StandardScaler)
        '''
        try:
            numeric_columns = ['reading score', 'writing score']
            categorical_columns = ['gender', 'race/ethnicity', 'parental level of education',
                                   'lunch', 'test preparation course']
            
            #numeric pipeline
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler(with_mean=False)) # scaling without centering to handle sparse matrix issue
            ])

            #categorical pipeline
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')), # replace missing values with most frequent value
                ('one_hot_encoder', OneHotEncoder()), # convert categorical variables to numerical variables
                ('scaler', StandardScaler(with_mean=False)) # scale the data
            ])
            logging.info("Numerical and categorical pipeline created/ encoding completed")

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numeric_columns), # pipeline and columns
                ('cat_pipeline', cat_pipeline, categorical_columns) # pipeline and columns
            ])


            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):    
        try:
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessor object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'math score'
            numeric_columns = ['reading score', 'writing score']

            input_feature_train_df = train_df.drop(columns =[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns =[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            logging.info("Applying preprocessing object on training and testing dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df) # fit and transform on training data
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df) # only transform on testing data 

            #combines the target feature and the input features into a single numpy array
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)] # concatenate numpy arrays
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logging.info("Concatenating input and target features array")

            #save the preprocessor as a pickle file
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path, # return the path where preprocessor is saved
            )


        except Exception as e:
            raise CustomException(e, sys)