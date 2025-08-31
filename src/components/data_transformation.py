#feature eng. data cleaning and hadling missing value catgorical and numerical features
import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer #for creating pipeline for like onehot encoding standardscaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
     preprocessor_ob_file_path=os.path.join('artifacts',"preprocessor.pkl")


class DataTransformation:
    def __init__(self):
          self.data_transformation_congif=DataTransformationConfig()

    def get_data_transformer_object(self):  #fro FE
          #this is for data transformation
          try:
               numerical_columns=['writing_score','reading_score']
               categorical_columns=[
                    "gender",
                    "race_ethnicity",
                    "parental_level_of_education",
                    "lunch",
                    "test_preparation_course"
               ]

               num_pipeline=Pipeline(
                 steps=[
                      ("imputer",SimpleImputer(strategy="median")),  # handling missing value and doing standard scaling which will run on to training dataset
                      ("scaler",StandardScaler(with_mean=False))   #with_mean=False as standard scaler is subtracting the mean from each feature which caused negative vlaues. but standard scaler assume that features have positive vale\ue so by setting it false it does not subtrtct mean from each feature
                       ]
                                    
                                    )
               cat_pipeline=Pipeline(
                    steps=[
                         ("imputer",SimpleImputer(strategy="most_frequent")),   # handling missing value and doing standard scaling which will run on to training dataset
                         ("one_hot_encoder",OneHotEncoder()),
                         ("scaler",StandardScaler(with_mean=False))   # change standardscaler

                          ]
               )

               logging.info(f"categorical columns:{categorical_columns}")
               logging.info(f"Numerical columns:{numerical_columns}")

               #for combining both numerical pipeline adn categorical pipeline 

               preprocessor=ColumnTransformer(
                    [
                         ("num pipeline",num_pipeline,numerical_columns),
                         ("cat_pipelines",cat_pipeline,categorical_columns)
                    ]
               )
               return preprocessor
          
          except Exception as e:
               raise CustomException(e,sys)
          
    def initiate_data_transformation(self, train_path, test_path):
         try:
              train_df=pd.read_csv(train_path)
              test_df=pd.read_csv(test_path)

              logging.info("Read train and test data")

              logging.info("obtaining preprocessing object")

              preprocessing_obj=self.get_data_transformer_object()

              target_column_name="math_score"
              numerical_columns=['writing_score','reading_score']

              input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
              target_feature_train_df=train_df[target_column_name]

              input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
              target_feature_test_df=test_df[target_column_name]

              logging.info(
                   f"applying preprocessing object on training dataframe and testing dataframe."
                   )
              
              input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df) #Fitting the preprocessor on train features.
              input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

              train_arr=np.c_[
                   input_feature_train_arr, np.array(target_feature_train_df)
              ]   #Combineing transformed features and target column back into one array (np.c_[] means concatenate column-wise).
              test_arr=np.c_[
                   input_feature_test_arr,np.array(target_feature_test_df)
              ]

              logging.info(f"Saved preprocessing object.")

              save_object(
                   file_path=self.data_transformation_congif.preprocessor_ob_file_path,
                   obj=preprocessing_obj
              )

              return(
                   train_arr,
                   test_arr,
                   self.data_transformation_congif.preprocessor_ob_file_path,
              )
              


         except Exception as e:
              raise CustomException(e,sys)

              
          
            
          
               
          





