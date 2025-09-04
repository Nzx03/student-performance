import os
import sys #for using custom exception
from src.exception import CustomException
from src.logger import logging
import pandas as pd 
from sklearn.model_selection import train_test_split
#collecting data splitting and then transforming it
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

#using decorators
@dataclass         #define class variable without using __Init__
class DataIngestionConfig:
    train_data_path:str= os.path.join('artifacts',"train.csv")  #for training data path
    #artifacts for output
    test_data_path:str= os.path.join('artifacts',"test.csv")
    raw_data_path:str= os.path.join('artifacts',"data.csv")

class DataIngestion:
    # for reading,saving and splitting data
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
       logging.info("entered the data ingestion method or component")
       try:
           df=pd.read_csv('notebook\data\stud.csv')
           logging.info("Read the dataset as dataframe")

           os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)  #True= if already there then it wil keep it
           df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True) #index=False is dont save row number and header= true (for saving column number)

           logging.info("Train Test split Initiated")
           train_set,test_set= train_test_split(df,test_size=0.2, random_state=42)

           train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
           test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
          

           logging.info('Ingestion of the data is completed')


           return(self.ingestion_config.train_data_path,
                  self.ingestion_config.test_data_path,
                  )
       except Exception as e:
           
           raise CustomException(e,sys)
       

if __name__=='__main__':   #so that this code only runs when script is executed directly (not when imported)
     obj=DataIngestion()
     train_data,test_data=obj.initiate_data_ingestion()

     data_transformation=DataTransformation()
     train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)   #combined both data ingestion and data transformation

     modeltrainer=ModelTrainer()
     print(modeltrainer.initiate_model_trainer(train_arr,test_arr))
     