import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split 
from dataclasses import dataclass # automatically generate special methods like __init__()

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv') #defines where the train data will be saved
    test_data_path: str = os.path.join('artifacts', 'test.csv') #defines where the test data will be saved
    raw_data_path: str = os.path.join('artifacts', 'raw.csv') #defines where the raw data will be saved

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig() # create an instance of DataIngestionConfig

    #if data is in the database we will write the code here to fetch the data
    def initiate_data_ingestion(self): 
        logging.info("Entered the data ingestion method or component")
        #for every logging we need a try and except block
        try:
            df = pd.read_csv('notebook/data/StudentsPerformance.csv') # reading the data from the source   
            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True) # create the artifacts folder if not exists
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True) #save the raw data

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42) # split the data into train and test set
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True) #save the train data
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True) #save the test data
            logging.info("Ingestion of the data is completed")

            #returning the train and test data path for transformation component
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()        
