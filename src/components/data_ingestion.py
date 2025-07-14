import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.logger import logging
from dataclasses import dataclass
from src.exception import customException

from sklearn.model_selection import train_test_split
import pandas as pd

from src.components.data_transformation import DataTransormation
from src.components.data_transformation import DataTransormationConfig

@dataclass
class DataIngestionConfig:
    train_data_path : str=os.path.join('artifacts',"train.csv")  #giving inputs
    test_data_path : str=os.path.join('artifacts',"test.csv")
    raw_data_path : str=os.path.join('data',"raw","raw.csv")  

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()   #when we call dataingestion class it will call the dataingestionconfig class as sub variable

    def initiate_data_ingestion(self):
        try:
            # here we try to read the data from dataset or from database 
            df = pd.read_csv("notebook/data/stud.csv")
            logging.info ("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True) #creating the directory with respect to path given when it does not exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set = train_test_split(df,test_size =0.2,random_state= 42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise customException(e,sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data= obj.initiate_data_ingestion()

    data_transformation = DataTransormation()
    data_transformation.initiate_data_transformer(train_data,test_data)
     