from src.DimondPricePrediction.components.data_ingestion import DataIngestion

import os
import sys
import pandas as pd
from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import customexception

obj=DataIngestion()

train_data_path,test_data_path = obj.initiate_data_ingestion()

data_transformation = DataTransformation()

train_arr,test_arr=data_transformation.initialize_data_tranformation(train_data_path,test_data_path)

model_trainer_obj = ModelTrainer(train_arr,test_arr)
model_trainer.initiate_model_training(train_arr,test_arr)
