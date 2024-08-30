import os 
import urllib.request as request
import zipfile
import gdown
from chest_cancer_classifier import logger
from chest_cancer_classifier.utils.common import get_size
from chest_cancer_classifier.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self,config:DataIngestionConfig):
        self.config = config
        
    def download_file(self) -> str:
        try:
            dataset_url = self.config.source_url
            zip_download_dir = self.config.local_data_file
            os.makedirs("artifacts/dataingestion",exist_ok = True)
            logger.info(f"Downloading dat from {dataset_url} into file {zip_download_dir}")
            
            file_id = dataset_url.split('/')[-2]
            prefix = "https://drive.google.com/uc?/export=download&id="
            gdown.download(prefix + file_id , "archive(6).zip")
            
            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")
            
        except Exception as e:
            raise e
        
    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path,exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file,'r') as zip_file:
            zip_file.extractall(unzip_path)