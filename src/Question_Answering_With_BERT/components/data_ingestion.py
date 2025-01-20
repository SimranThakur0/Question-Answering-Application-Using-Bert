import os
import urllib.request as request
import json
from pathlib import Path
from zipfile import ZipFile
from logging import logger  
from utils.common import get_size  
from entity.config_entity import DataIngestionConfig  


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        """
        Downloads the dataset from the source URL if it doesn't already exist locally.
        """
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url=self.config.source_URL,
                filename=self.config.local_data_file
            )
            logger.info(f"{filename} downloaded with the following info: \n{headers}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")

    def extract_zip_file(self):
        """
        Extracts the zip file containing the dataset into the specified directory.
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        logger.info(f"Extracted {self.config.local_data_file} to {unzip_path}")

    def validate_and_process_data(self):
        """
        Validates and processes the dataset to ensure it is in the correct format for QA tasks.

        Example: Checks if the dataset follows the SQuAD format.
        """
        data_file_path = Path(self.config.unzip_dir) / self.config.dataset_file_name

        if not data_file_path.exists():
            raise FileNotFoundError(f"Dataset file {data_file_path} not found after extraction!")

        # Validate JSON format
        try:
            with open(data_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Dataset {data_file_path} successfully loaded and validated!")

            # Example: Check for required SQuAD keys
            if "data" in data and "version" in data:
                logger.info("Dataset format is valid for question answering!")
            else:
                raise ValueError("Dataset format is invalid for question answering tasks.")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

    def prepare_data(self):
        """
        Coordinates the data ingestion process: downloading, extracting, and validating.
        """
        self.download_file()
        self.extract_zip_file()
        self.validate_and_process_data()
