from Question_Answering_With_BERT.config.configuration import ConfigurationManager
from Question_Answering_With_BERT.components.data_ingestion import DataIngestion
from Question_Answering_With_BERT.logging import logger


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        logger.info(f"Data ingestion config: {data_ingestion_config}")

        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()
        data_ingestion.validate_and_process_data()

        logger.info("Data ingestion pipeline completed successfully!")
