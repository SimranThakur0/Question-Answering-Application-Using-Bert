import os
from Question_Answering_With_BERT.logging import logger  # Adjusted import based on your logging setup
from entity.config_entity import DataTransformationConfig  # Adjusted import based on your entity setup


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def convert_examples_to_features(self, example_batch):
        """
        Converts raw examples into tokenized features suitable for model input.
        """
        input_encodings = self.tokenizer(
            example_batch['dialogue'], max_length=1024, truncation=True
        )

        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(
                example_batch['summary'], max_length=128, truncation=True
            )

        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }

    