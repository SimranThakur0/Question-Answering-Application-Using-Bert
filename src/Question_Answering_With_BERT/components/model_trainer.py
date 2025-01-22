import os
from transformers import AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import load_from_disk
from Question_Answering_With_BERT.logging import logger  # Adjusted import based on your logging setup
from Question_Answering_With_BERT.entity.config_entity import ModelTrainingConfig  # Adjusted import based on your entity setup


class ModelTrainer:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt)

    def train(self):
        """
        Trains the model using the processed dataset.
        """
        try:
            # Load the processed dataset
            dataset = load_from_disk(self.config.data_path)

            # Initialize training arguments
            training_args = TrainingArguments(
                output_dir=self.config.root_dir,
                num_train_epochs=self.config.num_train_epochs,
                per_device_train_batch_size=self.config.train_batch_size,
                per_device_eval_batch_size=self.config.eval_batch_size,
                warmup_steps=self.config.warmup_steps,
                weight_decay=self.config.weight_decay,
                logging_dir=self.config.logging_dir,
                logging_steps=self.config.logging_steps,
                evaluation_strategy="steps",
                save_steps=self.config.save_steps,
                save_total_limit=2,
                load_best_model_at_end=True,
                report_to="none",
            )

            # Initialize the Trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset["train"],
                eval_dataset=dataset["validation"],
            )

            # Train the model
            logger.info("Starting model training...")
            trainer.train()

            # Save the trained model
            trainer.save_model(self.config.root_dir)
            logger.info(f"Model saved at {self.config.root_dir}")

        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
