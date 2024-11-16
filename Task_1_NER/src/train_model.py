from transformers import TrainingArguments, Trainer, AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification


def train_model(train_dataset, test_dataset, tokenizer, label_mapping):
    """
    Train a NER model on the tokenized dataset.

    Args:
    - train_dataset (Dataset): Tokenized training dataset.
    - test_dataset (Dataset): Tokenized test dataset.
    - tokenizer (AutoTokenizer): Tokenizer used for the model.
    - label_mapping (dict): Mapping of labels to numeric values.

    Returns:
    - Trainer: Trained model and trainer object.
    """
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",               # Directory to save results
        evaluation_strategy="epoch",          # Evaluate after each epoch
        learning_rate=2e-5,                   # Learning rate
        per_device_train_batch_size=4,       # Batch size for training
        per_device_eval_batch_size=4,        # Batch size for evaluation
        num_train_epochs=3,                   # Number of epochs
        weight_decay=0.01,                    # Regularization parameter
        save_total_limit=2,                   # Keep only the last two checkpoints
        logging_dir="./logs",                 # Directory for logs
        logging_steps=50                      # Log every 50 steps
    )

    # Load the model for token classification
    model = AutoModelForTokenClassification.from_pretrained(
        "bert-base-cased", 
        num_labels=len(label_mapping)        # Number of unique labels
    )

    # Initialize the Trainer
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
    )


    # Train the model
    print("Starting model training...")
    trainer.train()
    print("Model training completed.")

    # Save the model and tokenizer
    print("Saving model and tokenizer...")
    model.save_pretrained("Task_1_NER/model")  # Save model weights
    tokenizer.save_pretrained("Task_1_NER/model")  # Save tokenizer
    print("Model and tokenizer saved.")

    return trainer