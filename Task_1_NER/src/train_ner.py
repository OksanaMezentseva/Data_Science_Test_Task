from transformers import AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset
from data_preprocessing import load_and_convert_labels
from tokenize_and_align_labels import tokenize_and_align_labels

# Path to the dataset
dataset_path = "/home/oks/VSCode_Projects/Data_Science_Test_Task/Task_1_NER/data/mountains_ner.csv"

# Load and preprocess the dataset
processed_df = load_and_convert_labels(dataset_path)

# Define label mapping
label_mapping = {
    "O": 0,
    "B-MOUNTAIN": 1,
    "I-MOUNTAIN": 2
}

# Convert processed DataFrame to Dataset and tokenize with label alignment
tokenized_dataset = Dataset.from_pandas(processed_df).map(
    lambda x: tokenize_and_align_labels(x, label_mapping), batched=False
)