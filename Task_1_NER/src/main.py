# Import necessary modules and functions
from data_preprocessing import load_and_convert_labels
from tokenize_and_align_labels import tokenize_and_align_labels
from data_split import split_dataset
from train_model import train_model
from transformers import AutoTokenizer
from datasets import Dataset

# Define the main function
def main():
    """
    Main function to prepare data, train the NER model, and save results.
    """
    # Define file path to the dataset
    dataset_path = "/home/oks/VSCode_Projects/Data_Science_Test_Task/Task_1_NER/data/mountains_ner.csv"

    # Load and preprocess the dataset
    print("Loading and preprocessing dataset...")
    processed_df = load_and_convert_labels(dataset_path)

    # Define label mapping
    label_mapping = {"O": 0, "B-MOUNTAIN": 1, "I-MOUNTAIN": 2}

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    # Tokenize the dataset
    print("Tokenizing dataset...")
    tokenized_dataset = Dataset.from_pandas(processed_df).map(
        lambda x: tokenize_and_align_labels(x, label_mapping),
        batched=False
    )

    # Split dataset into train and test
    print("Splitting dataset into train and test sets...")
    split_data = split_dataset(tokenized_dataset)
    train_dataset = split_data['train']
    test_dataset = split_data['test']

    # Train the model
    print("Training the model...")
    trainer = train_model(train_dataset, test_dataset, tokenizer, label_mapping)

    print("Training completed.")

# Run the main function
if __name__ == "__main__":
    main()
