from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import Dataset
import torch
from sklearn.metrics import classification_report

# Load the tokenizer and the model
model_path = "/path/to/Task_1_NER/model"  
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

# Load the test dataset
test_dataset = Dataset.load_from_disk("/path/to/test_dataset")

# Label mapping
label_mapping = {0: "O", 1: "B-MOUNTAIN", 2: "I-MOUNTAIN"}
id_to_label = {v: k for k, v in label_mapping.items()}

# Function to predict on a batch of examples
def predict_batch(batch):
    tokens = batch["tokens"]
    labels = batch["labels"]

    # Tokenize the batch
    tokenized_inputs = tokenizer(
        tokens, is_split_into_words=True, truncation=True, padding=True, return_tensors="pt"
    )

    # Get predictions
    with torch.no_grad():
        outputs = model(**tokenized_inputs)

    predictions = torch.argmax(outputs.logits, dim=2).numpy()
    true_labels = labels.numpy()

    # Map predictions and labels back to human-readable format
    predicted_labels = [
        [id_to_label[label] for label in sentence if label != -100]
        for sentence in predictions
    ]
    true_labels = [
        [id_to_label[label] for label in sentence if label != -100]
        for sentence in true_labels
    ]

    return true_labels, predicted_labels

# Evaluate on the test set
all_true_labels = []
all_predicted_labels = []

for batch in test_dataset:
    true_labels, predicted_labels = predict_batch(batch)
    all_true_labels.extend(true_labels)
    all_predicted_labels.extend(predicted_labels)

# Calculate metrics
print(classification_report(all_true_labels, all_predicted_labels, target_names=label_mapping.values()))
