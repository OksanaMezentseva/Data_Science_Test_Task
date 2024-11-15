# Import necessary libraries
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# Load the trained model and tokenizer
model_name = "Task_1_NER/model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

def predict(text):
    """
    Perform NER prediction on the given text.
    
    Parameters:
    - text (str): The input text for NER.
    
    Returns:
    - List of predictions for each token in the input text.
    """
    tokens = tokenizer(text, return_tensors="pt")
    outputs = model(**tokens)
    predictions = outputs.logits.argmax(dim=-1).squeeze().tolist()
    return predictions

# Example usage
text = "Mount Everest is the highest mountain."
print(predict(text))