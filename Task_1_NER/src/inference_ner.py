import os
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# Define paths dynamically
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(base_dir, ".."))
model_dir = os.path.join(project_root, "model")

# Load the trained model and tokenizer
if not os.path.exists(model_dir):
    raise FileNotFoundError(f"Model directory not found at {model_dir}")

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForTokenClassification.from_pretrained(model_dir)

def predict(text):
    """
    Perform NER prediction on the given text.
    
    Parameters:
    - text (str): The input text for NER.
    
    Returns:
    - List of predictions for each token in the input text.
    """
    # Tokenize the input text
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**tokens)
    
    # Get predicted labels
    predictions = outputs.logits.argmax(dim=-1).squeeze().tolist()
    
    # Map predictions back to tokens
    tokenized_words = tokenizer.tokenize(text)
    result = list(zip(tokenized_words, predictions))
    
    return result

# Example usage
if __name__ == "__main__":
    text = "Mount Everest is the highest mountain."
    print(predict(text))
