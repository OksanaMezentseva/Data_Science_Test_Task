from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# Load the trained model and tokenizer
model_dir = "Task_1_NER/model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForTokenClassification.from_pretrained(model_dir)

# Example sentence for inference
example_text = "Mount Everest is the tallest mountain in the world."

# Tokenize the input
inputs = tokenizer(
    example_text,
    return_tensors="pt",
    truncation=True,
    padding=True
)

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)

# Decode predictions
predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())
labels = [model.config.id2label[label] for label in predictions]

# Print results
print("Token predictions:")
for token, label in zip(tokens, labels):
    print(f"{token}: {label}")
