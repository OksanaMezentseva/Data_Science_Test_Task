from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# Load the trained model and tokenizer
model_weights = "/home/oks/VSCode_Projects/Data_Science_Test_Task/Task_1_NER/model_weights"
model_tokenizer = "/home/oks/VSCode_Projects/Data_Science_Test_Task/Task_1_NER/tokenizer"
tokenizer = AutoTokenizer.from_pretrained(model_tokenizer)
model = AutoModelForTokenClassification.from_pretrained(model_weights)

# Example sentence for inference
example_text = "Mount Kilimanjaro is located in Africa."

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
