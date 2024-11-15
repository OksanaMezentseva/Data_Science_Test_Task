from transformers import AutoTokenizer

# Load the tokenizer for the model
model_name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_and_align_labels(example, label_mapping):
    """
    Tokenizes sentences and aligns NER labels with tokenized tokens for a single example.
    
    Parameters:
    - example (dict): A single example containing tokens and labels.
    - label_mapping (dict): A dictionary for label mapping.
    
    Returns:
    - dict: Tokenized inputs with aligned labels.
    """
    # Ensure tokens are provided as a list and tokenized accordingly
    tokens = example["tokens"]
    if isinstance(tokens[0], list):
        tokens = [str(token) for sublist in tokens for token in sublist]
    else:
        tokens = [str(token) for token in tokens]

    tokenized_inputs = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        padding=True
    )

    labels = []
    word_ids = tokenized_inputs.word_ids()
    previous_word_id = None

    for word_id in word_ids:
        if word_id is None:
            labels.append(-100)
        elif word_id != previous_word_id and word_id < len(example["labels"]):
            labels.append(label_mapping[example["labels"][word_id]])
        else:
            labels.append(-100)
        previous_word_id = word_id

    # Ensure labels have the same length as input_ids
    if len(labels) != len(tokenized_inputs["input_ids"]):
        raise ValueError(f"Length mismatch after alignment: {len(labels)} vs {len(tokenized_inputs['input_ids'])}")

    tokenized_inputs["labels"] = labels
    return tokenized_inputs