# Author: T M Feroz Ali
#
# Description:
# 1. Bert-based-part-of-speech-tagging-for-English model
# 2. Export to onnx using torch.onnx.export (which takes input data)
# 3. Pad the input tokens to make it fixed length
# 4. Wrap the BERT model with another class to get clean model outputting only logits

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Load model and tokenizer
model_id = "QCRI/bert-base-multilingual-cased-pos-english"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForTokenClassification.from_pretrained(model_id)
model.eval()

# Sample input
text = "Where can I find an oven"
inputs = tokenizer(text, return_tensors="pt", 
    padding="max_length",     # Pad to max_length
    max_length=32,            # Desired fixed length
    truncation=True           # Truncate if input is longer than max_length
)

breakpoint()

# Sample output
with torch.no_grad():
    outputs = model(**inputs)
    output_logits = outputs.logits


# BERT model for token classification, the output is indeed of type: <class 'transformers.modeling_outputs.TokenClassifierOutput'>
# This means the model returns a dictionary-like object with attributes such as: logits: the actual output tensor of shape (batch_size, sequence_length, num_labels)
# What to Change for ONNX Export ?
# When exporting to ONNX, you need to extract the actual tensor (logits) from the TokenClassifierOutput object. The ONNX exporter doesn't know how to handle custom Python classes like TokenClassifierOutput.

# Define a wrapper to extract logits
class TokenClassificationWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return outputs.logits  # Only return the tensor


# Wrap the model
wrapped_model = TokenClassificationWrapper(model)

# Output of the wrapped model
with torch.no_grad():
    output_logits = wrapped_model(**inputs)


# Prepare the input and output names and data for onnx conversion
# Check if token_type_ids is required
input_names = ["input_ids", "attention_mask"]
input_tensors = (inputs["input_ids"], inputs["attention_mask"])

if "token_type_ids" in inputs:
    input_names.append("token_type_ids")
    input_tensors += (inputs["token_type_ids"],)

# Export to ONNX
torch.onnx.export(
    model,
    input_tensors,
    "bert_model_qcri_pos_tagging.onnx",
    input_names=input_names,
    output_names=["output_logits"],  # Use generic name
    opset_version=14         # Use a newer opset version
)

print("Model exported with custom input/output names.")
