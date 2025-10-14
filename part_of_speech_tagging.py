# Author: T M Feroz Ali
# Description:
# 1. Bert-based-part-of-speech-tagging-for-English
# 2. Export to onnx using optimum.exporters.onnx import main_export 
# 3. Compare the outputs between onnx and pytorch model

from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline
import torch
import torch.nn.functional as F
import onnxruntime as ort
from optimum.exporters.onnx import main_export
import numpy as np


# Model and input
model_id = "QCRI/bert-base-multilingual-cased-pos-english"
text = "Where can i find an oven"

# Load tokenizer and HF model
tokenizer = AutoTokenizer.from_pretrained(model_id)
hf_model = AutoModelForTokenClassification.from_pretrained(model_id)

# Hugging Face pipeline output
pipeline = TokenClassificationPipeline(model=hf_model, tokenizer=tokenizer)
hf_pipeline_output = pipeline(text)
print("Hugging Face Pipeline Output:")
print(hf_pipeline_output)
print("=" * 50)

# HF model without pipeline. Tokenizer and model seperately
# Tokenize with offsets (return_offsets_mapping=True). This gives start and end positions for each token.
inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
print(f"Input Tokens: {inputs}")
breakpoint()
offset_mapping = inputs.pop("offset_mapping")[0].tolist()

# HF raw logits
with torch.no_grad():
    hf_logits = hf_model(**inputs).logits
hf_probs = F.softmax(hf_logits, dim=-1)
hf_predicted_indices = torch.argmax(hf_probs, dim=-1)[0]
hf_scores = hf_probs[0].max(dim=-1).values.tolist()
hf_labels = hf_model.config.id2label
hf_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# Manual HF output
hf_manual_output = []
for i, (token, label_idx, score, (start, end)) in enumerate(zip(hf_tokens, hf_predicted_indices, hf_scores, offset_mapping)):
    if token not in ["[CLS]", "[SEP]"]:
        hf_manual_output.append({
            "entity": hf_labels[label_idx.item()],
            "score": score,
            "index": i,
            "word": token,
            "start": start,
            "end": end
        })

print("Manual HF Output:")
print(hf_manual_output)
print("=" * 50)

######################################################################################
"""
# Export to onnx
model_id = "QCRI/bert-base-multilingual-cased-pos-english"
output_dir = "onnx_models/bert"

main_export(
    model_name_or_path=model_id,
    output=output_dir,
    task="token-classification"
)

print(f"Model exported to ONNX format in '{output_dir}' directory.")
"""

# ONNX inference
onnx_session = ort.InferenceSession("onnx_models/bert/model.onnx")
onnx_inputs = {k: v.cpu().numpy() for k, v in inputs.items()}
onnx_outputs = onnx_session.run(None, onnx_inputs)
onnx_logits = onnx_outputs[0]
onnx_probs = np.exp(onnx_logits) / np.sum(np.exp(onnx_logits), axis=-1, keepdims=True)
onnx_predicted_indices = np.argmax(onnx_probs, axis=-1)[0]
onnx_scores = np.max(onnx_probs, axis=-1)[0]

# ONNX formatted output
onnx_output = []
for i, (token, label_idx, score, (start, end)) in enumerate(zip(hf_tokens, onnx_predicted_indices, onnx_scores, offset_mapping)):
    if token not in ["[CLS]", "[SEP]"]:
        onnx_output.append({
            "entity": hf_labels[label_idx],
            "score": float(score),
            "index": i,
            "word": token,
            "start": start,
            "end": end
        })

print("ONNX Output:")
print(onnx_output)
print("=" * 50)

# Compare side by side
print("Comparison:")
for hf_item, onnx_item in zip(hf_manual_output, onnx_output):
    print(f"HF: {hf_item}")
    print(f"ONNX: {onnx_item}")
    print("---")
