# submission/model.py
import os
import sys
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Add parent dir so we can import lora.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lora import LoRALinear

BASE_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "base_model")
ARTIFACTS_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

RANK  = 8
ALPHA = 16.0


def load_model():
    """
    Load the fine-tuned model with LoRA adapters.
    Returns a transformers.PreTrainedModel ready for inference.
    No retraining occurs here.
    """
    # Load base model and freeze all weights
    model = GPT2LMHeadModel.from_pretrained(BASE_MODEL_PATH)
    for p in model.parameters():
        p.requires_grad = False

    # Inject LoRA layers (same structure as training)
    for block in model.transformer.h:
        attn = block.attn
        old  = attn.c_attn
        in_f  = old.weight.shape[0]
        out_f = old.weight.shape[1]

        lora_layer = LoRALinear(in_f, out_f, RANK, ALPHA)
        with torch.no_grad():
            lora_layer.weight.copy_(old.weight.T)
        attn.c_attn = lora_layer

    # Load saved adapter weights
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            safe_name = name.replace(".", "_")
            pt_path   = os.path.join(ARTIFACTS_PATH, f"{safe_name}.pt")
            if os.path.exists(pt_path):
                weights = torch.load(pt_path, map_location="cpu", weights_only=True)
                with torch.no_grad():
                    module.lora_A.copy_(weights["lora_A"])
                    module.lora_B.copy_(weights["lora_B"])

    model.eval()
    return model


if __name__ == "__main__":
    print("Loading model...")
    model = load_model()
    print(f"Model loaded: {type(model).__name__}")

    # Count LoRA layers
    lora_layers = [m for m in model.modules() if isinstance(m, LoRALinear)]
    print(f"LoRA layers found: {len(lora_layers)}")

    # Quick inference test
    tokenizer = GPT2Tokenizer.from_pretrained(BASE_MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    prompt = "Classify the sentiment of this review as positive or negative: This product is excellent and works perfectly. "
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)

    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\nTest prompt : {prompt}")
    print(f"Model output: {response.strip()}")