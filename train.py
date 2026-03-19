# train.py
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from lora import LoRALinear

# ── Config ─────────────────────────────────────────────────────────────────────
RANK       = 8
ALPHA      = 16.0
LR         = 3e-4
EPOCHS     = 3
BATCH_SIZE = 4
MAX_LEN    = 64
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")

# ── Dataset ────────────────────────────────────────────────────────────────────
class SentimentDataset(Dataset):
    def __init__(self, path, tokenizer, max_len):
        self.examples = []
        with open(path) as f:
            for line in f:
                ex = json.loads(line)
                text = ex["instruction"] + " " + ex["response"]
                enc = tokenizer(
                    text,
                    truncation=True,
                    max_length=max_len,
                    padding="max_length",
                    return_tensors="pt",
                )
                ids = enc["input_ids"].squeeze()
                self.examples.append(ids)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids = self.examples[idx]
        return ids[:-1], ids[1:]   # input, target (next-token prediction)


# ── Inject LoRA ────────────────────────────────────────────────────────────────
def inject_lora(model, rank, alpha):
    """Replace query and value projections in every attention block with LoRALinear."""
    count = 0
    for block in model.transformer.h:
        attn = block.attn
        # GPT-2 uses a single c_attn (QKV combined) and c_proj
        # We wrap c_attn as a whole — it's a Conv1D, so we handle it via its weight
        old = attn.c_attn          # shape: (768, 2304)  →  Q, K, V concatenated
        in_f  = old.weight.shape[0]   # 768
        out_f = old.weight.shape[1]   # 2304

        lora_layer = LoRALinear(in_f, out_f, rank, alpha)
        with torch.no_grad():
            lora_layer.weight.copy_(old.weight.T)  # Conv1D stores transposed

        attn.c_attn = lora_layer
        count += 1

    print(f"  Injected LoRA into {count} attention layers")
    return model


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("base_model")
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained("base_model")
    print(f"Base model loaded — {sum(p.numel() for p in model.parameters()):,} params")

    # Freeze all base weights
    for p in model.parameters():
        p.requires_grad = False

    # Inject LoRA
    model = inject_lora(model, RANK, ALPHA)
    model.to(DEVICE)

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Dataset
    dataset = SentimentDataset("data/train.jsonl", tokenizer, MAX_LEN)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Optimizer — only trainable params
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR
    )
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    print(f"\nTraining for {EPOCHS} epoch(s)...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for step, (x, y) in enumerate(loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            out  = model(x, labels=y)
            loss = out.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (step + 1) % 10 == 0:
                print(f"  Epoch {epoch+1} | Step {step+1}/{len(loader)} "
                      f"| Loss {total_loss/(step+1):.4f}")

        avg = total_loss / len(loader)
        print(f"  Epoch {epoch+1} complete — avg loss: {avg:.4f}")

    # Save LoRA weights
    print("\nSaving LoRA adapter weights to submission/artifacts/...")
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            safe_name = name.replace(".", "_")
            torch.save(
                {"lora_A": module.lora_A.data, "lora_B": module.lora_B.data},
                f"submission/artifacts/{safe_name}.pt"
            )

    import os
    saved = os.listdir("submission/artifacts")
    print(f"  Saved {len(saved)} adapter file(s): {saved}")
    print("\nTraining complete.")


if __name__ == "__main__":
    main()