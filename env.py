# env.py — Full judge for LoRA from Scratch environment
import os
import sys
import json
import importlib
import subprocess
import torch

PROMPT = """
## Task: Implement LoRA Fine-Tuning from Scratch

You are working inside a virtual machine. Python 3.10, PyTorch 2.1, and the HuggingFace
transformers library are pre-installed. A GPU is available. Internet access is NOT available.

Your task is to implement Low-Rank Adaptation (LoRA) from scratch and use it to fine-tune
the pretrained language model located at base_model/ on the instruction-following dataset
at data/train.jsonl.

Required outputs:
  1. submission/lora.py       — LoRALinear(nn.Module) class
  2. submission/model.py      — load_model() -> transformers.PreTrainedModel
  3. submission/artifacts/    — saved LoRA adapter weights (.pt files)

Success criteria:
  - LoRALinear uses genuine low-rank decomposition (verified structurally)
  - load_model() returns a PreTrainedModel containing LoRALinear layers
  - Model achieves >= 72% accuracy on hidden sentiment classification task
  - No retraining inside load_model()
"""


def judge(base_dir: str = ".") -> bool:
    """
    Grade the submission. Returns True = pass, False = fail.
    Never raises exceptions.
    """
    submission_dir = os.path.join(base_dir, "submission")
    artifacts_dir  = os.path.join(submission_dir, "artifacts")
    judge_data     = os.path.join(base_dir, "judge_data", "eval.jsonl")

    print("\n" + "="*60)
    print("JUDGE RUNNING")
    print("="*60)

    try:
        # ── 1. File existence ──────────────────────────────────────
        print("\n[1] Checking required files...")
        for path, label in [
            (os.path.join(submission_dir, "lora.py"),  "submission/lora.py"),
            (os.path.join(submission_dir, "model.py"), "submission/model.py"),
            (artifacts_dir,                             "submission/artifacts/"),
        ]:
            if not os.path.exists(path):
                print(f"    FAIL — missing: {label}")
                return False
            print(f"    OK   — found: {label}")

        if not os.listdir(artifacts_dir):
            print("    FAIL — submission/artifacts/ is empty")
            return False
        print(f"    OK   — artifacts: {os.listdir(artifacts_dir)}")

        # ── 2. Source scan — no peft/loralib allowed ───────────────
        print("\n[2] Scanning lora.py for forbidden imports...")
        with open(os.path.join(submission_dir, "lora.py")) as f:
            source = f.read()
        for forbidden in ["peft", "loralib"]:
            if forbidden in source:
                print(f"    FAIL — forbidden library '{forbidden}' found in lora.py")
                return False
        print("    OK   — no forbidden imports found")

        # ── 3. Structural check ────────────────────────────────────
        print("\n[3] Checking LoRALinear structure...")
        sys.path.insert(0, submission_dir)
        sys.path.insert(0, base_dir)

        for mod in ["lora", "model"]:
            if mod in sys.modules:
                del sys.modules[mod]

        lora_mod   = importlib.import_module("lora")
        LoRALinear = getattr(lora_mod, "LoRALinear", None)

        if LoRALinear is None:
            print("    FAIL — LoRALinear class not found in lora.py")
            return False

        rank  = 4
        layer = LoRALinear(in_features=64, out_features=64, rank=rank, alpha=8.0)
        params = dict(layer.named_parameters())

        # Must have two low-rank matrices
        low_rank = [
            (n, p) for n, p in params.items()
            if rank in p.shape and p.shape != (64, 64)
        ]
        if len(low_rank) < 2:
            print(f"    FAIL — no low-rank matrices found. Params: "
                  f"{[(n, tuple(p.shape)) for n, p in params.items()]}")
            return False
        print(f"    OK   — low-rank matrices: "
              f"{[(n, tuple(p.shape)) for n, p in low_rank]}")

        # Must have frozen base weight
        frozen = [p for p in layer.parameters() if not p.requires_grad]
        if not frozen:
            print("    FAIL — no frozen parameters (W_pretrained must be frozen)")
            return False
        print(f"    OK   — frozen params: {len(frozen)}")

        # Forward shape check
        x   = torch.randn(2, 64)
        out = layer(x)
        if out.shape != (2, 64):
            print(f"    FAIL — forward() shape {tuple(out.shape)}, expected (2, 64)")
            return False
        print(f"    OK   — forward() output shape: {tuple(out.shape)}")

        # ── 4. load_model() subprocess test ───────────────────────
        print("\n[4] Testing load_model() in subprocess (30s timeout)...")
        result = subprocess.run(
            [sys.executable, "-c",
             f"import sys; sys.path.insert(0, r'{submission_dir}'); "
             f"sys.path.insert(0, r'{base_dir}'); "
             f"from model import load_model; m = load_model(); "
             f"print(type(m).__name__)"],
            timeout=30,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"    FAIL — load_model() raised exception:\n{result.stderr.strip()}")
            return False
        print(f"    OK   — load_model() returned: {result.stdout.strip()}")

        # ── 5. Type + LoRA layer check ─────────────────────────────
        print("\n[5] Checking model type and LoRA layers...")
        if "model" in sys.modules:
            del sys.modules["model"]

        model_mod = importlib.import_module("model")
        import transformers

        model = model_mod.load_model()

        if not isinstance(model, transformers.PreTrainedModel):
            print(f"    FAIL — got {type(model).__name__}, expected PreTrainedModel")
            return False
        print(f"    OK   — type: {type(model).__name__}")

        lora_layers = [m for m in model.modules() if type(m).__name__ == "LoRALinear"]
        if not lora_layers:
            print("    FAIL — no LoRALinear layers found in loaded model")
            return False
        print(f"    OK   — LoRALinear layers in model: {len(lora_layers)}")

        # ── 6. Hidden eval accuracy ────────────────────────────────
        print("\n[6] Running hidden evaluation...")
        if not os.path.exists(judge_data):
            print("    WARN — judge_data/eval.jsonl not found, skipping accuracy check")
            return True

        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(
            os.path.join(base_dir, "base_model")
        )
        tokenizer.pad_token = tokenizer.eos_token
        model.eval()

        examples = []
        with open(judge_data) as f:
            for line in f:
                examples.append(json.loads(line.strip()))

        correct = 0
        for i, ex in enumerate(examples):
            inputs = tokenizer(ex["prompt"], return_tensors="pt")
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    pad_token_id=tokenizer.eos_token_id
                )
            pred  = tokenizer.decode(out[0], skip_special_tokens=True).strip().lower()
            label = str(ex["label"]).strip().lower()
            if label in pred:
                correct += 1

        accuracy = correct / len(examples)
        print(f"    Accuracy: {correct}/{len(examples)} = {accuracy:.2%}")

        if accuracy < 0.72:
            print(f"    FAIL — {accuracy:.2%} is below 72% threshold")
            return False
        print(f"    OK   — {accuracy:.2%} passes the 72% threshold")

        # ── 7. Anti-retraining check ───────────────────────────────
        print("\n[7] Anti-retraining check...")
        before = {
            f: os.path.getmtime(os.path.join(artifacts_dir, f))
            for f in os.listdir(artifacts_dir)
        }
        model_mod.load_model()
        after = {
            f: os.path.getmtime(os.path.join(artifacts_dir, f))
            for f in os.listdir(artifacts_dir)
        }
        if before != after:
            print("    FAIL — artifacts/ modified during load_model() (retraining detected)")
            return False
        print("    OK   — artifacts unchanged after second load_model() call")

        # ── Result ─────────────────────────────────────────────────
        print("\n" + "="*60)
        print("RESULT: PASS ✓")
        print("="*60)
        return True

    except Exception as e:
        print(f"\n    Judge error: {e}")
        return False


if __name__ == "__main__":
    print("="*60)
    print("RL ENVIRONMENT: LoRA from Scratch + Fine-Tuning")
    print("="*60)
    print(PROMPT)

    result = judge(".")
    if not result:
        print("\nRESULT: FAIL ✗")