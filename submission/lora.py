# lora.py
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    LoRA reparameterisation of a linear layer.
    output = W_pretrained @ x + (alpha / rank) * B @ A @ x
    W_pretrained is frozen. A and B are trainable.
    """

    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Frozen pretrained weight
        self.weight = nn.Parameter(
            torch.zeros(out_features, in_features), requires_grad=False
        )

        # Trainable low-rank matrices
        # A: (rank, in_features)  — initialized with gaussian
        # B: (out_features, rank) — initialized with zeros (so adapter starts at 0)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        nn.init.kaiming_uniform_(self.lora_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base output (frozen)
        base = x @ self.weight.T
        # LoRA output
        lora = x @ self.lora_A.T @ self.lora_B.T
        return base + self.scaling * lora

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int, alpha: float) -> "LoRALinear":
        """Replace an existing nn.Linear with a LoRALinear, copying its weights."""
        lora = cls(linear.in_features, linear.out_features, rank, alpha)
        with torch.no_grad():
            lora.weight.copy_(linear.weight)
        return lora


if __name__ == "__main__":
    # Quick self-test
    layer = LoRALinear(in_features=64, out_features=64, rank=4, alpha=8.0)

    # Check frozen weight
    assert not layer.weight.requires_grad, "weight must be frozen"
    # Check trainable adapters
    assert layer.lora_A.requires_grad, "lora_A must be trainable"
    assert layer.lora_B.requires_grad, "lora_B must be trainable"
    # Check shapes
    assert layer.lora_A.shape == (4, 64), f"bad shape: {layer.lora_A.shape}"
    assert layer.lora_B.shape == (64, 4), f"bad shape: {layer.lora_B.shape}"
    # Check forward
    x = torch.randn(2, 64)
    out = layer(x)
    assert out.shape == (2, 64), f"bad output shape: {out.shape}"

    print("All checks passed.")
    print(f"  weight shape : {layer.weight.shape}  frozen={not layer.weight.requires_grad}")
    print(f"  lora_A shape : {layer.lora_A.shape}  trainable={layer.lora_A.requires_grad}")
    print(f"  lora_B shape : {layer.lora_B.shape}  trainable={layer.lora_B.requires_grad}")
    print(f"  forward output shape: {out.shape}")