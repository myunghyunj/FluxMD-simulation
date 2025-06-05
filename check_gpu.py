import torch
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.ones(1000, 1000, device=device)
    print(f"Successfully created tensor on MPS")
