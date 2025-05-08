import torch

print(f"CUDA Available: {torch.cuda.is_available()}, Device Count: {torch.cuda.device_count()}")
