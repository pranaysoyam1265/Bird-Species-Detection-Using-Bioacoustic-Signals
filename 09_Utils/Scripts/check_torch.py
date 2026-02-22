import torch

print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'
print(f'GPU: {gpu}')
