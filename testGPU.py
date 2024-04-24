import torch

# 检查 CUDA 是否可用
print(torch.__version__)
if torch.cuda.is_available():
    print("CUDA is available. GPU:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available.")
