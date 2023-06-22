import torch
import pickle

from safetensors import safe_open
from safetensors.torch import save_file

# Create tensors

# Create tensors
tensor1 = torch.tensor([1, 2, 3], dtype=torch.float32)
tensor2 = torch.tensor([[True, False], [False, True]], dtype=torch.bool)
tensor3 = torch.tensor([4, 5, 6], dtype=torch.int64)
float_tensor = torch.randn(3, 4)
half_tensor = torch.randn(2, 2).half()
int_tensor = torch.randint(0, 10, (2, 3))
bool_tensor = torch.tensor([True, False, True])

# Create a dictionary to hold all the tensors
tensor_dict = {
    'tensor1': tensor1,
    'tensor2': tensor2,
    'tensor3': tensor3,
    'float_tensor': float_tensor,
    'half_tensor': half_tensor,
    'int_tensor': int_tensor,
    'bool_tensor': bool_tensor,
}

# Save tensors in native torch file format (.pt)
torch.save(tensor_dict, '../target/all_tensors.pt')
# Save tensors in SafeTensor format (.sft)
save_file(tensor_dict, '../target/all_tensors.sft')

