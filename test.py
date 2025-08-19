import torch

# 1. Create the two tensors with the specified shapes
a = torch.randn(120, 1620)
b = torch.randn(1620, 8)

# 2. Add a new dimension to tensor 'a' at the end (axis 2)
#    This changes a's shape from (120, 1620) -> (120, 1620, 1)
a_expanded = a.unsqueeze(2)

# 3. Multiply the expanded 'a' with 'b'
#    PyTorch will broadcast both tensors to (120, 1620, 8)
result = a_expanded * b

# --- Alternative using None indexing (more concise) ---
# result = a[..., None] * b

# 4. Check the shapes to confirm
print(f"Shape of a: {a.shape}")
print(f"Shape of b: {b.shape}")
print(f"Shape of a_expanded: {a_expanded.shape}")
print(f"Shape of result: {result.shape}")