import torch
from torch.utils.data import Dataset

class NextTokenDataset(Dataset):
  def __init__(
      self,
      tokenized: list[list[int]],
      max_length: int = 40,
      pad_token_id: int = 0
  ):
    self.tokenized = tokenized
    self.max_length = max_length
    self.pad_token_id = pad_token_id

  def __len__(self) -> int:
    return len(self.tokenized)
  
  def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = self.tokenized[idx]

    if len(tokens) > self.max_length:
      tokens = tokens[:self.max_length]

    x = tokens[:-1]
    y = tokens[1:]

    seq_len = self.max_length - 1
    pad_len = seq_len - len(x)

    if pad_len > 0:
      x = x + [self.pad_token_id] * pad_len
      y = y + [self.pad_token_id] * pad_len

    return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
