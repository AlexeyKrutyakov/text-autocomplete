import torch
import torch.nn as nn
from torch import Tensor

class LSTMModel(nn.Module):
  def __init__(
      self,
      vocab_size: int,
      hidden_dim: int = 128,
      num_layers: int = 2,
      dropout: float = 0.2
  ) -> None:
    super().__init__()

    self.vocab_size = vocab_size
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers

    self.embedding = nn.Embedding(
      num_embeddings=vocab_size,
      embedding_dim=hidden_dim
    )
    self.lstm = nn.LSTM(
      input_size=hidden_dim,
      hidden_size=hidden_dim,
      num_layers=num_layers,
      batch_first=True,
      dropout=dropout if num_layers > 1 else 0.0
    )
    self.dropout = nn.Dropout(dropout)
    self.fc = nn.Linear(hidden_dim, vocab_size)


  def forward(self, x: Tensor) -> Tensor:
    embedded = self.embedding(x)
    out, _ = self.lstm(embedded)
    out = self.dropout(out)
    linear_out = self.fc(out)

    return linear_out
  
  @torch.no_grad()
  def generate(
    self,
    input_ids: Tensor,
    max_new_tokens: int = 1,
  ) -> Tensor:
    if input_ids.dim() == 1:
      input_ids = input_ids.unsqueeze(0)

    generated = input_ids.clone()

    hidden = None

    embedded = self.embedding(generated)
    _, hidden = self.lstm(embedded)

    for _ in range(max_new_tokens):
      last_token = generated[:, -1:]

      embedded = self.embedding(last_token)
      lstm_out, hidden = self.lstm(embedded, hidden)
      out = self.fc(lstm_out[:, -1, :])

      next_token = torch.argmax(out, dim=-1, keepdim=True)

      generated = torch.cat([generated, next_token], dim=1)

    return generated
