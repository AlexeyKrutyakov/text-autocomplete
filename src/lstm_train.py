import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from eval_lstm import evaluate_lstm

def train_lstm(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tokenizer,
    val_texts: list[list[int]],
    device: torch.device,
    num_epochs: int = 5,
    lr: float = 3e-3,
    eval_rouge_samples: int = 100,
) -> dict:
  criterion = nn.CrossEntropyLoss(ingore_index=0)
  optimizer = optim.Adam(model.parameters(), lr=lr)

  history = {'train_loss': [], 'val_loss': [], 'rouge1': [], 'rouge2': [], 'rougeL': []}

  for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    t0 = time.time()

    for i, (x_batch, y_batch) in enumerate(train_loader):
      x_batch, y_batch = x_batch.to(device), y_batch.to(device)

      optimizer.zero_grad()
      logits = model(x_batch)
      loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
      loss.backward()
      optimizer.step()

      total_loss += loss.item()

      if (i + 1) % 500 == 0:
        elapsed = time.time() - t0
        print(f'Epoch {epoch + 1} | Batch {i + 1}/{len(train_loader)} | Loss: {loss.item():.4f} | Time: {elapsed:.1f}s')
        t0 = time.time()

    total_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
      for x_batch, y_batch in val_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        logits = model(x_batch)
        loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
        val_loss += loss.item()
    val_loss /= len(val_loader)

    rouge_results = evaluate_lstm(
      model,
      tokenizer,
      val_texts,
      device,
      max_samples=eval_rouge_samples
    )

    history['train_loss'].append(total_loss)
    history['val_loss'].append(val_loss)
    history['rouge1'].append(rouge_results['rouge1'])
    history['rouge2'].append(rouge_results['rouge2'])
    history['rougeL'].append(rouge_results['rougeL'])

    print(f'Epoch {epoch + 1}/{num_epochs} | Train loss: {total_loss:.4f} | Val loss: {val_loss:.4f}')
    print(f'ROUGE-1: {rouge_results['rouge1']:.4f} | ROUGE-2: {rouge_results['rouge2']:.4f} | ROUGE-L: {rouge_results['rougeL']:.4f}')

  return history
