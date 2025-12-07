import torch
from torch import Tensor
from rouge_score import rouge_scorer
from tqdm import tqdm

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def split_text_for_eval(tokens: list[int], input_ratio: float = 0.75) -> tuple[list[int], list[int]]:
  split_idx = int(len(tokens) * input_ratio)

  return tokens[:split_idx], tokens[split_idx:]

def calculate_rouge(prediction: str, target: str) -> dict:
  scores = scorer.score(target, prediction)
  
  return {
    'rouge1': scores['rouge1'].fmeasure,
    'rouge2': scores['rouge2'].fmeasure,
    'rougeL': scores['rougeL'].fmeasure
  }

def evaluate_lstm(
    model,
    tokenizer,
    texts: list[list[int]],
    device: torch.device,
    input_ratio: float = 0.75,
    max_samples: int = 100
) -> dict:
  model.eval()

  rouge1_scores = []
  rouge2_scores = []
  rougeL_scores = []
  examples = []

  for tokens in tqdm(texts[:max_samples], desc='Evaluating LSTM'):
    if len(tokens) < 4:
      continue

    input_ids, target_ids = split_text_for_eval(tokens, input_ratio)

    if len(target_ids) == 0:
      continue

    input_tensor = torch.tensor(input_ids, dtype=torch.long).to(device)
    output_ids = model.generate(input_tensor, max_new_tokens=len(target_ids))
    generated_ids = output_ids[0, len(input_ids):].tolist()

    prediction = tokenizer.decode(generated_ids, skip_special_tokens=True)
    target = tokenizer.decode(target_ids, skip_special_tokens=True)

    scores = calculate_rouge(prediction, target)
    rouge1_scores.append(scores['rouge1'])
    rouge2_scores.append(scores['rouge2'])
    rougeL_scores.append(scores['rougeL'])

    if len(examples) < 5:
      examples.append({
        'input': tokenizer.decode(input_ids, skip_special_tokens=True),
        'prediction': prediction,
        'target': target
      })

  return {
    'rouge1': sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0,
    'rouge2': sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0,
    'rougeL': sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0,
    'examples': examples
  }