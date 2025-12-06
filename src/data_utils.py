import re
import html
from transformers import AutoTokenizer
import json
import random

def load_texts(filepath: str) -> list[str]:
  with open(filepath, 'r', encoding='utf-8') as f:
    texts = [line.strip() for line in f]

  return texts

def clean_text(text: str) -> str:
  # decode html entities
  text = html.unescape(text)

  text = text.replace('\n', ' ')

  # remove `RT` (retweet)
  text = re.sub(r'^RT\s+@\w+:\s*', '', text, flags=re.IGNORECASE)

  text = text.lower()

  # remove links
  text = re.sub(r'http\S+|www\.\S+', '', text)

  # remove @mentions
  text = re.sub(r'@\w+', '', text)

  # remove char `#`
  text = text.replace('#', '')

  # remove broken chars
  text = text.replace("\ufffd", "").replace("�", "").replace("ï¿½", "")
  
  # remove emojies
  emoji_pattern = re.compile(
    '['
      '\U0001F600-\U0001F64F'
      '\U0001F300-\U0001F5FF'
      '\U0001F680-\U0001F6FF'
      '\U0001F1E0-\U0001F1FF'
      '\U00002700-\U000027BF'
      '\U0001F900-\U0001F9FF'
    ']+',
    flags = re.UNICODE
  )
  text = emoji_pattern.sub('', text)

  # remove text smiles
  text = re.sub(r"[:;=8][-']?[)(\]\[dDpP3><|\\\/}{@]", " ", text)
  text = re.sub(r"[)(\]\[dDpP><|\\\/][-']?[:;=8]", " ", text)

  # remove repeated punctuation marks (`...`, `!!!`, `???`)
  text = re.sub(r"\.(\s*\.)+", " ", text)
  text = re.sub(r"!(\s*!)+", "!", text)
  text = re.sub(r"\?(\s*\?)+", "?", text)

  # remove streched words
  text = re.sub(r"(.)\1{2,}", r"\1\1\1", text)

  # punctuation (`word ,word`, `word , word`, ...)
  text = re.sub(r"\s+([,.!?])", r"\1", text)
  text = re.sub(r"([,.!?])(\w)", r"\1 \2", text)
  
  # normalize spaces
  text = re.sub(r"\s+", " ", text).strip()

  return text

def save_texts(texts: list[str], filepath: str) -> None:
  with open(filepath, 'w', encoding='utf-8') as f:
    for text in texts:
      f.write(text + '\n')

def tokenize(
    texts: list[str],
    model_name: str = 'gpt2'
) -> list[list[int]]:
  tokenizer = AutoTokenizer.from_pretrained(model_name)

  tokenized = []
  for text in texts:
    tokens = tokenizer.encode(text)
    tokenized.append(tokens)

  return tokenized

def save_tokenized(tokenized: list[list[int]], filepath: str) -> None:
  with open(filepath, 'w', encoding='utf-8') as f:
    json.dump(tokenized, f)

def load_tokenized(filepath: str) -> list[list[int]]:
  with open(filepath, 'r', encoding='utf-8') as f:
    return json.load(f)

def is_ascii(text: str, threshold: float = 0.8) -> bool:
  if not text:
    return False
  
  if "ï¿½" in text:
    return False
  
  ascii_chars = sum(1 for c in text if c.isascii())
  
  return ascii_chars / len(text) >= threshold

def filter_by_length(tokenized: list[list[int]], min_length: int = 5) -> list[list[int]]:
  return [t for t in tokenized if len(t) >= min_length]

def train_val_test_split(
    data: list,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
) -> tuple[list, list, list]:
  random.seed(seed)
  data_shuffled = data.copy()
  random.shuffle(data_shuffled)

  n = len(data_shuffled)
  train_end = int(n * train_ratio)
  val_end = int(n* (train_ratio + val_ratio))

  train = data_shuffled[:train_end]
  val = data_shuffled[train_end:val_end]
  test = data_shuffled[val_end:]

  return train, val, test
