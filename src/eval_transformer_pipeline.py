import torch
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)


def split_text_for_eval(tokens: list[int], input_ratio: float = 0.75) -> tuple[list[int], list[int]]:
    split_idx = int(len(tokens) * input_ratio)
    return tokens[:split_idx], tokens[split_idx:]


def calculate_rouge(prediction: str, target: str) -> dict:
    scores = scorer.score(target, prediction)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure,
    }


def generate_continuation(
    model,
    tokenizer,
    input_ids: list[int],
    max_new_tokens: int,
    device: torch.device,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 50,
) -> str:
    input_tensor = torch.tensor([input_ids]).to(device)
    attention_mask = torch.ones_like(input_tensor)

    output_ids = model.generate(
        input_tensor,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated_ids = output_ids[0, len(input_ids):].tolist()
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def evaluate_transformer(
    model,
    tokenizer,
    texts: list[list[int]],
    device: torch.device,
    input_ratio: float = 0.75,
    max_samples: int = 100,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 50,
) -> dict:
    model.eval()

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    examples = []

    for tokens in tqdm(texts[:max_samples], desc="Evaluating Transformer"):
        if len(tokens) < 4:
            continue

        input_ids, target_ids = split_text_for_eval(tokens, input_ratio)

        if len(target_ids) == 0:
            continue

        # Generate continuation
        prediction = generate_continuation(
            model, tokenizer, input_ids, 
            max_new_tokens=len(target_ids),
            device=device,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
        )
        target = tokenizer.decode(target_ids, skip_special_tokens=True)

        # Calculate ROUGE
        scores = calculate_rouge(prediction, target)
        rouge1_scores.append(scores['rouge1'])
        rouge2_scores.append(scores['rouge2'])
        rougeL_scores.append(scores['rougeL'])

        # Save examples
        if len(examples) < 5:
            examples.append({
                'input': tokenizer.decode(input_ids, skip_special_tokens=True),
                'prediction': prediction,
                'target': target,
            })

    return {
        'rouge1': sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0,
        'rouge2': sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0,
        'rougeL': sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0,
        'examples': examples,
    }