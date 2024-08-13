from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from grow import grow_depth, grow_width
from typing import Union, List

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GPTNeoXForCausalLM
from peft import PeftModel
from datasets import load_metric


def generate_instruction_answer(
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
        prompt: Union[str, List[str]],
        max_length: int = 128
    ) -> List[str]:
    """
    Run a generation loop till max length and do a weighted sum of the logits of each token
    before doing an argmax to get the token

    Args:
        model_1: First model
        model_2: Second model
        alpha: Interpolation ratio
    """
    # Tokenize the list of prompts
    tokenizer.pad_token = tokenizer.eos_token
    tokenized = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    input_ids = tokenized['input_ids']
    attn_mask = tokenized['attention_mask']

    gen_tokens_batch = []


    # Distribute inference to multiple GPUs if available
    model = torch.nn.DataParallel(model)


    for _ in range(max_length):
        model_pred = model(input_ids=input_ids, attention_mask=attn_mask)

        next_token_logits = model_pred[0][:, -1, :]

        next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # Append the generated tokens for all batches (batch_size, max_seq_len)
        gen_tokens_batch.append(next_tokens)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        attn_mask = torch.cat([attn_mask, torch.ones_like(attn_mask[:, :1])], dim=-1)

    # Decode the generated tokens
    gen_tokens_batch = torch.stack(gen_tokens_batch, dim=1)[:, :, 0]

    # Pad all tokens after eos with eos
    eos_mask = gen_tokens_batch == tokenizer.eos_token_id
    eos_mask = eos_mask.cumsum(dim=1) > 0

    gen_tokens_batch[eos_mask] = tokenizer.eos_token_id

    generated_text = tokenizer.batch_decode(gen_tokens_batch, skip_special_tokens=True)

    return generated_text


def generate_example_callback(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str):
        
        if not torch.cuda.is_available():
            model = model.to(torch.float32)

        # Output of model
        print('-'*50)
        print('Model output:')
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        output = generator(prompt, max_length=128, num_return_sequences=1, truncation=True, pad_token_id=tokenizer.eos_token_id)[0]['generated_text']
        print(output)



def load_model(model_name: str, path: str = None):
    """
    Load a model from a path and model name

    Args:
        model_name: Name of the model
        path: Path to the peft model
    """

    print(f"Loading model...")

    base_model = GPTNeoXForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        # torch_dtype=torch.float16,
        # device_map=0,
    )
    if path is not None:
        model = PeftModel.from_pretrained(base_model, path)
        model = model.merge_and_unload()
    else:
        model = base_model
    return model
    

def plot_losses(alphas, losses, save_path="interpolation_loss.png", **kwargs):

    title = kwargs.get('title', 'Model Evaluation Loss During Interpolation')
    xlabel = kwargs.get('xlabel', 'Interpolation Alpha (fraction of grown model)')
    ylabel = kwargs.get('ylabel', 'Loss')

    plt.figure(figsize=(10, 5))
    plt.plot(alphas, losses, marker='o', linestyle='-', color='b')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(save_path)


def flatten_rouge_scores(rouge_scores):
    flattened_scores = {}
    for score_type, aggregate_score in rouge_scores.items():
        for stat in ['precision', 'recall', 'fmeasure']:
            flattened_scores[f'{score_type}_low_{stat}'] = getattr(aggregate_score.low, stat)
            flattened_scores[f'{score_type}_mid_{stat}'] = getattr(aggregate_score.mid, stat)
            flattened_scores[f'{score_type}_high_{stat}'] = getattr(aggregate_score.high, stat)
    return flattened_scores

def evaluate_model_loss(model, test_dl, device, tokenizer):
    model.eval()
    total_loss = 0

    for batch in tqdm(test_dl):
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()

            # generated_ids = torch.argmax(outputs.logits, dim=-1)
            # generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            # reference_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
            # rouge.add_batch(predictions=generated_texts, references=reference_texts)
    
    avg_eval_loss = total_loss / len(test_dl)
    # final_rouge_scores = rouge.compute()
    # flattened_rouge_scores = flatten_rouge_scores(final_rouge_scores)
    print(f"Average Evaluation Loss: {avg_eval_loss}")
    # print("ROUGE Scores:", flattened_rouge_scores)

    # return avg_eval_loss, flattened_rouge_scores
    return avg_eval_loss


def evaluate_model_loss_rouge(model, test_dl, device, tokenizer):
    model.eval()
    total_loss = 0
    rouge = load_metric('rouge')

    for batch in tqdm(test_dl):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["labels"].detach().cpu().numpy()

        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()

            generated_ids = torch.argmax(outputs.logits, dim=-1)
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            reference_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
            rouge.add_batch(predictions=generated_texts, references=reference_texts)
    
    avg_eval_loss = total_loss / len(test_dl)
    final_rouge_scores = rouge.compute()
    flattened_rouge_scores = flatten_rouge_scores(final_rouge_scores)
    print(f"Average Evaluation Loss: {avg_eval_loss}")
    print("ROUGE Scores:", flattened_rouge_scores)

    return avg_eval_loss, flattened_rouge_scores
    # return flattened_rouge_scores
