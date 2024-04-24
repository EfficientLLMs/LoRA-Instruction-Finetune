from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from grow import grow_depth, grow_width

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_metric

def generate_example_callback(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str, pipeline: pipeline):
        
        if not torch.cuda.is_available():
            model = model.to(torch.float32)

        # Output of model
        print('-'*50)
        print('Model output:')
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        output = generator(prompt, max_length=128, num_return_sequences=1, truncation=True, pad_token_id=tokenizer.eos_token_id)[0]['generated_text']
        print(output)



    

def plot_losses(alphas, losses, save_path="interpolation_loss.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(alphas, losses, marker='o', linestyle='-', color='b')
    plt.title('Model Evaluation Loss During Interpolation')
    plt.xlabel('Interpolation Alpha (fraction of grown model)')
    plt.ylabel('Loss')
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


def evaluate_model_rouge(model, test_dl, device, tokenizer):
    model.eval()
    total_loss = 0
    rouge = load_metric('rouge')

    for batch in tqdm(test_dl):
        batch = {k: v.to(device) for k, v in batch.items()}
        # labels = batch["labels"].detach().cpu().numpy()

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
