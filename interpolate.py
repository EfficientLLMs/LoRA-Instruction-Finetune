from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, GPTNeoXForCausalLM, AutoTokenizer, pipeline
import torch
import copy
from datasets import load_metric
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from typing import Union, List

from grow import grow_depth, grow_width

def plot_losses(alphas, losses, save_path="interpolation_loss.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(alphas, losses, marker='o', linestyle='-', color='b')
    plt.title('Model Evaluation Loss During Interpolation')
    plt.xlabel('Interpolation Alpha (fraction of grown model)')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(save_path)

def load_model(path, model_name):
    print(f"Loading model...")

    base_model = GPTNeoXForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        # torch_dtype=torch.float16,
        # device_map=0,
    )
    model = PeftModel.from_pretrained(base_model, path)
    model = model.merge_and_unload()
    return model


def interpolate(model_1: AutoModelForCausalLM, model_2: AutoModelForCausalLM, alpha: float) -> AutoModelForCausalLM:
    """
    Interpolate between two models with the ratio of weights alpha * model_1 + (1 - alpha) * model_2

    Args:
        model_1: First model
        model_2: Second model
        alpha: Interpolation ratio
    """

    # Get the state_dict of the models
    state_dict_1 = model_1.state_dict()
    state_dict_2 = model_2.state_dict()

    # Interpolate between the weights - only change nn.Linear weights
    new_state_dict = {}
    for key in state_dict_1:
        new_state_dict[key] = alpha * state_dict_1[key] + (1 - alpha) * state_dict_2[key]

    # Create a copy of the first model and load the new state_dict
    model = copy.deepcopy(model_1)
    model.load_state_dict(new_state_dict)

    return model


def interpolate_logits(
        model_1: AutoModelForCausalLM, 
        model_2: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
        alpha: float, 
        prompt: Union[str, List[str]]
    ):
    """
    Run a generation loop till max length and do a weighted sum of the logits of each token
    before doing an argmax to get the token

    Args:
        model_1: First model
        model_2: Second model
        alpha: Interpolation ratio
    """
    # Tokenize the list of prompts

    # ValueError: Asking to pad but the tokenizer does not have a padding token. Please select a token to use as `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` or add a new pad token via `tokenizer.add_special_tokens({'pad_token': '[PAD]'})`
    tokenizer.pad_token = tokenizer.eos_token
    tokenized = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=512)
    input_ids = tokenized['input_ids']
    attn_mask = tokenized['attention_mask']

    gen_tokens_batch = []

    # Run a loop up to max_seq_len to make greedy predictions

    # Distribute inference to multiple GPUs if available
    model_1 = torch.nn.DataParallel(model_1)
    model_2 = torch.nn.DataParallel(model_2)

    
    for i in range(128):
        model_1_pred = model_1(input_ids=input_ids, attention_mask=attn_mask)
        model_2_pred = model_2(input_ids=input_ids, attention_mask=attn_mask)

        next_token_logits = alpha * model_1_pred[0][:, -1, :] + (1 - alpha) * model_2_pred[0][:, -1, :]

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


def interpolate_models_logits(model_1: PeftModel, model_2: PeftModel, n_interpolations: int, callback: callable, tokenizer: AutoTokenizer, prompt: str):
    """
    Interpolate between two models n_interpolation times and call the callback function with the 
    interpolated model. Define the ratios as equally spaced between 0 and 1 based on n_interpolations.
    For instance, if n_interpolations=5, the ratios will be [0, 0.25, 0.5, 0.75, 1]

    Args:
        model_1: First model
        model_2: Second model
        n_interpolations: Number of interpolations
        callback: Function to call with the interpolated model
    """

    for i in range(n_interpolations):
        alpha = i / (n_interpolations - 1)
        print('-'*50)
        print(f"Interpolating with alpha={alpha}")
        model_interpolated = interpolate_logits(model_1, model_2, tokenizer, alpha, prompt)
        callback(model_interpolated)

def interpolate_models_generation(model_1: PeftModel, model_2: PeftModel, n_interpolations: int, callback: callable):
    """
    Interpolate between two models n_interpolation times and call the callback function with the 
    interpolated model. Define the ratios as equally spaced between 0 and 1 based on n_interpolations.
    For instance, if n_interpolations=5, the ratios will be [0, 0.25, 0.5, 0.75, 1]

    Args:
        model_1: First model
        model_2: Second model
        n_interpolations: Number of interpolations
        callback: Function to call with the interpolated model
    """

    for i in range(n_interpolations):
        alpha = i / (n_interpolations - 1)
        print('-'*50)
        print(f"Interpolating with alpha={alpha}")
        model_interpolated = interpolate(model_1, model_2, alpha)
        callback(model_interpolated)

def interpolate_models_loss(model_1, model_2, n_interpolations, dataloader, device, tokenizer):
    losses = []
    alphas = []
    for i in range(n_interpolations):
        alpha = i / (n_interpolations - 1)
        alphas.append(alpha)
        print('-'*50)
        print(f"Interpolating with alpha={alpha}")
        model_interpolated = interpolate(model_1, model_2, alpha)
        loss_tracking_callback(model_interpolated, dataloader, device, tokenizer, losses)
    return losses, alphas

def generate_example_callback(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str):
    
    if not torch.cuda.is_available():
        model = model.to(torch.float32)

    # Output of model
    print('-'*50)
    print('Model output:')
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    output = generator(prompt, max_length=128, num_return_sequences=1, truncation=True, pad_token_id=tokenizer.eos_token_id)[0]['generated_text']
    print(output)

def flatten_rouge_scores(rouge_scores):
    flattened_scores = {}
    for score_type, aggregate_score in rouge_scores.items():
        for stat in ['precision', 'recall', 'fmeasure']:
            flattened_scores[f'{score_type}_low_{stat}'] = getattr(aggregate_score.low, stat)
            flattened_scores[f'{score_type}_mid_{stat}'] = getattr(aggregate_score.mid, stat)
            flattened_scores[f'{score_type}_high_{stat}'] = getattr(aggregate_score.high, stat)
    return flattened_scores

def evaluate_model(model, test_dl, device, tokenizer):
    model.eval()
    total_loss = 0
    # rouge = load_metric('rouge')

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

def loss_tracking_callback(model, dataloader, device, tokenizer, losses):
    model.to(device)
    loss = evaluate_model(model, dataloader, device, tokenizer)
    losses.append(loss)
    

if __name__ == "__main__":

    # Add command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_small", type=str, default="./output/70m_no_trainer/")
    parser.add_argument("--output_large", type=str, default="./output/410m_no_trainer/")
    parser.add_argument('--base_model_small', type=str, default="EleutherAI/pythia-70m")
    parser.add_argument('--base_model_large', type=str, default="EleutherAI/pythia-410m")
    parser.add_argument('--n_interpolations', type=int, default=5)

    args = parser.parse_args()

    # Get fine-tuned 70m model
    model_70m = "EleutherAI/pythia-70m"
    # path_lora_70m = "models/70m_no_trainer"
    path_lora_70m = "output/70m/no_trainer"
    model_70m_ft = load_model(path_lora_70m, model_70m)
    

    # Get fine-tuned 410m model
    model_410m = "EleutherAI/pythia-410m"
    # path_lora_410m = "models/410m_no_trainer"
    path_lora_410m = "output/410m/no_trainer"
    model_410m_ft = load_model(path_lora_410m, model_410m)

    # Get the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_70m)

    # Interpolate logits
    # prompt = [
    #     "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWrite three advantages of fruits\n\n### Response:\n",
    #     "Complete the following sentence: The best way to learn is to \n",
    # ]

    # interpolate_logits(model_70m_ft, model_410m_ft, tokenizer, 0, prompt)
    
    prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWrite three advantages of fruits\n\n### Response:\n"
    # interpolate_logits(model_70m_ft, model_410m_ft, tokenizer, 0.5, prompt)

    interpolate_models_logits(model_70m_ft, model_410m_ft, args.n_interpolations, print, tokenizer, prompt)

    # TODO: remove hard-coding here

    # Depth from 6 -> 24
    # intermediate_grown = grow_depth.expand_layers(model_70m_ft, 6, 12, expand_type='alternate')
    # intermediate_grown = grow_depth.expand_layers(intermediate_grown, 12, 24, expand_type='alternate')

    # # Width from 512 -> 1024
    # model_70m_ft_grown_410m = grow_width.expand_width(intermediate_grown, 512, 1024)

    # print("Model growth complete")

    # # Define a prompt
    # # prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWrite three advantages of fruits\n\n### Response:\n"

    # # Tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(model_70m)
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "right"

    # # Define a partial function for the generate_example_callback
    # # partial_callback = lambda model: generate_example_callback(model, tokenizer, prompt)

    # # Interpolate 2 times between 70m and 410m models
    # # interpolate_models_generation(model_70m_ft_grown_410m, model_410m_ft, 5, partial_callback)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # eval_dataloader = torch.load("data/eval_dataloader.pt")
    # losses, alphas = interpolate_models_loss(model_70m_ft_grown_410m, model_410m_ft, 5, eval_dataloader, device, tokenizer)
    # plot_losses(alphas, losses, save_path='interpolation_loss_fine_tuned.png')
    
