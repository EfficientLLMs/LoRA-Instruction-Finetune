import random
import os
import torch
import argparse
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from datasets import load_metric
import wandb


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

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
    rouge = load_metric('rouge')

    for batch in test_dl:
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

def train_model(model, train_dl, test_dl, epochs, lr, device, tokenizer, output):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    print("Start training...")

    for epoch in tqdm(range(epochs)):
        total_loss = 0
        for batch in tqdm(train_dl):
            optimizer.zero_grad()

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dl)
        print(f"Epoch {epoch} - Average Training Loss: {avg_train_loss}")
        wandb.log({"epoch": epoch, "training_loss": avg_train_loss})

        eval_loss, eval_rouge_scores = evaluate_model(model, test_dl, device, tokenizer)
        wandb.log({"epoch": epoch, "evaluation_loss": eval_loss, **eval_rouge_scores})

        model.save_pretrained(output)
    print("Training finished...")


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1006)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", type=str, default="./output/410m/no_trainer/")
    args = parser.parse_args()

    # seed
    seed_everything(args.seed)

    # dataloader
    train_dataloader = torch.load("./data/train_dataloader.pt")
    eval_dataloader = torch.load("./data/eval_dataloader.pt")

    # wandb
    wandb.init(project="lora-instruction-finetune", entity="irisiris")
    wandb.config = {
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "batch_size": train_dataloader.batch_size,
        "seed": args.seed
    }

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-410m",  # standard model; the same tokenizer is used for all models
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # lora and model
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["query_key_value"],
        bias="none",
        task_type='CAUSAL_LM',
    )
    model = GPTNeoXForCausalLM.from_pretrained(
        "EleutherAI/pythia-410m",
        device_map=args.device,
    )
    model = get_peft_model(model, lora_config)

    # training and save
    train_model(model, train_dataloader, eval_dataloader, args.epochs, args.lr, args.device, tokenizer, args.output)
    wandb.finish()