import random
import os
import torch
import argparse
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm
import evaluate
import wandb


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def evaluate_model(model, test_dl, device, tokenizer):
    model.eval()
    total_loss = 0
    rouge = evaluate.load('rouge')

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
    print(f"Average Evaluation Loss: {avg_eval_loss}")
    print("ROUGE Scores:", final_rouge_scores)

    return avg_eval_loss, final_rouge_scores

def train_model(model, train_dl, test_dl, epochs, lr, device, tokenizer, output):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-6)
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

        scheduler.step()

        avg_train_loss = total_loss / len(train_dl)
        print(f"Epoch {epoch} - Average Training Loss: {avg_train_loss}")

        eval_loss, eval_rouge_scores = evaluate_model(model, test_dl, device, tokenizer)
        wandb.log({
            "epoch": epoch, 
            "learning_rate": scheduler.get_last_lr()[0],
            "training_loss": avg_train_loss,
            "evaluation_loss": eval_loss, 
            **eval_rouge_scores
        })

        model.save_pretrained(output)
    print("Training finished...")


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1006)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", type=str, default="./weight/pythia_410m_r=64/")
    args = parser.parse_args()

    # seed
    seed_everything(args.seed)

    # dataloader
    train_dataloader = torch.load("./data/train_dataloader.pt")
    eval_dataloader = torch.load("./data/eval_dataloader.pt")

    # wandb
    wandb.init(
        name="410m_r=64_1e-4_schedule",
        project="lora-instruction-finetune", 
        entity="vibhamasti"
    )
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
    tokenizer.padding_side = "left"

    
    # larger base model + expanded adapter
    base_model = GPTNeoXForCausalLM.from_pretrained(
        "EleutherAI/pythia-410m",
        device_map=args.device,
    )
    config_lora = LoraConfig(
        r=64, 
        target_modules=["query_key_value"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, config_lora)

    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
        print(name, param.requires_grad)

    # training and save
    train_model(model, train_dataloader, eval_dataloader, args.epochs, args.lr, args.device, tokenizer, args.output)
    wandb.finish()