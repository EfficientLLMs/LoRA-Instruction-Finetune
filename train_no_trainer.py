import random
import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator, GPTNeoXForCausalLM
from peft import LoraConfig, get_peft_model
import json
from torch.utils.data import DataLoader
from tqdm import tqdm


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def prompt_no_input(row):
    return ("Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:\n").format_map(row)

def prompt_with_input(row):
    return ("Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n").format_map(row)

def create_prompt(row):
    return prompt_no_input(row) if row["input"] == "" else prompt_with_input(row)

def pack(dataset, max_seq_len, tokenizer):
    tkds_ids = tokenizer([s["example"] for s in dataset])["input_ids"]
    
    all_token_ids = []
    for tokenized_input in tkds_ids:
        all_token_ids.extend(tokenized_input + [tokenizer.eos_token_id])
    
    packed_ds = []
    for i in range(0, len(all_token_ids), max_seq_len+1):
        input_ids = all_token_ids[i : i + max_seq_len+1]
        if len(input_ids) == (max_seq_len+1):
            packed_ds.append({"input_ids": input_ids, "labels": input_ids})
    return packed_ds

def get_dataset(args, tokenizer):
    with open(args.data, "r") as f:
        alpaca = json.load(f)
    prompts = [create_prompt(row) for row in alpaca]
    outputs = [row['output'] + tokenizer.eos_token for row in alpaca]
    dataset = [{"prompt":s, "output":t, "example": s+t} for s, t in zip(prompts, outputs)]
    random.shuffle(dataset)
    # dataset = dataset[:5000]
    train_size = int(args.ratio * len(dataset))
    train_dataset = dataset[:train_size]
    eval_dataset = dataset[train_size:]
    train_ds_packed = pack(train_dataset, args.max_seq_length, tokenizer)
    eval_ds_packed = pack(eval_dataset, args.max_seq_length, tokenizer)
    return train_ds_packed, eval_ds_packed

def evaluate_model(model, test_dl, device):
    model.eval()
    total_loss = 0

    for batch in test_dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
    
    avg_eval_loss = total_loss / len(test_dl)
    print(f"Average Evaluation Loss: {avg_eval_loss}")

def train_model(model, train_dl, test_dl, epochs, lr, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    print("Start training...")

    for epoch in tqdm(range(epochs)):
        total_loss = 0
        for batch in tqdm(train_dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dl)
        print(f"Epoch {epoch} - Average Training Loss: {avg_train_loss}")

        evaluate_model(model, test_dl, device)
        torch.cuda.empty_cache()

    print("Training finished...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1006)
    parser.add_argument("--model", type=str, default="NousResearch/Llama-2-7b-hf")
    parser.add_argument("--data", type=str, default="./data/alpaca_gpt4_data.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--ratio", type=float, default=0.8)
    parser.add_argument("--output", type=str, default="./output/160m/no_trainer/")
    args = parser.parse_args()

    seed_everything(args.seed)

    # tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-160m-deduped",
        revision="step143000",
        cache_dir="./pythia-160m-deduped/step143000",
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_ds_packed, eval_ds_packed = get_dataset(args, tokenizer)
    train_dataloader = DataLoader(
        train_ds_packed,
        batch_size=args.batch_size,
        collate_fn=default_data_collator,
    )
    eval_dataloader = DataLoader(
        eval_ds_packed,
        batch_size=args.batch_size,
        collate_fn=default_data_collator,
        shuffle=False,
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        # target_modules=["q_proj", "k_proj", "v_proj"],
        target_modules=["query_key_value"],
        bias="none",
        task_type='CAUSAL_LM',
    )
    # model = AutoModelForCausalLM.from_pretrained(args.model, device_map=args.device)
    model = GPTNeoXForCausalLM.from_pretrained(
        "EleutherAI/pythia-160m-deduped",
        revision="step143000",
        cache_dir="./pythia-160m-deduped/step143000",
        device_map=args.device,
    )
    model = get_peft_model(model, lora_config)

    train_model(model, train_dataloader, eval_dataloader, args.epochs, args.lr, args.device)

    model.save_pretrained(args.output)